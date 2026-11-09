"""Video/photos -> textured 3D mesh.

Pipeline:
  1. frames   - sample sharp frames from videos (ffmpeg) and add photos
  2. sparse   - COLMAP: features, matching, camera poses (structure-from-motion)
  3. dense    - OpenMVS: dense point cloud, mesh, texture
               (fallback: COLMAP patch-match + Poisson, CUDA only)
  4. cleanup  - drop debris and the table, put it on the bed, export GLB

External tools are found on PATH, through FRAM3_COLMAP / FRAM3_OPENMVS_DIR, or
run in a container image when FRAM3_CONTAINER is set (e.g. "podman").
"""
from __future__ import annotations

import json
import math
import re
import os
import shlex
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter
from struct import error as struct_error

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".3gp"}
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic"}

PRESETS = {
    # max image side, target frame count, OpenMVS resolution level (0 = full)
    "snel":   {"max_side": 1600, "frames": 80,  "mvs_level": 2},
    "normaal": {"max_side": 2400, "frames": 150, "mvs_level": 1},
    "hoog":   {"max_side": 3200, "frames": 250, "mvs_level": 0},
}


class Cancelled(Exception):
    pass


class ToolMissing(RuntimeError):
    pass


@dataclass
class Tools:
    colmap: list[str] | None
    openmvs_dir: str | None
    cuda: bool

    @staticmethod
    def detect() -> "Tools":
        colmap = os.environ.get("FRAM3_COLMAP") or shutil.which("colmap")
        mvs_dir = os.environ.get("FRAM3_OPENMVS_DIR")
        if not mvs_dir:
            for cand in ("/usr/local/bin/OpenMVS", "/usr/bin/OpenMVS", "/usr/local/bin", "/usr/bin"):
                if Path(cand, "DensifyPointCloud").exists():
                    mvs_dir = cand
                    break
        cuda = shutil.which("nvidia-smi") is not None and subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True).returncode == 0
        return Tools(colmap.split() if colmap else None, mvs_dir, cuda)

    def summary(self) -> dict:
        return {"colmap": bool(self.colmap), "openmvs": bool(self.openmvs_dir),
                "cuda": self.cuda, "ffmpeg": bool(shutil.which("ffmpeg"))}


@dataclass
class Job:
    id: str
    workdir: Path
    preset: str = "normaal"
    remove_floor: bool = True    # cut away the table/room around the object
    status: str = "wachtrij"     # wachtrij | bezig | klaar | fout | geannuleerd
    stage: str = ""
    progress: float = 0.0
    log: list[str] = field(default_factory=list)
    result: str | None = None    # path of result.glb
    error: str | None = None
    warning: str | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _proc: subprocess.Popen | None = field(default=None, repr=False)

    def save(self):
        """Persist enough to list and resume this job after a restart."""
        try:
            (self.workdir / "job.json").write_text(json.dumps({
                "id": self.id, "preset": self.preset, "remove_floor": self.remove_floor, "status": self.status, "stage": self.stage,
                "progress": self.progress, "error": self.error, "warning": self.warning,
                "result": self.result}))
        except OSError:
            pass

    @staticmethod
    def load(workdir: Path) -> "Job":
        info = json.loads((workdir / "job.json").read_text())
        job = Job(info["id"], workdir, info.get("preset", "normaal"), info.get("remove_floor", True))
        job.status, job.stage = info.get("status", "fout"), info.get("stage", "")
        job.progress, job.error, job.result = info.get("progress", 0.0), info.get("error"), info.get("result")
        job.warning = info.get("warning")
        if job.status in ("bezig", "wachtrij"):
            # The server stopped (laptop off, crash) while this ran.
            job.status, job.error = "onderbroken", "De scan is onderbroken. Je kunt hem hervatten."
        if job.result and not Path(job.result).exists():
            job.result = None
        return job

    def inputs(self) -> list[Path]:
        return sorted(p for p in (self.workdir / "input").glob("*") if p.is_file())

    def public(self) -> dict:
        return {"id": self.id, "status": self.status, "stage": self.stage,
                "progress": round(self.progress, 3), "log": self.log[-200:],
                "error": self.error, "warning": self.warning, "has_result": self.result is not None,
                "preset": self.preset}

    def cancel(self):
        self._cancel.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


STAGES = [("frames", 0.10), ("features", 0.10), ("matching", 0.15), ("sparse", 0.15),
          ("dense", 0.30), ("mesh", 0.10), ("texture", 0.05), ("cleanup", 0.05)]
STAGE_NL = {"frames": "Frames kiezen", "features": "Kenmerken zoeken", "matching": "Beelden koppelen",
            "sparse": "Camera's bepalen", "dense": "Dichte puntenwolk", "mesh": "Mesh bouwen",
            "texture": "Textuur", "cleanup": "Opschonen"}


class Reconstructor:
    def __init__(self, job: Job, inputs: list[Path], tools: Tools | None = None):
        self.job = job
        self.inputs = inputs
        self.tools = tools or Tools.detect()
        self.p = PRESETS.get(job.preset, PRESETS["normaal"])
        self.remove_floor = job.remove_floor
        self.w = job.workdir

    # ---------------------------------------------------------------- utils
    def log(self, msg: str):
        self.job.log.append(msg)

    def stage(self, name: str, frac: float = 0.0):
        if self.job._cancel.is_set():
            raise Cancelled()
        self.progress(name, frac)

    def progress(self, name: str, frac: float):
        frac = max(0.0, min(1.0, frac))
        done = 0.0
        for n, weight in STAGES:
            if n == name:
                self.job.progress = done + weight * frac
                break
            done += weight
        if self.job.stage != STAGE_NL[name]:
            self.job.stage = STAGE_NL[name]
            self.log(f"== {STAGE_NL[name]} ==")
            self.job.save()

    def run(self, cmd: list[str], cwd: Path | None = None, on_line: Callable[[str], None] | None = None):
        if self.job._cancel.is_set():
            raise Cancelled()
        self.log("$ " + " ".join(shlex.quote(c) for c in cmd))
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace")
        self.job._proc = proc
        tail: list[str] = []
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail = (tail + [line])[-40:]
                if len(self.job.log) < 20000:
                    self.job.log.append("  " + line[:300])
                if on_line:
                    try:
                        on_line(line)
                    except Exception:
                        pass
        proc.wait()
        self.job._proc = None
        if self.job._cancel.is_set():
            raise Cancelled()
        if proc.returncode != 0:
            raise RuntimeError(f"{Path(cmd[0]).name} faalde (code {proc.returncode}):\n" + "\n".join(tail[-10:]))

    def colmap(self, *args: str, on_line: Callable[[str], None] | None = None):
        if not self.tools.colmap:
            raise ToolMissing("COLMAP is niet gevonden. Draai scripts/install-reconstruction.sh.")
        self.run(self.tools.colmap + list(args), on_line=on_line)

    def counter(self, stage: str, pattern: str, total: int | None = None):
        """Progress callback: pattern captures 'n' or 'n' and 'total' from tool output."""
        rx = re.compile(pattern)

        def cb(line: str):
            m = rx.search(line)
            if not m:
                return
            n = int(m.group(1))
            t = int(m.group(2)) if m.lastindex and m.lastindex >= 2 else total
            if t:
                self.progress(stage, n / t)
        return cb

    def mvs(self, tool: str, *args: str):
        self.run([str(Path(self.tools.openmvs_dir, tool))] + list(args), cwd=self.w / "mvs")

    # ------------------------------------------------------------- pipeline
    def execute(self) -> Path:
        """Run the pipeline. Stages whose output already exists in the job folder
        (from an interrupted run) are skipped, so a scan can be resumed."""
        images = self.w / "images"
        sparse_done = (self.w / "dense" / "sparse").is_dir() and (self.w / "dense" / "images").is_dir()
        if sparse_done and images.is_dir():
            # Do not resume on a camera model from an older version that placed
            # only a few of the frames: that never gives a usable scan.
            from .align import read_cameras
            try:
                placed = len(read_cameras(self.w / "dense" / "sparse")[0])
            except Exception:
                placed = 0
            total = len(list(images.iterdir()))
            if total and placed < 0.3 * total:
                self.log(f"Vorige poging plaatste maar {placed} van {total} beelden: opnieuw vanaf het begin")
                sparse_done = False
                for f in (self.w / "mvs").glob("*") if (self.w / "mvs").is_dir() else []:
                    if f.is_file():
                        f.unlink()
        if sparse_done:
            self.log("Hervatten: frames en camera's zijn al klaar")
        else:
            shutil.rmtree(images, ignore_errors=True)
            for d in ("sparse", "dense", "cameras"):
                shutil.rmtree(self.w / d, ignore_errors=True)
            (self.w / "colmap.db").unlink(missing_ok=True)
            images.mkdir(parents=True, exist_ok=True)
            n = self.collect_frames(images)
            if n < 8:
                raise RuntimeError(f"Te weinig bruikbare beelden ({n}). Film rustig rondom het object, minstens 20 beelden.")
            self.log(f"{n} beelden klaar voor reconstructie")
            self.sparse(images, sequential=self.has_video())
        mesh = self.dense_and_mesh()
        result = self.cleanup(mesh)
        self.tidy()
        return result

    def tidy(self):
        """Delete intermediate files once the scan is done (depth maps, frames,
        undistorted images: several GB). Keeps the input, the textured OpenMVS
        mesh and result.glb. FRAM3_KEEP_WORK=1 keeps everything for debugging."""
        if os.environ.get("FRAM3_KEEP_WORK") == "1":
            return
        before = _du(self.w)
        # Keep the camera model (under 1 MB): cleanup needs it to set a scan
        # upright, so a finished scan can always be cleaned again.
        cams = self.w / "cameras"
        if (self.w / "dense" / "sparse").is_dir() and not cams.exists():
            shutil.copytree(self.w / "dense" / "sparse", cams)
        for d in list(self.w.glob("cand*")) + [self.w / "images", self.w / "dense", self.w / "sparse"]:
            shutil.rmtree(d, ignore_errors=True)
        (self.w / "colmap.db").unlink(missing_ok=True)
        mvs = self.w / "mvs"
        keep = ("scene_textured",)
        for f in mvs.glob("*") if mvs.exists() else []:
            if f.is_file() and not f.name.startswith(keep):
                f.unlink(missing_ok=True)
        after = _du(self.w)
        self.log(f"  Tussenbestanden opgeruimd: {before / 1e9:.1f} GB → {after / 1e9:.2f} GB")

    def has_video(self) -> bool:
        return any(p.suffix.lower() in VIDEO_EXT for p in self.inputs)

    # 1. frames ---------------------------------------------------------------
    def collect_frames(self, out: Path) -> int:
        self.stage("frames")
        videos = [p for p in self.inputs if p.suffix.lower() in VIDEO_EXT]
        photos = [p for p in self.inputs if p.suffix.lower() in PHOTO_EXT]
        per_video = max(10, self.p["frames"] // max(1, len(videos))) if videos else 0
        idx = 0
        for vi, v in enumerate(videos):
            for f in self.sharp_frames(v, per_video, vi):
                idx += 1
                self.save_image(f, out / f"f{idx:05d}.jpg")
            self.stage("frames", (vi + 1) / (len(videos) + 1))
        for ph in photos:
            try:
                img = _open_image(ph)
            except Exception as e:
                self.log(f"Foto overgeslagen ({ph.name}): {e}")
                continue
            idx += 1
            self.save_image(img, out / f"p{idx:05d}.jpg")
        return idx

    def sharp_frames(self, video: Path, want: int, vi: int) -> list[Image.Image]:
        """Sample 3x more frames than needed, keep the sharpest one in each window."""
        dur = _probe_duration(video)
        # More than ~8 frames per second of video adds near-duplicates: lots of
        # extra work (and memory) for no extra detail.
        want = max(10, min(want, int(math.ceil(dur * 8))))
        cand_dir = self.w / f"cand{vi}"
        cand_dir.mkdir(exist_ok=True)
        fps = max(0.5, min(30.0, (want * 3) / max(dur, 1.0)))
        self.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
                  "-vf", f"fps={fps:.4f},scale='min({self.p['max_side']},iw)':-2",
                  "-q:v", "2", str(cand_dir / "c%06d.jpg")])
        files = sorted(cand_dir.glob("c*.jpg"))
        if not files:
            return []
        scores = [_sharpness(f) for f in files]
        picked = []
        median = float(np.median(scores))
        # Exactly `want` evenly spaced windows; the sharpest frame of each wins.
        for chunk in np.array_split(np.arange(len(files)), min(want, len(files))):
            best = max(chunk, key=lambda j: scores[j])
            if scores[best] >= 0.35 * median:  # skip windows that are all motion blur
                picked.append(Image.open(files[best]).convert("RGB"))
        self.log(f"{video.name}: {len(files)} kandidaten, {len(picked)} scherpe frames gekozen")
        return picked

    def save_image(self, img: Image.Image, dst: Path):
        ms = self.p["max_side"]
        if max(img.size) > ms:
            img = img.copy()
            img.thumbnail((ms, ms), Image.LANCZOS)
        img.save(dst, quality=95)

    # 2. sparse -----------------------------------------------------------------
    def sparse(self, images: Path, sequential: bool):
        db = self.w / "colmap.db"
        sp = self.w / "sparse"
        sp.mkdir(exist_ok=True)
        gpu = "1" if self.tools.cuda else "0"
        n_imgs = len(list(images.iterdir()))
        small = max(Image.open(next(images.iterdir())).size) <= 1400
        self.stage("features")
        extra = []
        if small:
            # Phone/WhatsApp video is often only ~1000 px: lower the SIFT threshold
            # to find enough features on small, soft frames.
            extra = ["--SiftExtraction.peak_threshold", "0.004", "--SiftExtraction.max_num_features", "12000"]
            self.log("  Kleine beelden: gevoeligere kenmerkdetectie")
        self.colmap("feature_extractor", "--database_path", str(db), "--image_path", str(images),
                    "--ImageReader.single_camera", "1" if sequential else "0",
                    "--ImageReader.camera_model", "OPENCV",
                    "--SiftExtraction.use_gpu", gpu, *extra,
                    on_line=self.counter("features", r"Processed file \[(\d+)/(\d+)\]"))
        self.stage("matching")
        exhaustive = not (sequential and n_imgs > 120)
        self.match(db, gpu, exhaustive)
        self.stage("sparse")
        model, registered = self.map(db, images, sp, n_imgs)
        if registered < 0.6 * n_imgs and not exhaustive:
            # Sequential matching misses loops and revisits; match every pair and retry.
            self.log(f"Maar {registered} van {n_imgs} beelden geplaatst: opnieuw met alle beeldparen")
            self.stage("matching", 0.5)
            self.match(db, gpu, True)
            shutil.rmtree(sp, ignore_errors=True)
            sp.mkdir()
            self.stage("sparse")
            model, registered = self.map(db, images, sp, n_imgs)
        if registered < 0.6 * n_imgs:
            # No good starting pair: common with soft phone video and flat
            # scenes. A less strict start (smaller angle, fewer matches) lets the
            # mapper begin and then add the other images normally.
            self.log(f"Maar {registered} van {n_imgs} beelden geplaatst: opnieuw met soepelere start")
            shutil.rmtree(sp, ignore_errors=True)
            sp.mkdir()
            self.stage("sparse")
            model2, reg2 = self.map(db, images, sp, n_imgs, relaxed=True)
            if reg2 > registered:
                model, registered = model2, reg2
        self.registered = (registered, n_imgs)
        if registered < max(8, 0.3 * n_imgs):
            raise RuntimeError(
                f"Maar {registered} van de {n_imgs} beelden konden geplaatst worden, te weinig voor een goed model. "
                "Probeer de originele video (niet via WhatsApp, die maakt hem kleiner en waziger), "
                "film helemaal rondom het object met genoeg licht, of kies een andere kwaliteit.")
        if registered < 0.6 * n_imgs:
            self.job.warning = (f"Maar {registered} van de {n_imgs} beelden gebruikt: het model kan gaten hebben. "
                                "Film voor een beter resultaat helemaal rondom het object.")
            self.log("LET OP: " + self.job.warning)
        self.colmap("image_undistorter", "--image_path", str(images), "--input_path", str(model),
                    "--output_path", str(self.w / "dense"), "--output_type", "COLMAP",
                    "--max_image_size", str(self.p["max_side"]))

    def match(self, db: Path, gpu: str, exhaustive: bool):
        if exhaustive:
            self.colmap("exhaustive_matcher", "--database_path", str(db), "--SiftMatching.use_gpu", gpu,
                        on_line=self.counter("matching", r"Matching block \[(\d+)/(\d+)"))
        else:
            self.colmap("sequential_matcher", "--database_path", str(db),
                        "--SequentialMatching.overlap", "25",
                        "--SequentialMatching.quadratic_overlap", "1",
                        "--SiftMatching.use_gpu", gpu,
                        on_line=self.counter("matching", r"Matching image \[(\d+)/(\d+)\]"))

    def map(self, db: Path, images: Path, sp: Path, n_imgs: int, relaxed: bool = False) -> tuple[Path, int]:
        # "Registering image #12 (7)": the number in brackets is how many images are placed so far.
        extra = ["--Mapper.init_min_tri_angle", "4", "--Mapper.init_min_num_inliers", "50"] if relaxed else []
        self.colmap("mapper", "--database_path", str(db), "--image_path", str(images),
                    "--output_path", str(sp), *extra,
                    on_line=self.counter("sparse", r"Registering image #\d+ \((\d+)\)", total=n_imgs))
        from .align import read_cameras
        best, best_n = None, 0
        for d in sp.iterdir():
            if not d.is_dir():
                continue
            try:
                n = len(read_cameras(d)[0])
            except Exception:
                continue
            if n > best_n:
                best, best_n = d, n
        if best is None:
            raise RuntimeError("COLMAP kon geen camera-posities bepalen. Zorg voor meer overlap en textuur in de beelden.")
        self.log(f"{best_n} van {n_imgs} beelden geplaatst (model {best.name})")
        return best, best_n

    # 3. dense + mesh ---------------------------------------------------------
    def dense_and_mesh(self) -> Path:
        if self.tools.openmvs_dir:
            return self._openmvs()
        if self.tools.cuda:
            return self._colmap_dense()
        raise ToolMissing("Voor de dichte reconstructie is OpenMVS (werkt op CPU) of COLMAP met CUDA nodig. "
                          "Draai scripts/install-reconstruction.sh.")

    def _openmvs(self) -> Path:
        mvs = self.w / "mvs"
        mvs.mkdir(exist_ok=True)
        self.stage("dense")
        # Each step is skipped when its output exists (resume). DensifyPointCloud also
        # reuses depth maps (*.dmap) that an interrupted run already computed.
        if not (mvs / "scene.mvs").exists():
            self.mvs("InterfaceCOLMAP", "-i", str(self.w / "dense"), "-o", "scene.mvs",
                     "--image-folder", str(self.w / "dense" / "images"))
        if not (mvs / "scene_dense.mvs").exists():
            done = len(list(mvs.glob("*.dmap")))
            if done:
                self.log(f"Hervatten: {done} dieptekaarten waren al klaar")
            self.mvs("DensifyPointCloud", "scene.mvs", "--resolution-level", str(self.p["mvs_level"]),
                     "--max-threads", "0")
        self.stage("mesh")
        if not (mvs / "scene_mesh.ply").exists():
            self.mvs("ReconstructMesh", "scene_dense.mvs", "-p", "scene_dense.ply", "-o", "scene_mesh.mvs")
        self.stage("texture")
        if not (mvs / "scene_textured.obj").exists():
            self.mvs("TextureMesh", "scene_dense.mvs", "-m", "scene_mesh.ply", "-o", "scene_textured.mvs",
                         "--export-type", "obj", "--resolution-level", str(self.p["mvs_level"]),
                     "--empty-color", "8421504")  # unseen faces neutral grey instead of orange
        for cand in ("scene_textured.obj", "scene_textured.ply", "scene_mesh.ply"):
            if (mvs / cand).exists():
                return mvs / cand
        raise RuntimeError("OpenMVS leverde geen mesh op.")

    def _colmap_dense(self) -> Path:
        d = self.w / "dense"
        self.stage("dense")
        self.colmap("patch_match_stereo", "--workspace_path", str(d), "--PatchMatchStereo.geom_consistency", "true")
        self.colmap("stereo_fusion", "--workspace_path", str(d), "--output_path", str(d / "fused.ply"))
        self.stage("mesh")
        self.colmap("poisson_mesher", "--input_path", str(d / "fused.ply"), "--output_path", str(d / "meshed.ply"))
        return d / "meshed.ply"

    # 4. cleanup ---------------------------------------------------------------
    def gravity(self):
        """Rotation that makes the scan upright, plus the camera orbit (centre xy,
        radius) after that rotation. Falls back to COLMAP's default axes."""
        import trimesh
        from .align import gravity_rotation, look_at_point, read_poses
        for model in (self.w / "dense" / "sparse", self.w / "cameras",
                      *sorted((self.w / "sparse").glob("*"), key=lambda d: -_du(d))):
            try:
                centers, ups, fwds = read_poses(model)
            except (FileNotFoundError, OSError, struct_error):
                continue
            if len(centers) < 3:
                continue
            rot = gravity_rotation(centers, ups)
            c = trimesh.transform_points(centers, rot)
            target = look_at_point(centers, fwds)
            if target is not None:
                mid = trimesh.transform_points(target[None], rot)[0, :2]
            else:
                mid = c[:, :2].mean(axis=0)
            radius = float(np.median(np.linalg.norm(c[:, :2] - mid, axis=1)))
            self.log(f"  Rechtop gezet met {len(centers)} camerastanden")
            return rot, (mid, radius)
        self.log("  Geen cameragegevens: standaardoriëntatie gebruikt")
        return trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]), None

    def isolate_object(self, geoms, orbit_xy, orbit_r):
        """Remove the table and the room around the object (option remove_floor)."""
        import trimesh
        from .align import object_mask
        if not self.remove_floor:
            return
        sizes = [len(g.faces) for g in geoms]
        verts, faces, off = [], [], 0
        for g in geoms:
            verts.append(g.vertices)
            faces.append(g.faces + off)
            off += len(g.vertices)
        whole = trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=False)
        keep, info = object_mask(whole, orbit_xy, orbit_r, remove_floor=True)
        if not keep.any():
            self.log("  Object niet gevonden: niets weggesneden")
            return
        start = 0
        for g, n in zip(geoms, sizes):
            k = keep[start:start + n]
            start += n
            if not k.all() and k.any():
                g.update_faces(k)
                g.remove_unreferenced_vertices()
        if info.get("floor_removed"):
            self.log(f"  Ondergrond weggesneden ({info['floor_removed']:,} driehoeken)".replace(",", "."))
        if info.get("outside_removed"):
            self.log(f"  Omgeving buiten de camerakring weggesneden ({info['outside_removed']:,} driehoeken)".replace(",", "."))
        if info.get("isolate_skipped"):
            self.log("  Object niet duidelijk los te maken: losse delen rond het object gelaten")

    def cleanup(self, mesh_path: Path) -> Path:
        """Orient Z-up, drop floating debris and put the scan on the bed, keeping the texture."""
        import trimesh
        from .repair import stats
        self.stage("cleanup")
        scene = trimesh.load(str(mesh_path), force="scene")
        geoms = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
        if not geoms:
            raise RuntimeError("De reconstructie leverde een lege mesh op.")
        before = [stats(g) for g in geoms]
        rot, orbit = self.gravity()
        for g in geoms:
            g.apply_transform(rot)
        if orbit is not None:
            # Level exactly on the table the object stands on.
            from .align import table_tilt
            whole = trimesh.util.concatenate([trimesh.Trimesh(g.vertices, g.faces, process=False) for g in geoms])
            mid, radius = orbit
            near = np.linalg.norm(whole.triangles_center[:, :2] - mid, axis=1) < 0.8 * radius
            if near.sum() > 100:
                whole = whole.submesh([np.nonzero(near)[0]], append=True)
            tilt = table_tilt(whole)
            if not np.allclose(tilt, np.eye(4)):
                ang = np.degrees(np.arccos(np.clip(tilt[2, 2], -1, 1)))
                for g in geoms:
                    g.apply_transform(tilt)
                pts = np.array([[*mid, 0.0]])
                orbit = (trimesh.transform_points(pts, tilt)[0, :2], radius)
                self.log(f"  Waterpas gezet op de tafel ({ang:.1f}° bijgesteld)")
        if orbit is not None:
            self.isolate_object(geoms, *orbit)
        # Debris: small clusters far from everything else. Clusters are found by
        # 3D proximity (a scan can be many touching islands that together form
        # the object). Masking faces keeps UVs and the texture intact.
        from .align import proximity_labels
        cent = np.vstack([g.triangles_center for g in geoms])
        area = np.concatenate([g.area_faces for g in geoms])
        diag = float(np.linalg.norm(np.ptp(cent, axis=0)))
        labels = proximity_labels(cent, max(0.02 * diag, 3.0 * float(np.sqrt(np.median(area)))))
        keep_all = np.ones(len(cent), dtype=bool)
        removed = 0
        for lab in np.unique(labels):
            f = labels == lab
            if np.linalg.norm(np.ptp(cent[f], axis=0)) < 0.08 * diag and area[f].sum() < 0.02 * area.sum():
                keep_all[f] = False
                removed += 1
        if area[~keep_all].sum() > 0.3 * area.sum():
            # Mostly small pieces: that is the object itself in a fragmented scan.
            keep_all[:] = True
            removed = 0
        if keep_all.any() and not keep_all.all():
            start = 0
            for g in geoms:
                k = keep_all[start:start + len(g.faces)]
                start += len(g.faces)
                if k.any() and not k.all():
                    g.update_faces(k)
                    g.remove_unreferenced_vertices()
            geoms[:] = [g for g in geoms if len(g.faces)]
        if removed:
            self.log(f"  Losse stukjes verwijderd ({removed} weg)")
        lo = np.min([g.bounds[0] for g in geoms], axis=0)
        hi = np.max([g.bounds[1] for g in geoms], axis=0)
        # Rest on the bed by the real base, not by a few stray specks under it:
        # the lowest 0.2% of the surface does not count, and faces reaching
        # clearly below that are removed.
        cz = np.concatenate([g.triangles_center[:, 2] for g in geoms])
        ca = np.concatenate([g.area_faces for g in geoms])
        order = np.argsort(cz)
        cum = np.cumsum(ca[order]) / ca.sum()
        base = float(cz[order][np.searchsorted(cum, 0.002)])
        # allowance: a bit of the height, and never less than a few triangles
        allow = max(0.03 * (hi[2] - lo[2]), 3.0 * float(np.sqrt(np.median(ca))))
        if base - lo[2] > allow:
            for g in geoms:
                # drop faces that reach well below the base (long thin slivers too)
                k = g.vertices[g.faces][:, :, 2].min(axis=1) >= base - allow
                if k.any() and not k.all():
                    g.update_faces(k)
                    g.remove_unreferenced_vertices()
            geoms[:] = [g for g in geoms if len(g.faces)]
            lo = np.min([g.bounds[0] for g in geoms], axis=0)
            hi = np.max([g.bounds[1] for g in geoms], axis=0)
        shift = [-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]]
        for g in geoms:
            g.apply_translation(shift)
        self.log("  Op het printbed gezet")
        textured = any(getattr(g.visual, "kind", None) == "texture" for g in geoms)
        self.log("  Textuur behouden" if textured else "  Geen textuur gevonden (alleen vorm)")
        out = self.w / "result.glb"
        trimesh.Scene(geoms).export(out)
        (self.w / "report.json").write_text(json.dumps(
            {"before": before, "after": [stats(g) for g in geoms], "textured": textured}, indent=2))
        self.job.progress = 1.0
        return out


def run_job(job: Job, inputs: list[Path], tools: Tools | None = None):
    job.status, job.error = "bezig", None
    job.save()
    try:
        job.result = str(Reconstructor(job, inputs, tools).execute())
        job.status, job.stage = "klaar", "Klaar"
        job.log.append("Klaar! Het model staat in de editor.")
    except Cancelled:
        job.status = "geannuleerd"
        job.log.append("Geannuleerd.")
    except Exception as e:  # noqa: BLE001 - shown to the user
        job.status, job.error = "fout", str(e)
        job.log.append(f"FOUT: {e}")
    finally:
        job.save()


# ------------------------------------------------------------------ helpers

def _du(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _probe_duration(video: Path) -> float:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", str(video)], capture_output=True, text=True)
        return float(out.stdout.strip())
    except Exception:
        return 30.0


def _sharpness(path: Path) -> float:
    """Variance of the Laplacian on a small grayscale copy: higher = sharper."""
    img = Image.open(path).convert("L")
    img.thumbnail((640, 640))
    lap = np.asarray(img.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], 1, 128)),
                     dtype=np.float32)
    return float(lap.var())


def _open_image(path: Path) -> Image.Image:
    if path.suffix.lower() == ".heic":
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError as e:
            raise RuntimeError("HEIC-foto's vereisen 'pillow-heif'") from e
    img = Image.open(path)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")
