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
    status: str = "wachtrij"     # wachtrij | bezig | klaar | fout | geannuleerd
    stage: str = ""
    progress: float = 0.0
    log: list[str] = field(default_factory=list)
    result: str | None = None    # path of result.glb
    error: str | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _proc: subprocess.Popen | None = field(default=None, repr=False)

    def public(self) -> dict:
        return {"id": self.id, "status": self.status, "stage": self.stage,
                "progress": round(self.progress, 3), "log": self.log[-200:],
                "error": self.error, "has_result": self.result is not None}

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
        self.w = job.workdir

    # ---------------------------------------------------------------- utils
    def log(self, msg: str):
        self.job.log.append(msg)

    def stage(self, name: str, frac: float = 0.0):
        if self.job._cancel.is_set():
            raise Cancelled()
        done = 0.0
        for n, weight in STAGES:
            if n == name:
                self.job.progress = done + weight * frac
                break
            done += weight
        if self.job.stage != STAGE_NL[name]:
            self.job.stage = STAGE_NL[name]
            self.log(f"== {STAGE_NL[name]} ==")

    def run(self, cmd: list[str], cwd: Path | None = None):
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
        proc.wait()
        self.job._proc = None
        if self.job._cancel.is_set():
            raise Cancelled()
        if proc.returncode != 0:
            raise RuntimeError(f"{Path(cmd[0]).name} faalde (code {proc.returncode}):\n" + "\n".join(tail[-10:]))

    def colmap(self, *args: str):
        if not self.tools.colmap:
            raise ToolMissing("COLMAP is niet gevonden. Draai scripts/install-reconstruction.sh.")
        self.run(self.tools.colmap + list(args))

    def mvs(self, tool: str, *args: str):
        self.run([str(Path(self.tools.openmvs_dir, tool))] + list(args), cwd=self.w / "mvs")

    # ------------------------------------------------------------- pipeline
    def execute(self) -> Path:
        images = self.w / "images"
        images.mkdir(parents=True, exist_ok=True)
        n = self.collect_frames(images)
        if n < 8:
            raise RuntimeError(f"Te weinig bruikbare beelden ({n}). Film rustig rondom het object, minstens 20 beelden.")
        self.log(f"{n} beelden klaar voor reconstructie")
        self.sparse(images, sequential=self.has_video())
        mesh = self.dense_and_mesh()
        return self.cleanup(mesh)

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
        window = max(1, len(files) // want)
        picked = []
        median = float(np.median(scores))
        for i in range(0, len(files), window):
            chunk = list(range(i, min(i + window, len(files))))
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
        self.stage("features")
        self.colmap("feature_extractor", "--database_path", str(db), "--image_path", str(images),
                    "--ImageReader.single_camera", "1" if sequential else "0",
                    "--ImageReader.camera_model", "OPENCV",
                    "--SiftExtraction.use_gpu", gpu)
        self.stage("matching")
        n_imgs = len(list(images.iterdir()))
        if sequential and n_imgs > 60:
            self.colmap("sequential_matcher", "--database_path", str(db),
                        "--SequentialMatching.overlap", "15",
                        "--SequentialMatching.loop_detection", "0",
                        "--SiftMatching.use_gpu", gpu)
        else:
            self.colmap("exhaustive_matcher", "--database_path", str(db), "--SiftMatching.use_gpu", gpu)
        self.stage("sparse")
        self.colmap("mapper", "--database_path", str(db), "--image_path", str(images),
                    "--output_path", str(sp))
        models = sorted([d for d in sp.iterdir() if d.is_dir()],
                        key=lambda d: (d / "images.bin").stat().st_size if (d / "images.bin").exists() else 0,
                        reverse=True)
        if not models:
            raise RuntimeError("COLMAP kon geen camera-posities bepalen. Zorg voor meer overlap en textuur in de beelden.")
        self.log(f"Model gekozen: {models[0].name} ({len(models)} deelmodel(len))")
        self.colmap("image_undistorter", "--image_path", str(images), "--input_path", str(models[0]),
                    "--output_path", str(self.w / "dense"), "--output_type", "COLMAP",
                    "--max_image_size", str(self.p["max_side"]))

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
        self.mvs("InterfaceCOLMAP", "-i", str(self.w / "dense"), "-o", "scene.mvs",
                 "--image-folder", str(self.w / "dense" / "images"))
        self.mvs("DensifyPointCloud", "scene.mvs", "--resolution-level", str(self.p["mvs_level"]),
                 "--max-threads", "0")
        self.stage("mesh")
        self.mvs("ReconstructMesh", "scene_dense.mvs", "-p", "scene_dense.ply", "-o", "scene_mesh.mvs")
        self.stage("texture")
        self.mvs("TextureMesh", "scene_dense.mvs", "-m", "scene_mesh.ply", "-o", "scene_textured.mvs",
                 "--export-type", "obj", "--resolution-level", str(self.p["mvs_level"]))
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
    def cleanup(self, mesh_path: Path) -> Path:
        import trimesh
        from .repair import RepairOptions, repair
        self.stage("cleanup")
        scene = trimesh.load(str(mesh_path), force="scene")
        geoms = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
        # COLMAP is Y-down/Z-forward; the editor and slicers are Z-up.
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
        cleaned, rep = repair(mesh, RepairOptions(keep_largest=True, remove_floor=False, fill_holes=False,
                                                   make_watertight=False, drop_to_bed=True))
        for s in rep.steps:
            self.log("  " + s)
        out = self.w / "result.glb"
        cleaned.export(out)
        (self.w / "report.json").write_text(json.dumps({"before": rep.before, "after": rep.after}, indent=2))
        self.job.progress = 1.0
        return out


def run_job(job: Job, inputs: list[Path], tools: Tools | None = None):
    job.status = "bezig"
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


# ------------------------------------------------------------------ helpers

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
