"""Drone footage: use the GPS of the drone for scale, gravity and matching.

Drone photos carry their position in EXIF (and DJI adds the height above the
take-off point in XMP). DJI videos carry it in a subtitle track or a .SRT file
next to the video ("video captions" on the drone). With those positions:

- frames are picked per metre flown instead of per second (hovering adds nothing),
- images are only matched with images taken nearby (fast on big sets),
- the camera model is aligned to the GPS positions, so the scan comes out in
  real metres with Z pointing up, and is then set to a print scale (1:100 ...).

Without GPS, or when the camera barely moved (a hand-held phone photo set also
has GPS), the normal object pipeline runs unchanged.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

EARTH_R = 6378137.0
MIN_SPREAD_M = 10.0          # less than this and the GPS noise (~2 m) dominates
SCALES = (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 25000, 50000)
GEO_FILE = "geo.json"        # image name -> [x east, y north, z up] in metres


# ------------------------------------------------------------------ reading GPS
def _rational(v) -> float:
    try:
        return float(v)
    except TypeError:
        return float(v[0]) / float(v[1])


def photo_gps(path: Path) -> tuple[float, float, float] | None:
    """(lat, lon, alt) of a photo. DJI's RelativeAltitude (height above take-off,
    barometric) is used when present: steadier than the GPS altitude."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            gps = im.getexif().get_ifd(0x8825)
    except Exception:
        return None
    if not gps or 2 not in gps or 4 not in gps:
        return None
    try:
        lat = sum(_rational(x) / 60 ** i for i, x in enumerate(gps[2]))
        lon = sum(_rational(x) / 60 ** i for i, x in enumerate(gps[4]))
        if gps.get(1) in ("S", b"S"):
            lat = -lat
        if gps.get(3) in ("W", b"W"):
            lon = -lon
        alt = _rational(gps[6]) if 6 in gps else 0.0
        if gps.get(5) in (1, b"\x01"):
            alt = -alt
    except (TypeError, ValueError, ZeroDivisionError, IndexError):
        return None
    if lat == 0 and lon == 0:
        return None
    try:
        head = path.open("rb").read(256 * 1024)
        m = re.search(rb'RelativeAltitude="?\s*([+-]?[\d.]+)', head)
        if m:
            alt = float(m.group(1))
    except OSError:
        pass
    return lat, lon, alt


_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->")


def parse_srt(text: str) -> list[tuple[float, float, float, float]]:
    """(seconds, lat, lon, alt) per subtitle block of a DJI .SRT file.

    Handles the newer "[latitude: 52.1] [longitude: 5.1] [rel_alt: 30.0 abs_alt: 45.2]"
    and the older "GPS(5.1,52.1,20) ... H 30.0m" / "GPS (5.1, 52.1, 20), D 12m, H 30m"."""
    out = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        m = _TIME.search(block)
        if not m:
            continue
        h, mi, s, frac = m.groups()
        t = int(h) * 3600 + int(mi) * 60 + int(s) + int(frac) / 10 ** len(frac)
        lat = re.search(r"latitude\s*:\s*([+-]?[\d.]+)", block)
        lon = re.search(r"longitude\s*:\s*([+-]?[\d.]+)", block)
        if lat and lon:
            la, lo = float(lat.group(1)), float(lon.group(1))
            alt = re.search(r"rel_alt\s*:\s*([+-]?[\d.]+)", block) or re.search(r"abs_alt\s*:\s*([+-]?[\d.]+)", block)
            al = float(alt.group(1)) if alt else 0.0
        else:
            g = re.search(r"GPS\s*\(\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*(?:,\s*([+-]?[\d.]+))?", block)
            if not g:
                continue
            lo, la = float(g.group(1)), float(g.group(2))   # DJI writes longitude first here
            hm = re.search(r"\bH\s*:?\s*([+-]?[\d.]+)\s*m", block)
            al = float(hm.group(1)) if hm else float(g.group(3) or 0.0)
        if la == 0 and lo == 0:
            continue
        out.append((t, la, lo, al))
    return out


def video_gps(video: Path, others: list[Path]) -> list[tuple[float, float, float, float]]:
    """GPS track of a drone video: a .SRT next to it (same name) or a subtitle
    track inside the file."""
    for p in others:
        if p.suffix.lower() == ".srt" and p.stem.lower() == video.stem.lower():
            try:
                track = parse_srt(p.read_text(errors="replace"))
            except OSError:
                track = []
            if track:
                return track
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
                            "-map", "0:s:0", "-f", "srt", "-"], capture_output=True, text=True, timeout=120)
        return parse_srt(r.stdout) if r.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        return []


# ------------------------------------------------------------------ geometry
def to_local(lla: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """lat/lon/alt rows -> metres east, north, up around `origin` (fine up to a few km)."""
    lla = np.asarray(lla, dtype=float).reshape(-1, 3)
    lat0 = math.radians(origin[0])
    x = np.radians(lla[:, 1] - origin[1]) * EARTH_R * math.cos(lat0)
    y = np.radians(lla[:, 0] - origin[0]) * EARTH_R
    return np.column_stack([x, y, lla[:, 2] - origin[2]])


def spread(xyz: np.ndarray) -> float:
    """Horizontal size of the flight in metres."""
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.ptp(np.asarray(xyz)[:, :2], axis=0)))


def track_at(track: list[tuple], times: np.ndarray) -> np.ndarray:
    """Interpolated lat/lon/alt of a GPS track at the given video times."""
    tr = np.array(track, dtype=float)
    order = np.argsort(tr[:, 0])
    tr = tr[order]
    return np.column_stack([np.interp(times, tr[:, 0], tr[:, i]) for i in (1, 2, 3)])


def pick_by_distance(pos: np.ndarray, scores: np.ndarray, want: int, min_ratio: float = 0.35) -> list[int]:
    """Indices of frames, about evenly spaced along the flight path (metres, not
    seconds): the sharpest frame of each stretch. Stretches that are all motion
    blur are skipped."""
    pos, scores = np.asarray(pos, float), np.asarray(scores, float)
    n = len(pos)
    if n == 0:
        return []
    step = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
    dist = np.cumsum(step)
    total = dist[-1]
    want = max(1, min(want, n))
    if total <= 1e-6:
        bins = np.array_split(np.arange(n), want)
    else:
        edges = np.linspace(0, total, want + 1)
        idx = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, want - 1)
        bins = [np.nonzero(idx == b)[0] for b in range(want)]
    median = float(np.median(scores))
    chosen = []
    for b in bins:
        if len(b) == 0:
            continue
        best = int(b[np.argmax(scores[b])])
        if scores[best] >= min_ratio * median:
            chosen.append(best)
    return chosen


def neighbour_pairs(names: list[str], xyz: np.ndarray, k: int = 30) -> list[tuple[str, str]]:
    """Image pairs worth matching: each image with its k nearest by position.
    Two orbits at different heights still pair up, because k reaches across."""
    xyz = np.asarray(xyz, float)
    n = len(names)
    pairs = set()
    if n < 2:
        return []
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    k = min(k, n - 1)
    for i in range(n):
        for j in np.argsort(d[i])[1:k + 1]:
            a, b = sorted((i, int(j)))
            pairs.add((a, b))
    return [(names[a], names[b]) for a, b in sorted(pairs)]


def print_scale(size_m: np.ndarray, bed_mm: float = 200.0) -> int:
    """The largest common model scale (1:N) at which the scan fits on the bed."""
    big = float(np.max(size_m)) * 1000.0
    for n in SCALES:
        if big / n <= bed_mm:
            return n
    return SCALES[-1]


# ------------------------------------------------------------------ pipeline steps
def plan(rec) -> dict | None:
    """GPS for every input, or None when this is not drone footage."""
    from .reconstruct import PHOTO_EXT, VIDEO_EXT
    videos = [p for p in rec.inputs if p.suffix.lower() in VIDEO_EXT]
    photos = [p for p in rec.inputs if p.suffix.lower() in PHOTO_EXT]
    tracks, fixes = {}, {}
    for v in videos:
        tr = video_gps(v, rec.inputs)
        if not tr:
            return None
        tracks[v] = tr
    for ph in photos:
        g = photo_gps(ph)
        if g is None:
            return None
        fixes[ph] = g
    every = [f for tr in tracks.values() for f in (x[1:] for x in tr)] + list(fixes.values())
    if not every:
        return None
    origin = np.array(every[0], dtype=float)
    if spread(to_local(np.array(every), origin)) < MIN_SPREAD_M:
        return None
    return {"origin": origin, "tracks": tracks, "photos": fixes, "videos": videos}


def collect_frames(rec, out: Path, geo: dict) -> int:
    """Like Reconstructor.collect_frames, but video frames are picked per metre
    flown, and every frame's GPS position goes to geo.json."""
    from PIL import Image
    rec.stage("frames")
    origin = geo["origin"]
    videos, photos = geo["videos"], geo["photos"]
    per_video = max(10, rec.p["frames"] // max(1, len(videos))) if videos else 0
    positions, idx = {}, 0
    for vi, v in enumerate(videos):
        files, fps = rec.frame_candidates(v, per_video, vi, density=4)
        if not files:
            continue
        times = np.arange(len(files)) / fps   # ffmpeg's fps filter starts at t=0
        pos = to_local(track_at(geo["tracks"][v], times), origin)
        from .reconstruct import _sharpness
        scores = [_sharpness(f) for f in files]
        chosen = pick_by_distance(pos, scores, per_video)
        rec.log(f"{v.name}: {spread(pos):.0f} m gevlogen, {len(files)} kandidaten, "
                f"{len(chosen)} beelden gekozen op afgelegde afstand")
        for j in chosen:
            idx += 1
            name = f"f{idx:05d}.jpg"
            with Image.open(files[j]) as im:
                rec.save_image(im.convert("RGB"), out / name)
            positions[name] = pos[j].tolist()
        rec.stage("frames", (vi + 1) / (len(videos) + 1))
    from .reconstruct import _open_image
    for ph, lla in photos.items():
        try:
            img = _open_image(ph)
        except Exception as e:
            rec.log(f"Foto overgeslagen ({ph.name}): {e}")
            continue
        idx += 1
        name = f"p{idx:05d}.jpg"
        rec.save_image(img, out / name)
        positions[name] = to_local(np.array(lla), origin)[0].tolist()
    (rec.w / GEO_FILE).write_text(json.dumps(positions))
    return idx


def load(workdir: Path) -> dict | None:
    try:
        return json.loads((workdir / GEO_FILE).read_text())
    except (OSError, ValueError):
        return None


def match(rec, db: Path, gpu: str):
    """Match each image only with the images taken closest to it."""
    geo = load(rec.w) or {}
    names = sorted(geo)
    pairs = neighbour_pairs(names, np.array([geo[n] for n in names]))
    lst = rec.w / "pairs.txt"
    lst.write_text("".join(f"{a} {b}\n" for a, b in pairs))
    rec.log(f"  {len(pairs)} beeldparen op basis van GPS (in plaats van {len(names) * (len(names) - 1) // 2})")
    rec.colmap("matches_importer", "--database_path", str(db), "--match_list_path", str(lst),
               "--match_type", "pairs", "--SiftMatching.use_gpu", gpu,
               on_line=rec.counter("matching", r"Matching block \[(\d+)/(\d+)"))


def align(rec, model: Path) -> Path:
    """Put the camera model on the GPS positions: metres, Z up. Returns the
    aligned model, or the input model when that fails."""
    from .align import read_cameras
    geo = load(rec.w) or {}
    ref = rec.w / "geo_ref.txt"
    ref.write_text("".join(f"{n} {x:.4f} {y:.4f} {z:.4f}\n" for n, (x, y, z) in geo.items()))
    out = rec.w / "sparse_geo"
    out.mkdir(exist_ok=True)
    try:
        rec.colmap("model_aligner", "--input_path", str(model), "--output_path", str(out),
                   "--ref_images_path", str(ref), "--ref_is_gps", "0",
                   "--alignment_max_error", "3")
        centers, _ = read_cameras(out)
        if len(centers) < 3:
            raise RuntimeError("leeg model")
    except Exception as e:
        rec.log(f"  Uitlijnen op GPS lukte niet ({str(e)[:80]}): scan zonder echte maat")
        return model
    (rec.w / "geo_aligned").write_text("1")
    rec.log("  Camera's op de GPS-posities gezet: de scan is in echte meters en staat rechtop")
    return out


def aligned(workdir: Path) -> bool:
    return (workdir / "geo_aligned").exists()
