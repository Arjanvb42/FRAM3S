"""Live view of a running scan: the frame being worked on, the cameras found so
far and the point cloud as it fills in. Everything is read from the files the
tools write in the job folder, so it never slows the scan down."""
from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image

MAX_POINTS = 200_000
VIEW_RADIUS = 100.0   # the camera orbit is shown about this wide (mm on the bed)


# ------------------------------------------------------------------ frames
def frame_files(wd: Path) -> list[Path]:
    """Frames chosen so far; while ffmpeg is still sampling, its candidates."""
    for d in (wd / "images", wd / "dense" / "images"):
        if d.is_dir():
            files = sorted(d.glob("*.jpg"))
            if files:
                return files
    cands = sorted(wd.glob("cand*/c*.jpg"))
    return cands[:-1]   # the last one may still be written


def thumbnail(path: Path, side: int = 720) -> bytes:
    with Image.open(path) as im:
        im.draft("RGB", (side, side))
        im = im.convert("RGB")
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        return buf.getvalue()


# ------------------------------------------------------------------ clouds
def read_points3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP points3D.bin -> (N,3) positions, (N,3) uint8 colours."""
    data = path.read_bytes()
    (n,) = struct.unpack_from("<Q", data, 0)
    xyz = np.empty((n, 3), np.float64)
    rgb = np.empty((n, 3), np.uint8)
    off = 8
    for i in range(n):
        xyz[i] = struct.unpack_from("<3d", data, off + 8)
        rgb[i] = struct.unpack_from("<3B", data, off + 32)
        (track,) = struct.unpack_from("<Q", data, off + 43)
        off += 51 + 8 * track
    return xyz, rgb


_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1",
              "uint8": "u1", "char": "i1", "int8": "i1", "short": "<i2", "int16": "<i2",
              "ushort": "<u2", "uint16": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}


def read_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Vertices (and colours) of a binary little-endian PLY, without the faces."""
    with path.open("rb") as f:
        n, fields, in_vertex = 0, [], False
        if f.readline().strip() != b"ply":
            raise ValueError("geen PLY")
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY-kop onvolledig")
            w = line.decode("ascii", "replace").split()
            if not w:
                continue
            if w[0] == "format" and w[1] != "binary_little_endian":
                raise ValueError("alleen binaire PLY")
            if w[0] == "element":
                in_vertex = w[1] == "vertex"
                if in_vertex:
                    n = int(w[2])
            elif w[0] == "property" and in_vertex:
                if w[1] == "list":
                    raise ValueError("lijst in vertex")
                fields.append((w[2], _PLY_TYPES[w[1]]))
            elif w[0] == "end_header":
                break
        arr = np.fromfile(f, dtype=np.dtype(fields), count=n)
    if len(arr) < n:
        raise ValueError("PLY nog niet compleet")
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    names = arr.dtype.names
    rgb = None
    if {"red", "green", "blue"} <= set(names):
        rgb = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


def _sparse_models(wd: Path) -> list[Path]:
    """Camera models, most recent first: the final one, else the mapper's snapshots."""
    out = []
    for d in (wd / "dense" / "sparse", wd / "cameras"):
        if (d / "images.bin").exists() or (d / "images.txt").exists():
            out.append(d)
    live = [d for d in list((wd / "snap").glob("*")) + list((wd / "sparse").glob("*"))
            if (d / "images.bin").exists() and (d / "points3D.bin").exists()]
    out += sorted(live, key=lambda d: -(d / "images.bin").stat().st_mtime_ns)
    return out


def cloud_source(wd: Path) -> tuple[str, Path, Path | None] | None:
    """(kind, point file, camera model) of the best thing to show right now."""
    models = _sparse_models(wd)
    dense = wd / "mvs" / "scene_dense.ply"
    if dense.exists() and models:
        return "dense", dense, models[0]
    for m in models:
        if (m / "points3D.bin").exists():
            return "sparse", m / "points3D.bin", m
    return None


def cloud_version(src) -> str:
    kind, pts, model = src
    return f"{kind}:{pts.stat().st_mtime_ns}:{pts.stat().st_size}"


def build_cloud(src) -> bytes:
    """Binary blob for the viewer: 'F3LV', n_cams, n_pts (uint32), then cameras
    as float32 (pos, forward, up), points as float32 xyz, colours as uint8 rgb.
    Upright, centred on what the cameras look at, in bed millimetres."""
    import trimesh
    from .align import gravity_rotation, look_at_point, read_poses
    kind, pts_file, model = src
    centers, ups, fwds = read_poses(model)
    if len(centers) < 3:
        raise ValueError("te weinig camera's")
    if kind == "dense":
        xyz, rgb = read_ply_points(pts_file)
    else:
        xyz, rgb = read_points3d(pts_file)
    if rgb is None:
        rgb = np.full((len(xyz), 3), 200, np.uint8)
    if len(xyz) > MAX_POINTS:
        pick = np.random.default_rng(0).choice(len(xyz), MAX_POINTS, replace=False)
        xyz, rgb = xyz[pick], rgb[pick]
    rot = gravity_rotation(centers, ups, fwds)
    R = rot[:3, :3]
    c = trimesh.transform_points(centers, rot)
    target = look_at_point(centers, fwds)
    mid = trimesh.transform_points(target[None], rot)[0] if target is not None else c.mean(axis=0)
    radius = float(np.median(np.linalg.norm(c[:, :2] - mid[:2], axis=1))) or 1.0
    p = trimesh.transform_points(xyz, rot) if len(xyz) else np.zeros((0, 3))
    # Stray points far outside the orbit only make the view tiny.
    near = np.linalg.norm(p - mid, axis=1) < 3.0 * radius if len(p) else np.zeros(0, bool)
    p, rgb = p[near], rgb[near]
    core = p[np.linalg.norm(p[:, :2] - mid[:2], axis=1) < 0.6 * radius] if len(p) else p
    floor = float(np.percentile(core[:, 2], 2)) if len(core) > 20 else float(mid[2])
    origin = np.array([mid[0], mid[1], floor])
    s = VIEW_RADIUS / radius
    cams = np.hstack([(c - origin) * s, fwds @ R.T, ups @ R.T]).astype("<f4")
    p = ((p - origin) * s).astype("<f4")
    return (b"F3LV" + struct.pack("<II", len(cams), len(p)) + cams.tobytes() + p.tobytes()
            + np.ascontiguousarray(rgb).tobytes())


def state(wd: Path, stage_key: str, stage_frac: float) -> dict:
    files = frame_files(wd)
    n = len(files)
    depth = len(list((wd / "mvs").glob("*.dmap"))) if (wd / "mvs").is_dir() else 0
    if stage_key in ("features", "matching") and n:
        idx = min(n - 1, int(stage_frac * n))
    elif stage_key == "dense" and depth and n:
        idx = min(n - 1, depth - 1)
    else:
        idx = n - 1
    out = {"stage": stage_key, "frames": n, "frame": idx, "depth_maps": depth, "cloud": None}
    try:
        src = cloud_source(wd)
        if src:
            out["cloud"] = {"kind": src[0], "version": cloud_version(src)}
    except OSError:
        pass
    return out
