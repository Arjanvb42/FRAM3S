"""Put a scan upright and cut it loose from the table it stood on.

COLMAP's world frame is arbitrary, so a fixed rotation leaves scans tilted or
on their side. Instead the cameras tell us where "up" is: people hold the
phone upright and walk around the object, so

  * the average camera up-vector points roughly against gravity, and
  * the camera positions lie in a band whose normal is the vertical axis.

After aligning, the table shows up as the large horizontal surface under the
object, and the object as the mesh part in the middle of the camera orbit.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _qvec_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def read_cameras(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Camera centres (N,3) and up-vectors (N,3) in world coordinates from a
    COLMAP model folder (images.bin or images.txt)."""
    centers, ups, _ = read_poses(model_dir)
    return centers, ups


def read_poses(model_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera centres, up-vectors and viewing directions in world coordinates."""
    rots, ts = [], []
    binf, txtf = model_dir / "images.bin", model_dir / "images.txt"
    if binf.exists():
        with binf.open("rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            for _ in range(n):
                _iid, *rest = struct.unpack("<I4d3dI", f.read(4 + 32 + 24 + 4))
                q, t = rest[0:4], rest[4:7]
                while f.read(1) not in (b"\x00", b""):
                    pass
                (npts,) = struct.unpack("<Q", f.read(8))
                f.seek(npts * 24, 1)
                rots.append(_qvec_to_R(q))
                ts.append(t)
    elif txtf.exists():
        # Two lines per image (pose, then 2D points, which may be empty).
        lines = [l for l in txtf.read_text().splitlines() if not l.startswith("#")]
        for l in lines[::2]:
            if not l.strip():
                continue
            v = l.split()
            rots.append(_qvec_to_R([float(x) for x in v[1:5]]))
            ts.append([float(x) for x in v[5:8]])
    else:
        raise FileNotFoundError(f"Geen COLMAP-model in {model_dir}")
    R = np.array(rots)
    t = np.array(ts)
    centers = -np.einsum("nji,nj->ni", R, t)          # C = -R^T t
    ups = np.einsum("nji,j->ni", R, np.array([0.0, -1.0, 0.0]))  # camera -Y in world
    fwds = np.einsum("nji,j->ni", R, np.array([0.0, 0.0, 1.0]))  # camera +Z in world
    return centers, ups, fwds


def look_at_point(centers: np.ndarray, fwds: np.ndarray) -> np.ndarray | None:
    """The point all cameras look at: least-squares intersection of their
    viewing rays. Better than the mean camera position when the orbit is not a
    full circle (a 270 degree arc pulls the mean off-centre)."""
    if len(centers) < 3:
        return None
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centers, fwds):
        d = d / (np.linalg.norm(d) or 1.0)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    if np.linalg.cond(A) > 1e6:   # all rays parallel: no single target
        return None
    p = np.linalg.solve(A, b)
    # must lie in front of most cameras
    if np.mean(np.einsum("ni,ni->n", p - centers, fwds) > 0) < 0.7:
        return None
    return p


def gravity_rotation(centers: np.ndarray, ups: np.ndarray) -> np.ndarray:
    """4x4 rotation that turns the scan's 'up' into +Z."""
    mean_up = ups.mean(axis=0)
    mean_up /= np.linalg.norm(mean_up) or 1.0
    up = mean_up
    if len(centers) >= 6:
        c = centers - centers.mean(axis=0)
        evals, evecs = np.linalg.eigh(c.T @ c / len(c))
        normal = evecs[:, 0]  # smallest spread = axis of the orbit
        # Use the orbit normal when the cameras really form a band around the
        # object (two big spreads, one small) and it agrees with the camera ups.
        flat = evals[0] < 0.35 * evals[1]
        if flat:
            if normal @ mean_up < 0:
                normal = -normal
            if normal @ mean_up > 0.5:
                up = normal
    return _rotation_to_z(up)


def proximity_labels(points: np.ndarray, cell: float) -> np.ndarray:
    """Cluster points that lie within about one `cell` of each other
    (26-connected occupied voxels). Returns a label per point."""
    from scipy import ndimage
    if len(points) == 0:
        return np.zeros(0, dtype=int)
    lo = points.min(0)
    span = np.ptp(points, axis=0)
    cell = max(cell, float(span.max()) / 400, 1e-12)   # cap the grid size
    ijk = np.floor((points - lo) / cell).astype(np.int64)
    dims = ijk.max(0) + 1
    grid = np.zeros(dims, dtype=bool)
    grid[tuple(ijk.T)] = True
    lab, _ = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=bool))
    return lab[tuple(ijk.T)]


def table_tilt(mesh, max_angle_deg: float = 25.0) -> np.ndarray:
    """Small rotation that makes the table plane exactly horizontal.

    Camera-based 'up' is only roughly right (people tilt the phone). The table
    or floor the object stands on is the best level reference, so take the
    dominant direction of the large flat surfaces (normals near up) of an
    already roughly upright mesh. Returns identity when there is no clear table.
    """
    M = np.eye(4)
    n = mesh.face_normals
    a = mesh.area_faces
    total = a.sum()
    normal = np.array([0.0, 0.0, 1.0])
    # Mean shift on the face normals: the table top (and any other level
    # surface) forms the biggest cluster of normals near 'up'. Narrowing the
    # cone each round pulls the estimate onto that cluster and away from
    # sloped parts of the object.
    for cone in (max_angle_deg, 15.0, 8.0, 4.0, 4.0):
        sel = n @ normal > np.cos(np.radians(cone))
        if sel.sum() < 30:
            return M
        new = (n[sel] * a[sel, None]).sum(0)
        normal = new / np.linalg.norm(new)
    if a[n @ normal > np.cos(np.radians(4.0))].sum() < 0.15 * total:
        return M   # no big flat surface: trust the cameras
    if np.degrees(np.arccos(min(1.0, normal[2]))) > max_angle_deg:
        return M
    return _rotation_to_z(normal)


def _rotation_to_z(v: np.ndarray) -> np.ndarray:
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, z)
    s, c = np.linalg.norm(axis), float(v @ z)
    M = np.eye(4)
    if s < 1e-9:
        if c < 0:
            M[:3, :3] = np.diag([1.0, -1.0, -1.0])
        return M
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    M[:3, :3] = np.eye(3) + s * K + (1 - c) * (K @ K)
    return M


def object_mask(mesh, orbit_center_xy: np.ndarray, orbit_radius: float,
                remove_floor: bool = True) -> tuple[np.ndarray, dict]:
    """Faces to keep: the object in the middle of the camera orbit, without the
    table under it. Works on an already gravity-aligned mesh (Z up)."""
    import trimesh

    info = {"floor_removed": 0, "outside_removed": 0}
    keep = np.ones(len(mesh.faces), dtype=bool)
    centers = mesh.triangles_center
    normals = mesh.face_normals
    areas = mesh.area_faces

    # 1. Only what lies within the orbit (the cameras look at the object, the
    #    room behind it is outside their circle).
    r = np.linalg.norm(centers[:, :2] - orbit_center_xy, axis=1)
    inside = r < 0.8 * orbit_radius
    if inside.sum() > 0.05 * len(keep):
        info["outside_removed"] = int((~inside).sum())
        keep &= inside

    # 2. The table: the height that holds the most horizontal surface area.
    if remove_floor:
        horiz = (normals[:, 2] > 0.85) & keep   # the surface it stands on faces up
        if horiz.sum() > 50:
            z = centers[horiz, 2]
            lo, hi = np.percentile(centers[keep, 2], [1, 99])
            bins = np.linspace(lo, hi, 80)
            hist, edges = np.histogram(z, bins=bins, weights=areas[horiz])
            i = int(np.argmax(hist))
            # only treat it as a table if that layer is low and substantial
            if i < len(hist) * 0.5 and hist[i] > 0.15 * areas[keep].sum():
                floor_z = (edges[i] + edges[i + 1]) / 2
                tol = (hi - lo) * 0.015
                below = centers[:, 2] < floor_z + tol
                info["floor_removed"] = int((below & keep).sum())
                keep &= ~below
                info["floor_z"] = float(floor_z)

    # 3. Of what is left, keep the cluster that sits in the middle (the object)
    #    and anything touching its footprint; drop leftover table edges.
    #    Scans are often hundreds of disconnected islands, so group by 3D
    #    proximity, not by mesh connectivity.
    before = int(keep.sum())
    if before:
        idx = np.nonzero(keep)[0]
        pts = centers[idx]
        diag = float(np.linalg.norm(np.ptp(pts, axis=0)))
        # cell: 3% of the size, but never smaller than a few triangles across
        cell = max(0.03 * diag, 3.0 * float(np.sqrt(np.median(areas[idx]))))
        labels = proximity_labels(pts, cell)
        best, best_score = None, -1.0
        for lab in np.unique(labels):
            f = labels == lab
            d = np.linalg.norm(pts[f, :2].mean(0) - orbit_center_xy)
            score = areas[idx[f]].sum() / (1.0 + d / max(orbit_radius, 1e-9) * 4)
            if score > best_score:
                best, best_score = lab, score
        main = pts[labels == best]
        bmin, bmax = main.min(0), main.max(0)
        pad = 0.1 * (bmax - bmin)
        bmin, bmax = bmin - pad, bmax + pad
        final = np.zeros(len(mesh.faces), dtype=bool)
        for lab in np.unique(labels):
            f = labels == lab
            cmin, cmax = pts[f].min(0), pts[f].max(0)
            if np.all(cmax >= bmin) and np.all(cmin <= bmax):
                final[idx[f]] = True
        # Guard: never throw away most of what is inside the camera circle.
        if areas[final].sum() < 0.25 * areas[keep].sum():
            info["isolate_skipped"] = True
        else:
            info["parts_kept"] = int(final.sum())
            keep = final
    return keep, info
