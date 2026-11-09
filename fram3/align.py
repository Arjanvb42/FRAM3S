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


def gravity_rotation(centers: np.ndarray, ups: np.ndarray, fwds: np.ndarray | None = None) -> np.ndarray:
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
    if fwds is not None and len(fwds) >= 6:
        # Filming from above, looking down: the image-up of each camera then
        # lies almost flat and says little about gravity. But a phone is rarely
        # rolled sideways, so the image's horizontal axis stays level whatever
        # the viewing angle. 'Up' is the direction all those axes are square to.
        mean_fwd = fwds.mean(axis=0)
        looking_down = np.mean(np.einsum("ij,ij->i", fwds, np.broadcast_to(up, fwds.shape)) < -0.6)
        if looking_down > 0.5 or np.linalg.norm(ups.mean(axis=0)) < 0.6:
            right = np.cross(fwds, ups)
            right /= np.linalg.norm(right, axis=1, keepdims=True) + 1e-12
            evals, evecs = np.linalg.eigh(right.T @ right / len(right))
            if evals[0] < 0.2 * evals[1]:      # the axes really span a level plane
                v = evecs[:, 0]
                if v @ (mean_up - mean_fwd) < 0:   # cameras look down and image-up points up
                    v = -v
                up = v
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


def table_plane(mesh, max_angle_deg: float = 45.0, iters: int = 300):
    """Find the table the object stands on: the plane, roughly facing up, that
    carries the most surface area. Fitted on face positions (RANSAC + least
    squares), because the normals of a scanned table are noisy.

    Returns (normal, offset, thickness) with normal . p = offset on the plane
    and thickness the spread of the table faces around it, or None.
    """
    n = mesh.face_normals
    a = mesh.area_faces
    c = mesh.triangles_center
    total = a.sum()
    # People tilt the phone a lot around low objects: camera 'up' can be 30+
    # degrees off, so accept planes up to 45 degrees from it.
    cand = np.nonzero(n[:, 2] > np.cos(np.radians(min(max_angle_deg + 10, 80))))[0]
    if len(cand) < 50:
        return None
    diag = float(np.linalg.norm(np.ptp(c, axis=0)))
    tol = 0.01 * diag
    rng = np.random.default_rng(0)
    p_pick = a[cand] / a[cand].sum()
    cc, ac = c[cand], a[cand]
    best, best_area = None, 0.0
    cos_max = np.cos(np.radians(max_angle_deg))
    for _ in range(iters):
        i = rng.choice(cand, 3, replace=False, p=p_pick)
        nrm = np.cross(c[i[1]] - c[i[0]], c[i[2]] - c[i[0]])
        L = np.linalg.norm(nrm)
        if L < 1e-12:
            continue
        nrm /= L
        if nrm[2] < 0:
            nrm = -nrm
        if nrm[2] < cos_max:
            continue
        ar = ac[np.abs((cc - c[i[0]]) @ nrm) < tol].sum()
        if ar > best_area:
            best, best_area = (nrm, float(c[i[0]] @ nrm)), ar
    if best is None or best_area < 0.15 * total:
        return None
    nrm, off = best
    for _ in range(3):   # refine with a weighted least-squares plane through the inliers
        inl = cand[np.abs(c[cand] @ nrm - off) < tol]
        if len(inl) < 10:
            return None
        w = a[inl]
        mu = np.average(c[inl], axis=0, weights=w)
        cov = ((c[inl] - mu) * w[:, None]).T @ (c[inl] - mu)
        nrm = np.linalg.eigh(cov)[1][:, 0]
        if nrm[2] < 0:
            nrm = -nrm
        off = float(mu @ nrm)
    if nrm[2] < cos_max:
        return None
    dist = c[inl] @ nrm - off
    thickness = float(np.sqrt(np.average(dist ** 2, weights=a[inl])))
    return nrm, off, thickness


def table_tilt(mesh, max_angle_deg: float = 45.0) -> np.ndarray:
    """Small rotation that makes the table plane exactly horizontal.

    Camera-based 'up' is only roughly right (people tilt the phone), the table
    the object stands on is the best level reference. Returns identity when
    there is no clear table.
    """
    plane = table_plane(mesh, max_angle_deg)
    if plane is None:
        return np.eye(4)
    return _rotation_to_z(plane[0])


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

    # 2. The table: the dominant up-facing plane low in the scan. Cut it and
    #    everything under it, with a margin for its bumpiness.
    if remove_floor:
        sub = mesh.submesh([np.nonzero(keep)[0]], append=True) if not keep.all() else mesh
        plane = table_plane(sub)
        if plane is not None:
            nrm, off, thick = plane
            d = centers @ nrm - off
            above = keep & (d > 0)
            lo, hi = np.percentile(centers[above, 2], [1, 99]) if above.sum() > 10 else (0.0, 0.0)
            height = hi - lo
            # It is the table under the object when there is real surface
            # standing on it near the middle of the orbit. (Do not require the
            # plane to be low: junk under a table can make it look high.)
            mid = np.linalg.norm(centers[:, :2] - orbit_center_xy, axis=1) < 0.4 * orbit_radius
            tol = max(0.015 * height, 2.5 * thick)
            on_top = above & mid & (d > tol)
            flat_enough = abs(nrm[0]) + abs(nrm[1]) < 0.2
            if flat_enough and areas[on_top].sum() > 0.03 * areas[keep].sum():
                floor_z = off / nrm[2]
                below = d < tol
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
        # The object is the tall, compact lump; leftover table (rippled paper,
        # a cloth) is wide and low, even when it lies nearer the orbit centre,
        # which hand-held filming does not pin down well.
        floor = info.get("floor_z")
        best, best_score = None, -1.0
        for lab in np.unique(labels):
            f = labels == lab
            p = pts[f]
            if len(p) < 3:
                continue
            z_lo, z_hi = np.percentile(p[:, 2], [2, 98])
            height = max(z_hi - (floor if floor is not None else z_lo), 1e-9)
            foot = float(np.linalg.norm(np.percentile(p[:, :2], 98, axis=0) - np.percentile(p[:, :2], 2, axis=0)))
            compact = min(height / max(foot, 1e-9), 2.0)
            d = np.linalg.norm(p[:, :2].mean(0) - orbit_center_xy)
            score = height * compact * np.sqrt(areas[idx[f]].sum()) / (1.0 + 2.0 * d / max(orbit_radius, 1e-9))
            if score > best_score:
                best, best_score = lab, score
        if best is None:
            best = labels[0]
        main = pts[labels == best]
        bmin, bmax = main.min(0), main.max(0)
        pad = 0.1 * (bmax - bmin)
        bmin, bmax = bmin - pad, bmax + pad
        final = np.zeros(len(mesh.faces), dtype=bool)
        inside_box = np.all((pts >= bmin) & (pts <= bmax), axis=1)
        for lab in np.unique(labels):
            f = labels == lab
            # keep parts that mostly lie within the object's box (a loose
            # handle, a separate lid), not a sheet that only pokes into it
            a_all = areas[idx[f]].sum()
            if lab == best or areas[idx[f & inside_box]].sum() >= 0.5 * a_all:
                final[idx[f]] = True
        # Guard: never throw away most of what is inside the camera circle.
        if areas[final].sum() < 0.25 * areas[keep].sum():
            info["isolate_skipped"] = True
        else:
            info["parts_kept"] = int(final.sum())
            keep = final
    return keep, info
