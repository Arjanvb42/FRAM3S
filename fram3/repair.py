"""Mesh cleanup for scans and 3D printing."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh


@dataclass
class RepairOptions:
    keep_largest: bool = True          # drop floating scan debris
    remove_floor: bool = False         # cut away a flat ground plane under the object
    fill_holes: bool = True
    make_watertight: bool = True       # rebuild as a closed solid (needed for printing)
    target_faces: int | None = None    # simplify down to this many faces
    target_size_mm: float | None = None  # scale so the largest side is this long
    drop_to_bed: bool = True           # put the lowest point at z=0 and center on x/y


@dataclass
class RepairReport:
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)


def stats(mesh: trimesh.Trimesh) -> dict:
    ext = mesh.extents if len(mesh.vertices) else np.zeros(3)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "size_mm": [round(float(v), 2) for v in ext],
        "volume_cm3": round(float(abs(mesh.volume)) / 1000, 3) if mesh.is_watertight else None,
    }


def remove_floor_plane(mesh: trimesh.Trimesh, tol_ratio: float = 0.01) -> trimesh.Trimesh:
    """Remove the dominant horizontal plane at the bottom of a scan (the table)."""
    z = mesh.vertices[:, 2]
    tol = max(mesh.extents) * tol_ratio
    floor_z = np.percentile(z, 2)
    centers = mesh.triangles_center
    normals = mesh.face_normals
    flat = np.abs(normals[:, 2]) > 0.9
    low = centers[:, 2] < floor_z + tol
    keep = ~(flat & low)
    out = mesh.copy()
    out.update_faces(keep)
    out.remove_unreferenced_vertices()
    return out


def boundary_loops(m: trimesh.Trimesh) -> list[list[int]]:
    """Ordered vertex loops around every opening in the mesh."""
    import networkx as nx
    edges = m.edges_sorted
    groups = trimesh.grouping.group_rows(edges, require_count=1)
    if len(groups) == 0:
        return []
    g = nx.Graph()
    g.add_edges_from(edges[groups])
    loops = []
    for comp in nx.connected_components(g):
        sub = g.subgraph(comp)
        try:
            cyc = [u for u, _ in nx.find_cycle(sub)]
        except nx.NetworkXNoCycle:
            continue
        if len(cyc) >= 3:
            loops.append(cyc)
    return loops


def cap_holes(m: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int]:
    """Close each opening with a fan of triangles around its center point."""
    loops = boundary_loops(m)
    if not loops:
        return m, 0
    verts = [m.vertices]
    faces = [m.faces]
    nv = len(m.vertices)
    for loop in loops:
        c = m.vertices[loop].mean(axis=0)
        verts.append(c[None])
        ci = nv
        nv += 1
        ring = np.array(loop + [loop[0]])
        fan = np.column_stack([ring[1:], ring[:-1], np.full(len(loop), ci)])
        faces.append(fan)
    out = trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=True)
    trimesh.repair.fix_winding(out)
    trimesh.repair.fix_normals(out)
    return out, len(loops)


def repair(mesh: trimesh.Trimesh, opts: RepairOptions) -> tuple[trimesh.Trimesh, RepairReport]:
    rep = RepairReport(before=stats(mesh))
    m = mesh.copy()

    m.merge_vertices(merge_tex=True, merge_norm=True)
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    rep.steps.append("Dubbele en kapotte driehoeken verwijderd")

    if opts.remove_floor:
        m = remove_floor_plane(m)
        rep.steps.append("Ondergrond (tafelvlak) weggesneden")

    if opts.keep_largest:
        parts = m.split(only_watertight=False)
        if len(parts) > 1:
            # Drop small floating debris, keep everything of meaningful size.
            diag = float(np.linalg.norm(m.extents))
            keep = [p for p in parts if float(np.linalg.norm(p.extents)) >= 0.08 * diag]
            if keep and len(keep) < len(parts):
                rep.steps.append(f"Losse stukjes verwijderd ({len(parts) - len(keep)} weg)")
                m = trimesh.util.concatenate(keep)

    if opts.target_faces and len(m.faces) > opts.target_faces:
        try:
            m = m.simplify_quadric_decimation(face_count=int(opts.target_faces))
            rep.steps.append(f"Vereenvoudigd tot {len(m.faces)} driehoeken")
        except Exception as e:  # fast_simplification not installed
            rep.steps.append(f"Vereenvoudigen overgeslagen: {e}")

    if opts.fill_holes and not m.is_watertight:
        trimesh.repair.fill_holes(m)
        if not m.is_watertight:
            m, n = cap_holes(m)
            if n:
                rep.steps.append(f"Gaten gedicht ({n} opening(en) afgesloten)")
        else:
            rep.steps.append("Kleine gaten gedicht")

    trimesh.repair.fix_normals(m)
    rep.steps.append("Normalen gecorrigeerd")

    if opts.make_watertight and not m.is_watertight:
        m = _make_solid(m, rep)

    if len(m.faces) > (opts.target_faces or 400_000):
        try:
            m = m.simplify_quadric_decimation(face_count=int(opts.target_faces or 400_000))
            rep.steps.append(f"Vereenvoudigd tot {len(m.faces)} driehoeken")
        except Exception:
            pass

    if opts.target_size_mm:
        longest = float(max(m.extents))
        if longest > 0:
            m.apply_scale(opts.target_size_mm / longest)
            rep.steps.append(f"Geschaald naar {opts.target_size_mm:g} mm")

    if opts.drop_to_bed:
        lo, hi = m.bounds
        m.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
        rep.steps.append("Op het printbed gezet")

    rep.after = stats(m)
    return m, rep


def _make_solid(m: trimesh.Trimesh, rep: RepairReport) -> trimesh.Trimesh:
    """Turn an open or self-intersecting mesh into a closed solid."""
    try:
        man = to_manifold(m)
        if not man.is_empty():
            mm = man.to_mesh()
            solid = trimesh.Trimesh(mm.vert_properties[:, :3], mm.tri_verts)
            if solid.is_watertight and len(solid.faces):
                rep.steps.append("Gesloten gemaakt (manifold)")
                return solid
    except Exception:
        pass
    # Fallback: voxel remesh. Robust for scans, costs some fine detail.
    pitch = float(max(m.extents)) / 256
    vox = m.voxelized(pitch).fill()
    solid = vox.marching_cubes
    solid.apply_transform(vox.transform)
    trimesh.smoothing.filter_taubin(solid, iterations=10)
    rep.steps.append(f"Gesloten gemaakt via voxel-remesh (resolutie {pitch:.2f} mm)")
    return solid


def to_manifold(mesh: trimesh.Trimesh):
    import manifold3d
    # Rebuild from bare geometry so normals/UV seams don't keep vertices split.
    m = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=False)
    m.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=4)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    return manifold3d.Manifold(manifold3d.Mesh(
        vert_properties=np.asarray(m.vertices, dtype=np.float32),
        tri_verts=np.asarray(m.faces, dtype=np.uint32)))


def boolean(solids: list[trimesh.Trimesh], holes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Exact union of solids minus holes. Always returns a closed (watertight) mesh."""
    import manifold3d

    def solid(mesh: trimesh.Trimesh):
        man = to_manifold(mesh)
        if man.is_empty() or man.status() != manifold3d.Error.NoError:
            # not closed: rebuild as a solid first
            fixed, _ = repair(mesh, RepairOptions(keep_largest=False, drop_to_bed=False))
            man = to_manifold(fixed)
        return man

    acc = manifold3d.Manifold.batch_boolean([solid(s) for s in solids], manifold3d.OpType.Add)
    if holes:
        acc = acc - manifold3d.Manifold.batch_boolean([solid(h) for h in holes], manifold3d.OpType.Add)
    out = acc.to_mesh()
    return trimesh.Trimesh(out.vert_properties[:, :3], out.tri_verts)
