"""Scan clean-up and registration checks without COLMAP (run: python3 tests/test_scan_logic.py)."""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fram3.reconstruct import Job, Reconstructor, Tools  # noqa: E402


def fake_scan(wd: Path, seed: int, arc: float = 1.5 * np.pi, tilt_deg: float = 8.0, fragment: bool = False):
    """Box on a big table plus a wall, randomly rotated, with cameras on a tilted partial orbit."""
    (wd / "dense" / "sparse").mkdir(parents=True)
    (wd / "mvs").mkdir()
    obj = trimesh.creation.box((0.4, 0.6, 1.0)).subdivide().subdivide().subdivide()
    obj.apply_translation((0, 0, 0.5))
    table = trimesh.creation.box((4, 4, 0.05)).subdivide().subdivide().subdivide().subdivide()
    if fragment:   # wrinkled like a real scan: face normals scatter by ~10 degrees
        table.vertices[:, 2] += np.random.default_rng(seed).normal(0, 0.03, len(table.vertices))
    table.apply_translation((0, 0, -0.025))
    wall = trimesh.creation.box((6, 0.1, 3))
    wall.apply_translation((0, 5, 1.5))
    rg = trimesh.transformations.random_rotation_matrix(np.random.default_rng(seed).random(3))
    # junk under the table edge (table leg / background sheet), deeper than the object is tall
    slab = trimesh.creation.box((3, 0.1, 3)).subdivide().subdivide().subdivide()
    slab.apply_translation((0, 1.6, -1.6))
    # a few dark specks floating just under the object's base, above the table cut
    specks = [trimesh.creation.icosphere(0, 0.01) for _ in range(3)]
    for i, sp in enumerate(specks):
        sp.apply_translation((0.1 * i, 0.0, 0.03))
    m = trimesh.util.concatenate([obj, table, wall, slab, *specks])
    m.apply_transform(rg)
    if fragment:
        # Like a real OpenMVS mesh: hundreds of loose islands, some faces missing.
        rng = np.random.default_rng(seed)
        m.update_faces(rng.random(len(m.faces)) > 0.3)
        m.unmerge_vertices()
    m.visual = trimesh.visual.TextureVisuals(uv=np.random.rand(len(m.vertices), 2),
                                             image=Image.new("RGB", (8, 8), (200, 50, 50)))
    m.export(wd / "mvs" / "scene_textured.obj")
    T = trimesh.transformations.rotation_matrix(np.radians(tilt_deg), [1, 0, 0])[:3, :3]
    lines = []
    for i, a in enumerate(np.linspace(0, arc, 24)):
        C = T @ np.array([2.5 * np.cos(a), 2.5 * np.sin(a), 0.7]) + [0, 0, 0.5]
        f = np.array([0, 0, 0.5]) - C
        f /= np.linalg.norm(f)
        x = np.cross(f, T @ [0, 0, 1])
        x /= np.linalg.norm(x)
        y = np.cross(f, x)
        R = np.stack([x, y, f]) @ rg[:3, :3].T
        t = -R @ (rg[:3, :3] @ C)
        M = np.eye(4)
        M[:3, :3] = R
        q = trimesh.transformations.quaternion_from_matrix(M)
        lines.append(f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 img{i}.jpg\n\n")
    (wd / "dense" / "sparse" / "images.txt").write_text("".join(lines))
    return len(obj.faces)


def test_cleanup_upright_without_table():
    for seed, frag, tilt in ((1, False, 8), (2, False, 21), (3, True, 8), (4, True, 21)):
        wd = Path(tempfile.mkdtemp())
        try:
            fake_scan(wd, seed, fragment=frag, tilt_deg=tilt)
            job = Job("t", wd, "snel")
            out = Reconstructor(job, [], Tools(None, None, False)).cleanup(wd / "mvs" / "scene_textured.obj")
            g = trimesh.load(out, force="scene").dump()[0]
            assert len(g.faces) > 0.5 * 0.7 * 640 * (0.7 if frag else 1), (seed, len(g.faces), job.log)
            assert abs(g.extents[2] - 1.0) < 0.03, (seed, g.extents, job.log)
            assert g.bounds[0][2] > -1e-9   # upright, table gone
            assert max(g.extents[:2]) < 0.8, (seed, g.extents)                  # wall gone
            assert g.visual.kind == "texture"
        finally:
            shutil.rmtree(wd)


def test_low_registration_fails_clearly():
    wd = Path(tempfile.mkdtemp())
    try:
        imgs = wd / "images"
        imgs.mkdir()
        for i in range(50):
            Image.new("RGB", (576, 1024)).save(imgs / f"{i:03}.jpg")
        job = Job("t", wd, "snel")
        r = Reconstructor(job, [], Tools(None, None, False))
        calls = []
        r.colmap = lambda *a, **k: calls.append(a[0])
        r.map = lambda *a: (wd, 9)
        try:
            r.sparse(imgs, sequential=False)
        except RuntimeError as e:
            assert "9 van de 50" in str(e)
        else:
            raise AssertionError("expected a clear error")
        # weak but usable: warning, not an error
        job2 = Job("u", wd, "snel")
        r2 = Reconstructor(job2, [], Tools(None, None, False))
        r2.colmap = lambda *a, **k: None
        r2.map = lambda *a: (wd, 25)
        r2.sparse(imgs, sequential=False)
        assert job2.warning and "25 van de 50" in job2.warning
    finally:
        shutil.rmtree(wd)


if __name__ == "__main__":
    test_cleanup_upright_without_table()
    test_low_registration_fails_clearly()
    print("ok")
