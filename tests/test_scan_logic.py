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
from fram3 import reconstruct as R  # noqa: E402


def fake_scan(wd: Path, seed: int, arc: float = 1.5 * np.pi, tilt_deg: float = 8.0, fragment: bool = False,
              sheet: bool = False):
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
    # and a long thin sliver from under the base up the side (seen on a real scan)
    sliver = trimesh.Trimesh([[0.2, 0.3, 0.3], [0.21, 0.3, 0.3], [0.2, 0.31, -0.3]], [[0, 1, 2]])
    parts = [obj, table, wall, slab, *specks, sliver]
    aim = np.array([0, 0, 0.5])
    if sheet:
        # a rippled notepad lying next to the object, where the cameras aim
        pad = trimesh.creation.box((1.6, 1.6, 0.04)).subdivide().subdivide().subdivide().subdivide()
        pad.vertices[:, 2] += 0.08 * np.sin(pad.vertices[:, 0] * 6) * np.cos(pad.vertices[:, 1] * 5) + 0.1
        pad.apply_translation((1.4, 0, 0))
        parts.append(pad)
        aim = np.array([1.0, 0, 0.5])
    m = trimesh.util.concatenate(parts)
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
        C = T @ np.array([2.5 * np.cos(a), 2.5 * np.sin(a), 0.7]) + aim
        f = aim - C
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
    cases = ((1, False, 8, False), (2, False, 21, False), (3, True, 8, False), (4, True, 21, False),
             (5, True, 35, False), (6, False, 30, True), (7, True, 30, True))
    for seed, frag, tilt, sheet in cases:
        wd = Path(tempfile.mkdtemp())
        try:
            fake_scan(wd, seed, fragment=frag, tilt_deg=tilt, sheet=sheet)
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
    real = R._images_with_features
    R._images_with_features = lambda db: 50
    try:
        imgs = wd / "images"
        imgs.mkdir()
        for i in range(50):
            Image.new("RGB", (576, 1024)).save(imgs / f"{i:03}.jpg")
        job = Job("t", wd, "snel")
        r = Reconstructor(job, [], Tools(None, None, False))
        calls = []
        r.colmap = lambda *a, **k: calls.append(a[0])
        r.map = lambda *a, **k: (wd, 9)
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
        r2.map = lambda *a, **k: (wd, 25)
        r2.sparse(imgs, sequential=False)
        assert job2.warning and "25 van de 50" in job2.warning
        # the relaxed retry is used when it places more images
        job3 = Job("v", wd, "snel")
        r3 = Reconstructor(job3, [], Tools(None, None, False))
        r3.colmap = lambda *a, **k: None
        r3.map = lambda *a, relaxed=False: (wd, 45 if relaxed else 2)
        r3.sparse(imgs, sequential=False)
        assert job3.warning is None and r3.registered == (45, 50)
    finally:
        R._images_with_features = real
        shutil.rmtree(wd)


def test_gpu_out_of_memory_falls_back_to_cpu():
    wd = Path(tempfile.mkdtemp())
    real = R._images_with_features, R._gpu_mem_mib
    try:
        imgs = wd / "images"
        imgs.mkdir()
        for i in range(20):
            Image.new("RGB", (1800, 3200)).save(imgs / f"{i:03}.jpg")
        state = {"n": 0}
        R._images_with_features = lambda db: state["n"]
        R._gpu_mem_mib = lambda: 4096
        runs = []

        def colmap(*a, **k):
            if a[0] == "feature_extractor":
                use_gpu = a[a.index("--SiftExtraction.use_gpu") + 1]
                runs.append((use_gpu, "--SiftExtraction.max_image_size" in a))
                if use_gpu == "1":
                    raise RuntimeError("CuTexImage::BindTexture: out of memory")
                state["n"] = 20
        job = Job("g", wd, "hoog")
        r = Reconstructor(job, [], Tools(None, None, True))
        r.colmap = colmap
        r.map = lambda *a, **k: (wd, 20)
        r.sparse(imgs, sequential=False)
        assert runs == [("1", True), ("0", False)], runs
        # nothing found at all: a clear message, no stack trace from the matcher
        state["n"] = 0
        r.colmap = lambda *a, **k: None
        try:
            r.sparse(imgs, sequential=False)
        except RuntimeError as e:
            assert "herkenningspunten" in str(e)
        else:
            raise AssertionError("expected a clear error")
    finally:
        R._images_with_features, R._gpu_mem_mib = real
        shutil.rmtree(wd)


def test_live_view_reads_growing_scan():
    """The live view shows cameras and points upright, centred and bed-sized."""
    import struct
    from fram3 import live
    wd = Path(tempfile.mkdtemp())
    try:
        fake_scan(wd, 3)
        model = wd / "dense" / "sparse"
        assert live.cloud_source(wd) is None or live.cloud_source(wd)[0] != "sparse"
        rng = np.random.default_rng(0)
        pts = rng.normal(0, 0.3, (500, 3))
        with (model / "points3D.bin").open("wb") as f:
            f.write(struct.pack("<Q", len(pts)))
            for i, p in enumerate(pts):
                k = i % 4
                f.write(struct.pack("<Q3d3BdQ", i, *p, 10, 20, 30, 0.5, k) + b"\0" * 8 * k)
        xyz, rgb = live.read_points3d(model / "points3D.bin")
        assert np.allclose(xyz, pts) and (rgb == [10, 20, 30]).all()
        src = live.cloud_source(wd)
        assert src[0] == "sparse"
        blob = live.build_cloud(src)
        nc, npts = struct.unpack_from("<II", blob, 4)
        assert blob[:4] == b"F3LV" and nc == 24 and 0 < npts <= 500
        assert len(blob) == 12 + nc * 36 + npts * 15
        cams = np.frombuffer(blob, "<f4", nc * 9, 12).reshape(nc, 9)
        r = np.linalg.norm(cams[:, :2], axis=1)
        assert abs(np.median(r) - live.VIEW_RADIUS) < 15, np.median(r)   # orbit around the middle
        assert (cams[:, 2] > 0).all() and (cams[:, 8] > 0.5).all()       # above the bed, upright
        # a dense cloud takes over as soon as it exists
        a = np.zeros(1000, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        a["x"], a["y"], a["z"] = rng.normal(0, 0.3, (3, 1000))
        with (wd / "mvs" / "scene_dense.ply").open("wb") as f:
            f.write(("ply\nformat binary_little_endian 1.0\nelement vertex 1000\n" + "".join(
                f"property {t} {n}\n" for n, t in [("x", "float"), ("y", "float"), ("z", "float"), ("red", "uchar"),
                                                    ("green", "uchar"), ("blue", "uchar")]) + "end_header\n").encode())
            a.tofile(f)
        assert live.cloud_source(wd)[0] == "dense"
        assert struct.unpack_from("<II", live.build_cloud(live.cloud_source(wd)), 4)[1] > 900
        st = live.state(wd, "dense", 0.5)
        assert st["cloud"]["kind"] == "dense" and st["frames"] == 0
    finally:
        shutil.rmtree(wd)


if __name__ == "__main__":
    test_cleanup_upright_without_table()
    test_low_registration_fails_clearly()
    test_gpu_out_of_memory_falls_back_to_cpu()
    test_live_view_reads_growing_scan()
    print("ok")
