"""Drone footage: GPS reading, frame choice, matching pairs and the scaled clean-up
(run: python3 tests/test_drone.py)."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fram3 import drone  # noqa: E402
from fram3.reconstruct import Job, Reconstructor, Tools  # noqa: E402

NEW_SRT = """1
00:00:00,000 --> 00:00:00,033
<font size="28">FrameCnt: 1, DiffTime: 33ms
2026-11-13 15:02:11.123
[iso: 100] [shutter: 1/1000.0] [fnum: 1.7] [ev: 0] [color_md: default] [focal_len: 24.00] [latitude: 52.091234] [longitude: 5.121234] [rel_alt: 30.200 abs_alt: 45.512] [ct: 5500] </font>

2
00:00:00,033 --> 00:00:00,066
<font size="28">FrameCnt: 2, DiffTime: 33ms
[latitude: 52.091240] [longitude: 5.121250] [rel_alt: 30.300 abs_alt: 45.600] </font>
"""

OLD_SRT = """1
00:00:01,000 --> 00:00:02,000
HOME(5.1200,52.0900) 2019.08.10 14:01:02
GPS(5.1212,52.0912,19) BAROMETER:30.4
ISO:100 Shutter:500 EV:0 Fnum:F2.8

2
00:00:02,000 --> 00:00:03,000
F/2.8, SS 1000, ISO 100, EV 0, GPS (5.1213, 52.0913, 19), D 12.3m, H 31.0m, H.S 2.0m/s, V.S 0.0m/s
"""


def test_parse_srt_both_formats():
    a = drone.parse_srt(NEW_SRT)
    assert len(a) == 2
    t, lat, lon, alt = a[0]
    assert t == 0 and abs(lat - 52.091234) < 1e-9 and abs(lon - 5.121234) < 1e-9 and alt == 30.2
    assert abs(a[1][0] - 0.033) < 1e-9
    b = drone.parse_srt(OLD_SRT)
    assert len(b) == 2
    assert b[0][0] == 1.0 and abs(b[0][1] - 52.0912) < 1e-9 and abs(b[0][2] - 5.1212) < 1e-9
    assert b[1][3] == 31.0   # "H 31.0m" is the height above take-off


def _gps_photo(path: Path, lat: float, lon: float, alt: float, size=(64, 48)):
    img = Image.new("RGB", size, (90, 120, 60))
    ex = Image.Exif()
    def dms(v):
        v = abs(v)
        d = int(v)
        m = int((v - d) * 60)
        return (float(d), float(m), (v - d - m / 60) * 3600)
    ex[0x8825] = {1: "N" if lat >= 0 else "S", 2: dms(lat), 3: "E" if lon >= 0 else "W", 4: dms(lon),
                  5: 0, 6: float(alt)}
    img.save(path, exif=ex)


def test_photo_gps_and_local_metres():
    wd = Path(tempfile.mkdtemp())
    try:
        p = wd / "DJI_0001.JPG"
        _gps_photo(p, 52.0912345, -5.1234567, 42.5)
        lat, lon, alt = drone.photo_gps(p)
        assert abs(lat - 52.0912345) < 1e-6 and abs(lon + 5.1234567) < 1e-6 and abs(alt - 42.5) < 1e-6
        Image.new("RGB", (8, 8)).save(wd / "plain.jpg")
        assert drone.photo_gps(wd / "plain.jpg") is None
        # 0.001 degree north is about 111 m, east at 52 degrees about 68 m
        xyz = drone.to_local(np.array([[52.001, 5.001, 10.0]]), np.array([52.0, 5.0, 0.0]))[0]
        assert abs(xyz[1] - 111.3) < 0.5 and abs(xyz[0] - 68.6) < 0.5 and xyz[2] == 10.0
    finally:
        shutil.rmtree(wd)


def test_frames_picked_per_metre_not_per_second():
    # 10 s hovering, then 10 s flying 50 m: frames should come from the flight
    t = np.linspace(0, 20, 400)
    x = np.where(t < 10, 0.0, (t - 10) * 5.0)
    pos = np.column_stack([x, np.zeros_like(x), np.full_like(x, 30.0)])
    scores = np.ones(len(t))
    chosen = drone.pick_by_distance(pos, scores, 20)
    in_hover = sum(1 for i in chosen if t[i] < 10)
    assert in_hover <= 1 and len(chosen) >= 18, (in_hover, len(chosen))
    # all hovering: falls back to evenly in time
    assert len(drone.pick_by_distance(np.zeros((50, 3)), np.ones(50), 10)) == 10


def test_neighbour_pairs_and_print_scale():
    # two orbits around a house: 40 frames at 15 m, 40 frames at 30 m height
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    low = np.column_stack([25 * np.cos(ang), 25 * np.sin(ang), np.full(40, 15.0)])
    high = np.column_stack([30 * np.cos(ang), 30 * np.sin(ang), np.full(40, 30.0)])
    xyz = np.vstack([low, high])
    names = [f"f{i:05d}.jpg" for i in range(80)]
    pairs = drone.neighbour_pairs(names, xyz, k=20)
    assert len(pairs) < 80 * 79 / 2 / 2                      # far fewer than all pairs
    assert any(int(a[1:6]) < 40 <= int(b[1:6]) for a, b in pairs)   # the orbits are linked
    assert drone.print_scale(np.array([12.0, 9.0, 7.0])) == 100     # a house: 120 mm
    assert drone.print_scale(np.array([300.0, 200.0, 20.0])) == 2000


def test_plan_needs_spread_out_gps():
    wd = Path(tempfile.mkdtemp())
    try:
        inputs = []
        for i in range(5):   # hand-held: a metre apart, so not a drone
            p = wd / f"IMG_{i}.jpg"
            _gps_photo(p, 52.0 + i * 0.000005, 5.0, 3.0)
            inputs.append(p)
        job = Job("t", wd, "snel")
        assert drone.plan(Reconstructor(job, inputs, Tools(None, None, False))) is None
        for i in range(5):   # a drone circling: 40 m across
            p = wd / f"DJI_{i}.JPG"
            a = i * 2 * np.pi / 5
            _gps_photo(p, 52.0 + 0.00018 * np.sin(a), 5.0 + 0.0003 * np.cos(a), 30.0)
        inputs = sorted(wd.glob("DJI_*"))
        rec = Reconstructor(job, inputs, Tools(None, None, False))
        geo = drone.plan(rec)
        assert geo and len(geo["photos"]) == 5
        (wd / "images").mkdir()
        assert drone.collect_frames(rec, wd / "images", geo) == 5
        pos = json.loads((wd / "geo.json").read_text())
        assert sorted(pos) == sorted(p.name for p in (wd / "images").iterdir())
        assert drone.spread(np.array(list(pos.values()))) > 30
    finally:
        shutil.rmtree(wd)


def test_video_with_srt_sidecar():
    if not shutil.which("ffmpeg"):
        print("  (ffmpeg ontbreekt: video-test overgeslagen)")
        return
    wd = Path(tempfile.mkdtemp())
    try:
        v = wd / "DJI_0007.MP4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30",
                        "-t", "8", "-pix_fmt", "yuv420p", str(v)], check=True)
        blocks = []
        for i in range(0, 241, 10):   # 8 s: hover 4 s, then fly 40 m east
            t = i / 30
            x = 0.0 if t < 4 else (t - 4) * 10
            lon = 5.0 + x / (6378137.0 * np.cos(np.radians(52)) * np.pi / 180)
            s, ms = divmod(int(t * 1000), 1000)
            blocks.append(f"{i // 10 + 1}\n00:00:{s:02d},{ms:03d} --> 00:00:{s:02d},{ms:03d}\n"
                          f"[latitude: 52.0] [longitude: {lon:.8f}] [rel_alt: 25.0 abs_alt: 40.0]\n")
        (wd / "DJI_0007.SRT").write_text("\n".join(blocks))
        inputs = [v, wd / "DJI_0007.SRT"]
        job = Job("t", wd, "snel")
        rec = Reconstructor(job, inputs, Tools(None, None, False))
        geo = drone.plan(rec)
        assert geo, job.log
        (wd / "images").mkdir()
        n = drone.collect_frames(rec, wd / "images", geo)
        pos = np.array(list(json.loads((wd / "geo.json").read_text()).values()))
        assert n == len(pos) >= 15, (n, job.log)
        assert (pos[:, 0] < 1.0).sum() <= 2, pos[:, 0]   # hardly any frames from the hover
    finally:
        shutil.rmtree(wd)


def fake_house(wd: Path):
    """A GPS-aligned (metres, Z up) scan of a house on a lawn with cameras circling it."""
    (wd / "dense" / "sparse").mkdir(parents=True)
    (wd / "mvs").mkdir()
    house = trimesh.creation.box((10, 8, 6)).subdivide().subdivide().subdivide()
    house.apply_translation((3, -2, 3.0))
    lawn = trimesh.creation.box((80, 80, 0.2))
    for _ in range(5):
        lawn = lawn.subdivide()
    lawn.apply_translation((0, 0, -0.1))
    m = trimesh.util.concatenate([house, lawn])
    m.visual = trimesh.visual.TextureVisuals(uv=np.random.rand(len(m.vertices), 2),
                                             image=Image.new("RGB", (8, 8), (200, 50, 50)))
    m.export(wd / "mvs" / "scene_textured.obj")
    aim = np.array([3, -2, 3.0])
    lines = []
    for i, a in enumerate(np.linspace(0, 2 * np.pi, 30, endpoint=False)):
        C = aim + [22 * np.cos(a), 22 * np.sin(a), 14]
        f = (aim - C) / np.linalg.norm(aim - C)
        x = np.cross(f, [0, 0, 1])
        x /= np.linalg.norm(x)
        y = np.cross(f, x)
        Rm = np.stack([x, y, f])
        t = -Rm @ C
        M = np.eye(4)
        M[:3, :3] = Rm
        q = trimesh.transformations.quaternion_from_matrix(M)
        lines.append(f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 f{i:05d}.jpg\n\n")
    (wd / "dense" / "sparse" / "images.txt").write_text("".join(lines))
    (wd / "geo_aligned").write_text("1")


def test_house_cleanup_keeps_gps_up_and_scales():
    wd = Path(tempfile.mkdtemp())
    try:
        fake_house(wd)
        job = Job("t", wd, "snel")
        out = Reconstructor(job, [], Tools(None, None, False)).cleanup(wd / "mvs" / "scene_textured.obj")
        g = trimesh.load(out, force="scene").dump()[0]
        # 1:50 -> the house is 200 x 160 x 120 mm, standing upright, lawn gone
        assert np.allclose(sorted(g.extents), [120, 160, 200], atol=3), (g.extents, job.log)
        assert g.bounds[0][2] > -1e-6
        assert any("1:50" in line for line in job.log), job.log
    finally:
        shutil.rmtree(wd)


if __name__ == "__main__":
    test_parse_srt_both_formats()
    test_photo_gps_and_local_metres()
    test_frames_picked_per_metre_not_per_second()
    test_neighbour_pairs_and_print_scale()
    test_plan_needs_spread_out_gps()
    test_video_with_srt_sidecar()
    test_house_cleanup_keeps_gps_up_and_scales()
    print("ok")
