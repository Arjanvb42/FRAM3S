"""Scan with a phone: the phone opens a page on this computer (QR code, same
Wi-Fi) and sends camera frames while you walk around the object.

A phone browser only gives camera access over HTTPS, so this runs a second,
small server on the local network with a self-signed certificate. It serves
nothing but the phone page, and only to the phone that has the one-time code
from the QR. The editor itself stays on 127.0.0.1.
"""
from __future__ import annotations

import io
import json
import secrets
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image

PORT = 8734
MAX_FRAMES = 400
MAX_FRAME_BYTES = 20 * 1024**2
WEB = Path(__file__).resolve().parent.parent / "web"


class Session:
    def __init__(self, root: Path):
        self.token = secrets.token_urlsafe(12)
        self.dir = root / self.token
        self.dir.mkdir(parents=True)
        self.device: dict | None = None
        self.frames = 0
        self.done = False
        self.closed = False
        self.seen = 0.0
        self.lock = threading.Lock()
        # LiDAR preview (only filled by the FRAM3 iPhone app): world points in
        # metres, their colours, and the camera poses.
        self.points: list[np.ndarray] = []
        self.colors: list[np.ndarray] = []
        self.poses: list[np.ndarray] = []
        self.depth_frames = 0

    def files(self) -> list[Path]:
        return sorted(self.dir.glob("t*.jpg"))

    def public(self) -> dict:
        return {"token": self.token, "connected": self.device is not None, "device": self.device,
                "frames": self.frames, "done": self.done, "depth_frames": self.depth_frames,
                "cloud": len(self.poses) if self.depth_frames else 0,
                "active": time.time() - self.seen < 5 if self.seen else False}


_session: Session | None = None
_server = None
_lock = threading.Lock()


def current(token: str | None = None) -> Session | None:
    s = _session
    if s is None or s.closed or (token is not None and not secrets.compare_digest(token, s.token)):
        return None
    return s


def lan_ip() -> str:
    """This computer's address on the local network (no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def certificate(folder: Path, ip: str) -> tuple[Path, Path]:
    """Self-signed certificate for this address, made once with openssl."""
    folder.mkdir(parents=True, exist_ok=True)
    key, crt, tag = folder / "key.pem", folder / "cert.pem", folder / "ip.txt"
    if key.exists() and crt.exists() and tag.exists() and tag.read_text() == ip:
        return key, crt
    r = subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825",
                        "-keyout", str(key), "-out", str(crt), "-subj", "/CN=FRAM3",
                        "-addext", f"subjectAltName=IP:{ip},DNS:localhost"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("Kon geen certificaat maken (openssl): " + r.stderr[-200:])
    key.chmod(0o600)
    tag.write_text(ip)
    return key, crt


def qr_svg(url: str) -> str | None:
    try:
        import segno
    except ImportError:
        return None
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="svg", scale=8, border=2, dark="#111", light="#fff",
                                    xmldecl=False, svgclass=None, lineclass=None, omitsize=True)
    return buf.getvalue().decode()


# ----------------------------------------------------------- phone-side app
phone_app = FastAPI(title="FRAM3 telefoon", docs_url=None, redoc_url=None, openapi_url=None)


def _need(token: str) -> Session:
    s = current(token)
    if s is None:
        raise HTTPException(404, "Deze koppeling is verlopen. Scan de QR-code opnieuw.")
    s.seen = time.time()
    return s


@phone_app.get("/p/{token}")
def phone_page(token: str):
    _need(token)
    return FileResponse(WEB / "phone.html", headers={"Cache-Control": "no-store"})


@phone_app.post("/p/{token}/hello")
async def phone_hello(token: str, request: Request):
    s = _need(token)
    body = await request.body()
    if len(body) > 20000:
        raise HTTPException(413, "Te groot")
    try:
        info = json.loads(body)
    except ValueError:
        raise HTTPException(400, "Ongeldig")
    cams = [str(c)[:80] for c in info.get("cameras", [])][:12]
    s.device = {"name": str(info.get("name", "Telefoon"))[:60], "cameras": cams,
                "lidar": info.get("lidar") if info.get("lidar") in (True, False) else None,
                "app": info.get("app") is True,
                "width": int(info.get("width") or 0), "height": int(info.get("height") or 0)}
    return {"ok": True, "max_frames": MAX_FRAMES}


@phone_app.post("/p/{token}/frame")
async def phone_frame(token: str, request: Request):
    s = _need(token)
    if s.done:
        raise HTTPException(409, "Deze scan is al afgerond.")
    body = await request.body()
    if len(body) > MAX_FRAME_BYTES:
        raise HTTPException(413, "Beeld te groot")
    try:
        with Image.open(io.BytesIO(body)) as im:
            if im.format != "JPEG":
                raise ValueError
            im.verify()
    except Exception:
        raise HTTPException(400, "Geen geldig beeld")
    with s.lock:
        if s.frames >= MAX_FRAMES:
            raise HTTPException(409, "Genoeg beelden.")
        s.frames += 1
        (s.dir / f"t{s.frames:04d}.jpg").write_bytes(body)
    return {"frames": s.frames, "max_frames": MAX_FRAMES}


def backproject(depth: np.ndarray, conf: np.ndarray | None, K: np.ndarray, image_size: tuple[int, int],
                pose: np.ndarray, step: int = 4, max_depth: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """LiDAR depth map -> world points (metres) and their pixel positions in
    the photo (0..1). ARKit conventions: K is for the full photo, the camera
    looks along -z with +y up, `pose` is camera-to-world."""
    h, w = depth.shape
    v, u = np.mgrid[step // 2:h:step, step // 2:w:step]
    d = depth[v, u]
    ok = np.isfinite(d) & (d > 0.05) & (d < max_depth)
    if conf is not None:
        ok &= conf[v, u] >= 1          # ARKit: 0 low, 1 medium, 2 high
    u, v, d = u[ok].astype(np.float64), v[ok].astype(np.float64), d[ok].astype(np.float64)
    sx, sy = w / image_size[0], h / image_size[1]
    x = (u + 0.5 - K[0, 2] * sx) / (K[0, 0] * sx) * d
    y = (v + 0.5 - K[1, 2] * sy) / (K[1, 1] * sy) * d
    cam = np.stack([x, -y, -d, np.ones_like(d)], axis=1)
    return (cam @ pose.T)[:, :3], np.stack([(u + 0.5) / w, (v + 0.5) / h], axis=1)


@phone_app.post("/p/{token}/arframe")
async def phone_arframe(token: str, meta: str = Form(...), image: UploadFile = File(...),
                        depth: UploadFile | None = File(None), conf: UploadFile | None = File(None)):
    """A frame from the FRAM3 iPhone app: photo, camera pose and intrinsics
    from ARKit, and (with LiDAR) a depth map with confidence."""
    s = _need(token)
    if s.done:
        raise HTTPException(409, "Deze scan is al afgerond.")
    body = await image.read(MAX_FRAME_BYTES + 1)
    if len(body) > MAX_FRAME_BYTES or len(meta) > 4000:
        raise HTTPException(413, "Beeld te groot")
    try:
        m = json.loads(meta)
        pose = np.array(m["pose"], np.float64).reshape(4, 4)        # row-major camera-to-world
        K = np.array(m["intrinsics"], np.float64).reshape(3, 3)     # row-major
        size = (int(m["width"]), int(m["height"]))
        with Image.open(io.BytesIO(body)) as im:
            if im.format != "JPEG":
                raise ValueError
            im.draft("RGB", (256, 256))
            thumb = np.asarray(im.convert("RGB"))
        if not (np.isfinite(pose).all() and np.isfinite(K).all() and K[0, 0] > 0 and min(size) > 0):
            raise ValueError
    except Exception:
        raise HTTPException(400, "Ongeldig beeld of gegevens")
    dm = cf = None
    if depth is not None:
        try:
            dw, dh = int(m["depth_width"]), int(m["depth_height"])
            raw = await depth.read(4 * 1024 * 1024 + 1)
            if not (0 < dw <= 1024 and 0 < dh <= 1024) or len(raw) != dw * dh * 4:
                raise ValueError
            dm = np.frombuffer(raw, "<f4").reshape(dh, dw)
            if conf is not None:
                craw = await conf.read(1024 * 1024 + 1)
                if len(craw) == dw * dh:
                    cf = np.frombuffer(craw, np.uint8).reshape(dh, dw)
        except Exception:
            raise HTTPException(400, "Ongeldige dieptekaart")
    with s.lock:
        if s.frames >= MAX_FRAMES:
            raise HTTPException(409, "Genoeg beelden.")
        s.frames += 1
        stem = s.dir / f"t{s.frames:04d}"
        stem.with_suffix(".jpg").write_bytes(body)
        stem.with_suffix(".json").write_text(json.dumps({
            "pose": pose.tolist(), "intrinsics": K.tolist(), "width": size[0], "height": size[1],
            "depth": [dm.shape[1], dm.shape[0]] if dm is not None else None}))
        s.poses.append(pose)
        if dm is not None:
            stem.with_suffix(".depth").write_bytes(dm.tobytes())
            if cf is not None:
                stem.with_suffix(".conf").write_bytes(cf.tobytes())
            pts, uv = backproject(dm, cf, K, size, pose)
            th, tw = thumb.shape[:2]
            col = thumb[np.clip((uv[:, 1] * th).astype(int), 0, th - 1), np.clip((uv[:, 0] * tw).astype(int), 0, tw - 1)]
            s.points.append(pts.astype(np.float32))
            s.colors.append(col)
            s.depth_frames += 1
    return {"frames": s.frames, "max_frames": MAX_FRAMES}


def preview_cloud(s: Session) -> bytes:
    """The LiDAR points so far, in the live view's format (see live.build_cloud):
    Z up, centred under the scanned thing, sized to fit the bed."""
    import struct
    from .live import MAX_POINTS, VIEW_RADIUS
    with s.lock:
        pts = np.concatenate(s.points) if s.points else np.zeros((0, 3), np.float32)
        col = np.concatenate(s.colors) if s.colors else np.zeros((0, 3), np.uint8)
        poses = np.array(s.poses)
    if len(pts) > MAX_POINTS:
        pick = np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False)
        pts, col = pts[pick], col[pick]
    up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)     # ARKit y-up -> FRAM3 z-up
    p = pts.astype(np.float64) @ up.T
    c = poses[:, :3, 3] @ up.T
    fwd = -poses[:, :3, 2] @ up.T
    cup = poses[:, :3, 1] @ up.T
    mid = np.median(p, axis=0) if len(p) else c.mean(axis=0)
    floor = float(np.percentile(p[:, 2], 2)) if len(p) > 20 else float(mid[2])
    origin = np.array([mid[0], mid[1], floor])
    radius = float(np.median(np.linalg.norm(c[:, :2] - origin[:2], axis=1))) or 1.0
    k = VIEW_RADIUS / max(radius, 0.05)
    cams = np.hstack([(c - origin) * k, fwd, cup]).astype("<f4")
    p = ((p - origin) * k).astype("<f4")
    return (b"F3LV" + struct.pack("<II", len(cams), len(p)) + cams.tobytes() + p.tobytes()
            + np.ascontiguousarray(col, dtype=np.uint8).tobytes())


@phone_app.post("/p/{token}/done")
def phone_done(token: str):
    s = _need(token)
    s.done = True
    return {"frames": s.frames}


@phone_app.get("/p/{token}/state")
def phone_state(token: str):
    s = _need(token)
    return {"frames": s.frames, "done": s.done}


# ------------------------------------------------------------- lifecycle
def start(data: Path) -> dict:
    """Open a pairing session and the local-network server. One at a time."""
    global _session, _server
    import shutil
    import uvicorn
    with _lock:
        stop()
        ip = lan_ip()
        if ip.startswith("127."):
            raise RuntimeError("Deze computer zit niet op een netwerk. Verbind met dezelfde wifi als je telefoon.")
        key, crt = certificate(data / "cert", ip)
        root = data / "phone"
        shutil.rmtree(root, ignore_errors=True)
        _session = Session(root)
        cfg = uvicorn.Config(phone_app, host="0.0.0.0", port=PORT, log_level="warning",
                             ssl_keyfile=str(key), ssl_certfile=str(crt))
        _server = uvicorn.Server(cfg)
        threading.Thread(target=_server.run, daemon=True).start()
        url = f"https://{ip}:{PORT}/p/{_session.token}"
        return _session.public() | {"url": url, "qr": qr_svg(url)}


def stop():
    """Close the session and stop listening on the network."""
    global _session, _server
    if _session is not None:
        _session.closed = True
    if _server is not None:
        _server.should_exit = True
        _server = None
