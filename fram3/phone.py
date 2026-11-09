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

from fastapi import FastAPI, HTTPException, Request
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

    def files(self) -> list[Path]:
        return sorted(self.dir.glob("t*.jpg"))

    def public(self) -> dict:
        return {"token": self.token, "connected": self.device is not None, "device": self.device,
                "frames": self.frames, "done": self.done,
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
