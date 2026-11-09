"""FRAM3 local server: serves the editor and the conversion/repair/scan API."""
from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import formats
from .reconstruct import Job, Tools, run_job, PRESETS, VIDEO_EXT, PHOTO_EXT
from .repair import RepairOptions, repair

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DATA = Path(os.environ.get("FRAM3_DATA", Path.home() / ".local/share/fram3"))
JOBS_DIR = DATA / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD = 4 * 1024**3  # 4 GB per file

app = FastAPI(title="FRAM3")
jobs: dict[str, Job] = {}


def _load_jobs():
    """Pick up scans from earlier runs, so interrupted ones can be resumed."""
    for d in sorted(JOBS_DIR.iterdir()) if JOBS_DIR.exists() else []:
        try:
            if (d / "job.json").exists():
                jobs[d.name] = Job.load(d)
            elif (d / "input").is_dir():
                # Scan from a version without job.json: done if result.glb exists, else interrupted.
                job = Job(d.name, d, os.environ.get("FRAM3_RESUME_PRESET", "normaal"))
                if (d / "result.glb").exists():
                    job.status, job.result, job.progress, job.stage = "klaar", str(d / "result.glb"), 1.0, "Klaar"
                else:
                    job.status, job.error = "onderbroken", "De scan is onderbroken. Je kunt hem hervatten."
                job.save()
                jobs[d.name] = job
        except Exception:
            pass


_load_jobs()


def _err(e: Exception, code: int = 400):
    raise HTTPException(status_code=code, detail=str(e))


@app.get("/api/status")
def status():
    return {"formats": formats.available_formats(), "tools": Tools.detect().summary(),
            "presets": list(PRESETS)}


@app.post("/api/import")
async def import_file(file: UploadFile = File(...)):
    """Any supported file -> GLB the editor can show."""
    data = await file.read()
    try:
        scene = formats.load(data, file.filename or "upload.bin")
        glb = scene.export(file_type="glb")
    except Exception as e:
        _err(e)
    return Response(glb, media_type="model/gltf-binary")


@app.post("/api/export")
async def export_file(file: UploadFile = File(...), to: str = Form(...), name: str = Form("model")):
    """GLB from the editor -> any export format."""
    to = to.lower().lstrip(".")
    data = await file.read()
    try:
        out = formats.convert(data, "scene.glb", to, internal=True)
    except Exception as e:
        _err(e)
    safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip() or "model"
    return Response(out, media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{safe}.{to}"'})


@app.post("/api/repair")
async def repair_mesh(file: UploadFile = File(...), keep_largest: bool = Form(True),
                      remove_floor: bool = Form(False), fill_holes: bool = Form(True),
                      make_watertight: bool = Form(True), target_faces: int | None = Form(None),
                      target_size_mm: float | None = Form(None), drop_to_bed: bool = Form(False)):
    data = await file.read()
    try:
        scene = formats.load(data, "scene.glb", internal=True)
        mesh = formats.scene_to_mesh(scene)
        fixed, rep = repair(mesh, RepairOptions(keep_largest, remove_floor, fill_holes, make_watertight,
                                                 target_faces or None, target_size_mm or None, drop_to_bed))
        glb = fixed.export(file_type="glb")
    except Exception as e:
        _err(e)
    return Response(glb, media_type="model/gltf-binary",
                    headers={"X-Fram3-Report": json.dumps({"steps": rep.steps, "before": rep.before,
                                                            "after": rep.after})})


@app.post("/api/boolean")
async def boolean_op(solids: UploadFile = File(...), holes: UploadFile | None = File(None)):
    """Union of all meshes in `solids`, minus all meshes in `holes` (GLB from the editor)."""
    from .repair import boolean
    try:
        def meshes(data: bytes):
            return [g for g in formats.load(data, "scene.glb", internal=True).dump() if len(g.faces)]
        sd = await solids.read()
        hd = await holes.read() if holes is not None else None
        if os.environ.get("FRAM3_DEBUG_DIR"):
            Path(os.environ["FRAM3_DEBUG_DIR"], "solids.glb").write_bytes(sd)
            if hd:
                Path(os.environ["FRAM3_DEBUG_DIR"], "holes.glb").write_bytes(hd)
        s = meshes(sd)
        h = meshes(hd) if hd else []
        result = boolean(s, h)
        if not len(result.faces):
            raise ValueError("Het resultaat is leeg: de gaten halen alles weg.")
        glb = result.export(file_type="glb")
    except Exception as e:
        _err(e)
    return Response(glb, media_type="model/gltf-binary")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    from .repair import stats
    data = await file.read()
    try:
        mesh = formats.scene_to_mesh(formats.load(data, "scene.glb", internal=True))
    except Exception as e:
        _err(e)
    return stats(mesh)


@app.post("/api/scan")
async def start_scan(files: list[UploadFile] = File(...), preset: str = Form("normaal"),
                     remove_floor: bool = Form(True)):
    if preset not in PRESETS:
        _err(ValueError(f"Onbekende kwaliteit: {preset}"))
    jid = uuid.uuid4().hex[:10]
    wd = JOBS_DIR / jid
    (wd / "input").mkdir(parents=True)
    paths = []
    for f in files:
        name = Path(f.filename or "file").name
        if Path(name).suffix.lower() not in VIDEO_EXT | PHOTO_EXT:
            continue
        dst = wd / "input" / name
        with dst.open("wb") as out:
            shutil.copyfileobj(f.file, out, 1024 * 1024)
        paths.append(dst)
    if not paths:
        shutil.rmtree(wd, ignore_errors=True)
        _err(ValueError("Geen video's of foto's gevonden in de upload."))
    job = Job(jid, wd, preset, remove_floor)
    job.save()
    jobs[jid] = job
    threading.Thread(target=run_job, args=(job, paths), daemon=True).start()
    return job.public()


@app.get("/api/scan/{jid}")
def scan_status(jid: str):
    job = jobs.get(jid) or _err(KeyError("Onbekende scan"), 404)
    return job.public()


@app.get("/api/scans")
def scan_list():
    out = []
    for jid, job in sorted(jobs.items(), key=lambda kv: kv[1].workdir.stat().st_mtime, reverse=True):
        out.append(job.public() | {"files": [p.name for p in job.inputs()]})
    return out[:20]


@app.post("/api/scan/{jid}/resume")
def scan_resume(jid: str):
    job = jobs.get(jid) or _err(KeyError("Onbekende scan"), 404)
    if job.status == "bezig":
        return job.public()
    inputs = job.inputs()
    if not inputs:
        _err(ValueError("De originele video/foto's van deze scan zijn er niet meer."))
    job._cancel.clear()
    job.log.append("== Hervatten ==")
    threading.Thread(target=run_job, args=(job, inputs), daemon=True).start()
    return job.public()


@app.post("/api/scan/{jid}/cancel")
def scan_cancel(jid: str):
    job = jobs.get(jid) or _err(KeyError("Onbekende scan"), 404)
    job.cancel()
    return job.public()


@app.get("/api/scan/{jid}/live")
def scan_live(jid: str):
    """What the scan is doing right now, for the live view."""
    from . import live
    job = jobs.get(jid) or _err(KeyError("Onbekende scan"), 404)
    return live.state(job.workdir, job.stage_key, job.stage_frac)


@app.get("/api/scan/{jid}/live/frame/{i}")
def scan_live_frame(jid: str, i: int):
    from . import live
    job = jobs.get(jid) or _err(KeyError("Onbekende scan"), 404)
    files = live.frame_files(job.workdir)
    if not files:
        _err(KeyError("Nog geen beelden"), 404)
    try:
        data = live.thumbnail(files[max(0, min(len(files) - 1, i))])
    except Exception as e:   # the file may just have been moved or is half written
        _err(e, 404)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/scan/{jid}/live/cloud")
def scan_live_cloud(jid: str):
    from . import live
    job = jobs.get(jid) or _err(KeyError("Onbekende scan"), 404)
    try:
        src = live.cloud_source(job.workdir)
        if not src:
            raise KeyError("Nog geen puntenwolk")
        data = live.build_cloud(src)
    except Exception as e:   # tools are writing these files while we read them
        _err(e, 404)
    return Response(data, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


@app.get("/api/scan/{jid}/result")
def scan_result(jid: str):
    job = jobs.get(jid)
    if not job or not job.result:
        _err(KeyError("Nog geen resultaat"), 404)
    return FileResponse(job.result, media_type="model/gltf-binary", filename=f"scan-{jid}.glb")


# ----------------------------------------------------------- phone scanning
@app.post("/api/phone/start")
def phone_start():
    from . import phone
    try:
        return phone.start(DATA)
    except Exception as e:
        _err(e)


@app.get("/api/phone/{token}")
def phone_status(token: str):
    from . import phone
    s = phone.current(token) or _err(KeyError("Koppeling verlopen"), 404)
    return s.public()


@app.get("/api/phone/{token}/frame/{i}")
def phone_frame(token: str, i: int):
    from . import live, phone
    s = phone.current(token) or _err(KeyError("Koppeling verlopen"), 404)
    files = s.files()
    if not files:
        _err(KeyError("Nog geen beelden"), 404)
    try:
        data = live.thumbnail(files[max(0, min(len(files) - 1, i))])
    except Exception as e:
        _err(e, 404)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/phone/{token}/finish")
def phone_finish(token: str, preset: str = Form("normaal"), remove_floor: bool = Form(True)):
    """Turn the frames the phone sent into a scan and stop listening on the network."""
    from . import phone
    s = phone.current(token) or _err(KeyError("Koppeling verlopen"), 404)
    if preset not in PRESETS:
        _err(ValueError(f"Onbekende kwaliteit: {preset}"))
    files = s.files()
    if len(files) < 8:
        _err(ValueError(f"Te weinig beelden ({len(files)}). Loop rustig rondom het object, minstens 20 beelden."))
    jid = uuid.uuid4().hex[:10]
    wd = JOBS_DIR / jid
    (wd / "input").mkdir(parents=True)
    paths = []
    for f in files:
        dst = wd / "input" / f"telefoon-{f.name[1:]}"
        shutil.move(f, dst)
        paths.append(dst)
    phone.stop()
    job = Job(jid, wd, preset, remove_floor)
    job.save()
    jobs[jid] = job
    threading.Thread(target=run_job, args=(job, paths), daemon=True).start()
    return job.public()


@app.post("/api/phone/stop")
def phone_stop():
    from . import phone
    phone.stop()
    return {"ok": True}


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")


def main():
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser(description="FRAM3 studio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8733)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not a.no_browser:
        import webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{a.host}:{a.port}")).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
