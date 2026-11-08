"""Import and export of 3D files.

Every mesh format goes through one in-memory representation: a trimesh.Scene.
trimesh handles the common formats natively. Formats it can't handle (FBX, USD,
Alembic, X3D, STEP) go through a Blender subprocess when Blender is installed,
or through cascadio for STEP.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class Format:
    ext: str
    name: str
    can_import: bool
    can_export: bool
    via: str  # "trimesh", "blender" or "cascadio"


# Order = order shown in the UI export menu.
FORMATS: list[Format] = [
    Format("stl", "STL (binair)", True, True, "trimesh"),
    Format("3mf", "3MF (slicers)", True, True, "trimesh"),
    Format("obj", "Wavefront OBJ", True, True, "trimesh"),
    Format("glb", "glTF binair (GLB)", True, True, "trimesh"),
    Format("gltf", "glTF (JSON)", True, True, "trimesh"),
    Format("ply", "PLY", True, True, "trimesh"),
    Format("off", "OFF", True, True, "trimesh"),
    Format("dae", "Collada (DAE)", True, True, "trimesh"),
    Format("fbx", "FBX", True, True, "blender"),
    Format("usdz", "USDZ", True, True, "blender"),
    Format("usd", "USD", True, True, "blender"),
    Format("abc", "Alembic", True, True, "blender"),
    Format("x3d", "X3D", True, True, "blender"),
    Format("step", "STEP (alleen import)", True, False, "cascadio"),
    Format("stp", "STEP (alleen import)", True, False, "cascadio"),
]
BY_EXT = {f.ext: f for f in FORMATS}

# Formats whose convention is Y-up and metres (glTF, FBX, USD, ...). FRAM3 works
# Z-up in millimetres like slicers, so these are converted on the way in and out.
Y_UP_METRES = {"glb", "gltf", "fbx", "usd", "usdz", "abc", "x3d", "dae"}
_Y_TO_Z = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
_Z_TO_Y = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])

# Formats that can carry several separate objects plus colors/textures.
SCENE_FORMATS = {"glb", "gltf", "3mf", "obj", "dae", "fbx", "usd", "usdz", "abc", "x3d"}


class FormatError(ValueError):
    pass


def blender_bin() -> str | None:
    env = os.environ.get("FRAM3_BLENDER")
    if env:
        return env
    if shutil.which("blender"):
        return "blender"
    if shutil.which("flatpak"):
        try:
            out = subprocess.run(["flatpak", "info", "org.blender.Blender"],
                                 capture_output=True, timeout=10)
            if out.returncode == 0:
                return "flatpak run org.blender.Blender"
        except Exception:
            pass
    return None


def available_formats() -> list[dict]:
    has_blender = blender_bin() is not None
    try:
        import cascadio  # noqa: F401
        has_cascadio = True
    except ImportError:
        has_cascadio = False
    out = []
    for f in FORMATS:
        ok = (f.via == "trimesh" or (f.via == "blender" and has_blender)
              or (f.via == "cascadio" and has_cascadio))
        out.append({"ext": f.ext, "name": f.name, "import": f.can_import and ok,
                    "export": f.can_export and ok, "via": f.via, "available": ok})
    return out


def ext_of(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


# --------------------------------------------------------------------- load

def load(data: bytes, filename: str, internal: bool = False) -> trimesh.Scene:
    """Read any supported file. internal=True means a GLB that FRAM3 itself wrote
    (already Z-up millimetres), so no axis/unit conversion is applied."""
    ext = ext_of(filename)
    fmt = BY_EXT.get(ext)
    if fmt is None:
        raise FormatError(f"Onbekend formaat: .{ext}")
    if fmt.via == "trimesh":
        loaded = trimesh.load(io.BytesIO(data), file_type=ext, force="scene")
    elif fmt.via == "cascadio":
        loaded = _load_step(data)
    else:
        loaded = _blender_convert_in(data, ext)
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    if not scene.geometry:
        raise FormatError("Het bestand bevat geen 3D-geometrie.")
    if not internal and ext in Y_UP_METRES:
        scene.apply_transform(_Y_TO_Z)
        if max(scene.extents) < 5:  # almost certainly metres: a real object under 5 mm is rare
            scene.apply_scale(1000)
    return scene


def _load_step(data: bytes) -> trimesh.Scene:
    try:
        import cascadio
    except ImportError as e:
        raise FormatError("STEP-import vereist het pakket 'cascadio' (pip install cascadio).") from e
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td, "in.step"), Path(td, "out.glb")
        src.write_bytes(data)
        cascadio.step_to_glb(str(src), str(dst), 0.1, 0.5)
        return trimesh.load(str(dst), force="scene")


# --------------------------------------------------------------------- save

def save(scene: trimesh.Scene, ext: str) -> bytes:
    """Write a Z-up millimetre scene. Y-up/metre formats are converted."""
    fmt = BY_EXT.get(ext)
    if fmt is None or not fmt.can_export:
        raise FormatError(f"Exporteren naar .{ext} wordt niet ondersteund.")
    if ext in Y_UP_METRES:
        scene = scene.copy()
        scene.apply_transform(_Z_TO_Y)
        scene.apply_scale(0.001)
    if fmt.via == "blender":
        glb = scene.export(file_type="glb")
        return _blender_convert_out(glb, ext)
    if ext == "gltf":
        # Embed buffers so the result is one self-contained file.
        files = scene.export(file_type="gltf", embed_buffers=True)
        out = files["model.gltf"] if isinstance(files, dict) else files
        return out.encode() if isinstance(out, str) else out
    if ext in ("stl", "ply", "off", "dae"):
        # Single-mesh formats: bake transforms and merge.
        out = scene_to_mesh(scene).export(file_type=ext)
        return out.encode() if isinstance(out, str) else out
    out = scene.export(file_type=ext)
    return out.encode() if isinstance(out, str) else out


def scene_to_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    meshes = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh)]
    if not meshes:
        raise FormatError("Geen mesh gevonden om te exporteren.")
    return trimesh.util.concatenate(meshes)


# ------------------------------------------------------------------ blender

_BLENDER_IMPORT = {
    "fbx": "bpy.ops.import_scene.fbx(filepath=src)",
    "usd": "bpy.ops.wm.usd_import(filepath=src)",
    "usdz": "bpy.ops.wm.usd_import(filepath=src)",
    "abc": "bpy.ops.wm.alembic_import(filepath=src)",
    "x3d": "bpy.ops.import_scene.x3d(filepath=src)",
    "glb": "bpy.ops.import_scene.gltf(filepath=src)",
}
_BLENDER_EXPORT = {
    "fbx": "bpy.ops.export_scene.fbx(filepath=dst, path_mode='COPY', embed_textures=True)",
    "usd": "bpy.ops.wm.usd_export(filepath=dst)",
    "usdz": "bpy.ops.wm.usd_export(filepath=dst)",
    "abc": "bpy.ops.wm.alembic_export(filepath=dst)",
    "x3d": "bpy.ops.export_scene.x3d(filepath=dst)",
    "glb": "bpy.ops.export_scene.gltf(filepath=dst, export_format='GLB')",
}


def _run_blender(src_ext: str, dst_ext: str, data: bytes) -> bytes:
    exe = blender_bin()
    if exe is None:
        raise FormatError(f".{src_ext}/.{dst_ext} vereist Blender. Installeer Blender of zet FRAM3_BLENDER.")
    with tempfile.TemporaryDirectory(dir=os.environ.get("FRAM3_TMP")) as td:
        src, dst = Path(td, f"in.{src_ext}"), Path(td, f"out.{dst_ext}")
        src.write_bytes(data)
        script = Path(td, "convert.py")
        script.write_text(
            "import bpy\n"
            f"src, dst = {str(src)!r}, {str(dst)!r}\n"
            "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
            f"{_BLENDER_IMPORT[src_ext]}\n"
            f"{_BLENDER_EXPORT[dst_ext]}\n"
        )
        cmd = exe.split() + ["--background", "--factory-startup", "--python", str(script)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if not dst.exists():
            tail = (proc.stderr or proc.stdout)[-800:]
            raise FormatError(f"Blender-conversie mislukt: {tail}")
        return dst.read_bytes()


def _blender_convert_in(data: bytes, ext: str) -> trimesh.Scene:
    glb = _run_blender(ext, "glb", data)
    return trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")


def _blender_convert_out(glb: bytes, ext: str) -> bytes:
    return _run_blender("glb", ext, glb)


def convert(data: bytes, src_name: str, dst_ext: str, internal: bool = False) -> bytes:
    return save(load(data, src_name, internal), dst_ext)
