"""
Binary -> ASCII FBX converter using Autodesk FBX SDK Python bindings.

Run inside the fbx_env conda env (Python 3.10) AFTER installing the
Autodesk FBX SDK Python bindings into that env's site-packages.

Usage:
    python fbx_bin2ascii.py <input.fbx> [output.fbx]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import fbx
    import FbxCommon
except ImportError as e:
    sys.stderr.write(
        "FATAL: fbx / FbxCommon not importable in this Python.\n"
        f"  sys.executable: {sys.executable}\n"
        f"  error: {e}\n"
        "Install the Autodesk FBX SDK Python bindings into this env first.\n"
    )
    sys.exit(2)


def fail(msg: str, code: int = 1) -> None:
    sys.stderr.write(f"FATAL: {msg}\n")
    sys.exit(code)


def convert(src: Path, dst: Path) -> None:
    if not src.is_file():
        fail(f"input file not found: {src}")
    if dst.exists() and src.resolve() == dst.resolve():
        fail("refusing to overwrite source with itself")

    manager, scene = FbxCommon.InitializeSdkObjects()
    if manager is None or scene is None:
        fail("FbxCommon.InitializeSdkObjects() returned None")

    # ---- Import (binary) ----
    importer = fbx.FbxImporter.Create(manager, "")
    if not importer.Initialize(str(src), -1, manager.GetIOSettings()):
        status = importer.GetStatus()
        fail(f"importer.Initialize failed: {status.GetErrorString()}")

    # Verify input file format
    reader_id = importer.GetFileFormat()
    reader_desc = manager.GetIOPluginRegistry().GetReaderFormatDescription(reader_id)
    print(f"[info] detected reader: {reader_desc}", flush=True)

    # Tell the importer to bring EVERYTHING in — no silent drops.
    ios = manager.GetIOSettings()
    for flag in (
        fbx.EXP_FBX_MATERIAL,
        fbx.EXP_FBX_TEXTURE,
        fbx.EXP_FBX_EMBEDDED,
        fbx.EXP_FBX_SHAPE,
        fbx.EXP_FBX_GOBO,
        fbx.EXP_FBX_ANIMATION,
        fbx.EXP_FBX_GLOBAL_SETTINGS,
    ):
        ios.SetBoolProp(flag, True)

    if not importer.Import(scene):
        status = importer.GetStatus()
        fail(f"importer.Import failed: {status.GetErrorString()}")

    # Capture source FBX file version so we can round-trip exactly.
    major = importer.GetFileVersion()[0] if hasattr(importer, "GetFileVersion") else None
    # FbxImporter.GetFileVersion() actually returns via out-params in C++; in Python use:
    try:
        v_major, v_minor, v_revision = importer.GetFileVersion()
        print(f"[info] source FBX version: {v_major}.{v_minor}.{v_revision}", flush=True)
    except Exception:
        v_major = v_minor = v_revision = None

    importer.Destroy()

    # ---- Export (ASCII) ----
    exporter = fbx.FbxExporter.Create(manager, "")

    # Find the ASCII writer explicitly. Description string is stable across SDK versions.
    plugin_registry = manager.GetIOPluginRegistry()
    ascii_id = -1
    writer_count = plugin_registry.GetWriterFormatCount()
    for i in range(writer_count):
        desc = plugin_registry.GetWriterFormatDescription(i)
        if "ascii" in desc.lower() and "fbx" in desc.lower():
            ascii_id = i
            print(f"[info] selected writer #{i}: {desc}", flush=True)
            break
    if ascii_id == -1:
        fail("could not locate an ASCII FBX writer in this SDK build")

    if not exporter.Initialize(str(dst), ascii_id, manager.GetIOSettings()):
        status = exporter.GetStatus()
        fail(f"exporter.Initialize failed: {status.GetErrorString()}")

    # Pin the target file version to the source's so we preserve schema.
    # Schema (v_major.v_minor) -> SDK version string mapping (per FBX SDK docs):
    SCHEMA_TO_VERSION_STR = {
        (7, 3): "FBX201300",
        (7, 4): "FBX201400",
        (7, 5): "FBX201600",
        (7, 6): "FBX201800",
        (7, 7): "FBX202000",
    }
    if v_major is not None:
        key = (v_major, v_minor)
        version_str = SCHEMA_TO_VERSION_STR.get(key)
        if version_str is None:
            fail(f"unknown source FBX schema {v_major}.{v_minor} — refusing to silently downgrade/upgrade")
        ok = exporter.SetFileExportVersion(version_str, fbx.FbxSceneRenamer.ERenamingMode.eNone)
        if not ok:
            fail(f"SetFileExportVersion({version_str}) returned False")
        print(f"[info] export pinned to {version_str}", flush=True)

    # Make the writer keep every channel.
    for flag in (
        fbx.EXP_FBX_MATERIAL,
        fbx.EXP_FBX_TEXTURE,
        fbx.EXP_FBX_EMBEDDED,
        fbx.EXP_FBX_SHAPE,
        fbx.EXP_FBX_GOBO,
        fbx.EXP_FBX_ANIMATION,
        fbx.EXP_FBX_GLOBAL_SETTINGS,
    ):
        ios.SetBoolProp(flag, True)

    if not exporter.Export(scene):
        status = exporter.GetStatus()
        fail(f"exporter.Export failed: {status.GetErrorString()}")

    exporter.Destroy()
    manager.Destroy()

    # ---- Verify output is actually ASCII ----
    with open(dst, "rb") as f:
        head = f.read(64)
    if head.startswith(b"Kaydara FBX Binary"):
        fail(f"output {dst} is still binary — writer selection failed silently")
    if not head.lstrip().startswith(b";"):
        # ASCII FBX files start with `; FBX <ver> project file` comment.
        print(f"[warn] output does not start with the expected ASCII comment; head={head!r}",
              flush=True)
    else:
        print(f"[ok]  wrote ASCII FBX: {dst} ({dst.stat().st_size:,} bytes)", flush=True)


def main(argv: list[str]) -> int:
    if len(argv) < 2 or len(argv) > 3:
        print(__doc__)
        return 64
    src = Path(argv[1]).resolve()
    if len(argv) == 3:
        dst = Path(argv[2]).resolve()
    else:
        dst = src.with_name(src.stem + "_ascii.fbx")
    convert(src, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
