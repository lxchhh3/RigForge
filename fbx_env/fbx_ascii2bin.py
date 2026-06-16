"""
ASCII -> Binary FBX converter using Autodesk FBX SDK Python bindings.
Mirror of fbx_bin2ascii.py with the writer selection inverted.

Usage:
    python fbx_ascii2bin.py <input_ascii.fbx> [output_binary.fbx]
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict, deque
from pathlib import Path

try:
    import fbx
    import FbxCommon
except ImportError as e:
    sys.stderr.write(
        "FATAL: fbx / FbxCommon not importable in this Python.\n"
        f"  sys.executable: {sys.executable}\n"
        f"  error: {e}\n"
    )
    sys.exit(2)


def fail(msg: str, code: int = 1) -> None:
    sys.stderr.write(f"FATAL: {msg}\n")
    sys.exit(code)


def _parse_deform_percents(ascii_text: str) -> dict:
    """Map (mesh_name, channel_name) -> deque of DeformPercent values (ASCII
    order), by walking BlendShapeChannel -> BlendShape -> Geometry -> Mesh.

    The FBX SDK importer does NOT read the ASCII `DeformPercent:` leaf into the
    channel property (it treats it as an animation-driven runtime value and
    resets it to 0). So we recover the intended values from the ASCII text and
    re-apply them to the SDK channels before export — otherwise every baked
    morph weight (factory bakes AND user overrides) is silently lost.
    """
    src = ascii_text
    model_name = {int(i): n for i, n in
                  re.findall(r'\tModel: (\d+), "Model::([^"]*)", "Mesh"', src)}
    bs_deformers = {int(i) for i in
                    re.findall(r'\tDeformer: (\d+), "[^"]*", "BlendShape" \{', src)}
    geom_ids = {int(i) for i in re.findall(r'\tGeometry: (\d+),', src)}

    dsts = defaultdict(list)
    for a, b in re.findall(r'C: "OO",\s*(\d+)\s*,\s*(\d+)', src):
        dsts[int(a)].append(int(b))

    def mesh_of_channel(cid: int):
        for bs in dsts.get(cid, []):
            if bs in bs_deformers:
                for geo in dsts.get(bs, []):
                    if geo in geom_ids:
                        for mdl in dsts.get(geo, []):
                            if mdl in model_name:
                                return model_name[mdl]
        return None

    out = defaultdict(deque)
    for m in re.finditer(
        r'\tDeformer: (\d+), "SubDeformer::([^"]*)", "BlendShapeChannel" \{', src
    ):
        cid, name = int(m.group(1)), m.group(2)
        end = src.find("\n\t}", m.start())
        block = src[m.start(): end if end != -1 else m.start() + 4000]
        dp = re.search(r"DeformPercent:\s*([-\d.]+)", block)
        mesh = mesh_of_channel(cid)
        out[(mesh, name)].append(float(dp.group(1)) if dp else 0.0)
    return out


def _restore_deform_percents(scene, src: Path) -> None:
    """Re-apply ASCII DeformPercent values onto the imported SDK channels."""
    try:
        ascii_text = src.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[warn] could not re-read ASCII for DeformPercent restore: {e}", flush=True)
        return
    wanted = _parse_deform_percents(ascii_text)
    if not wanted:
        return
    applied = nonzero = 0

    def walk(node):
        nonlocal applied, nonzero
        attr = node.GetNodeAttribute()
        if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
            mesh = node.GetMesh()
            for di in range(mesh.GetDeformerCount(fbx.FbxDeformer.EDeformerType.eBlendShape)):
                bs = mesh.GetDeformer(di, fbx.FbxDeformer.EDeformerType.eBlendShape)
                for ci in range(bs.GetBlendShapeChannelCount()):
                    ch = bs.GetBlendShapeChannel(ci)
                    q = wanted.get((node.GetName(), ch.GetName()))
                    if not q:
                        continue
                    val = q.popleft()
                    ch.DeformPercent.Set(float(val))
                    applied += 1
                    if val:
                        nonzero += 1
        for i in range(node.GetChildCount()):
            walk(node.GetChild(i))

    walk(scene.GetRootNode())
    print(f"[info] restored DeformPercent on {applied} channels ({nonzero} non-zero)", flush=True)


def convert(src: Path, dst: Path) -> None:
    if not src.is_file():
        fail(f"input file not found: {src}")
    if dst.exists() and src.resolve() == dst.resolve():
        fail("refusing to overwrite source with itself")

    # ---- Sanity-check that input really is ASCII ----
    with open(src, "rb") as f:
        head = f.read(32)
    if head.startswith(b"Kaydara FBX Binary"):
        fail(f"input {src} is already binary — use the bin->ascii script instead")
    if not head.lstrip().startswith(b";"):
        fail(f"input {src} does not look like ASCII FBX (head={head!r})")

    manager, scene = FbxCommon.InitializeSdkObjects()
    if manager is None or scene is None:
        fail("FbxCommon.InitializeSdkObjects() returned None")

    # ---- Import ----
    importer = fbx.FbxImporter.Create(manager, "")
    if not importer.Initialize(str(src), -1, manager.GetIOSettings()):
        fail(f"importer.Initialize failed: {importer.GetStatus().GetErrorString()}")

    reader_id = importer.GetFileFormat()
    reader_desc = manager.GetIOPluginRegistry().GetReaderFormatDescription(reader_id)
    print(f"[info] detected reader: {reader_desc}", flush=True)

    ios = manager.GetIOSettings()
    for flag in (
        fbx.EXP_FBX_MATERIAL, fbx.EXP_FBX_TEXTURE, fbx.EXP_FBX_EMBEDDED,
        fbx.EXP_FBX_SHAPE, fbx.EXP_FBX_GOBO, fbx.EXP_FBX_ANIMATION,
        fbx.EXP_FBX_GLOBAL_SETTINGS,
    ):
        ios.SetBoolProp(flag, True)

    if not importer.Import(scene):
        fail(f"importer.Import failed: {importer.GetStatus().GetErrorString()}")

    try:
        v_major, v_minor, v_revision = importer.GetFileVersion()
        print(f"[info] source FBX version: {v_major}.{v_minor}.{v_revision}", flush=True)
    except Exception:
        v_major = v_minor = v_revision = None

    importer.Destroy()

    # ---- Re-apply DeformPercent (SDK drops it on import) ----
    _restore_deform_percents(scene, src)

    # ---- Export (binary) ----
    exporter = fbx.FbxExporter.Create(manager, "")

    # Find the BINARY writer explicitly. The default writer (id 0) is typically
    # binary too, but we don't want to depend on that — we match by description
    # and reject any candidate whose description includes "ascii".
    plugin_registry = manager.GetIOPluginRegistry()
    binary_id = -1
    for i in range(plugin_registry.GetWriterFormatCount()):
        desc = plugin_registry.GetWriterFormatDescription(i)
        d = desc.lower()
        if "fbx" in d and "binary" in d and "encrypted" not in d:
            binary_id = i
            print(f"[info] selected writer #{i}: {desc}", flush=True)
            break
    if binary_id == -1:
        # Some SDK builds describe the binary writer simply as "FBX (*.fbx)".
        for i in range(plugin_registry.GetWriterFormatCount()):
            desc = plugin_registry.GetWriterFormatDescription(i)
            d = desc.lower()
            if "fbx" in d and "ascii" not in d and "encrypted" not in d:
                binary_id = i
                print(f"[info] selected writer #{i} (fallback): {desc}", flush=True)
                break
    if binary_id == -1:
        fail("could not locate a non-ASCII FBX writer")

    if not exporter.Initialize(str(dst), binary_id, manager.GetIOSettings()):
        fail(f"exporter.Initialize failed: {exporter.GetStatus().GetErrorString()}")

    SCHEMA_TO_VERSION_STR = {
        (7, 3): "FBX201300", (7, 4): "FBX201400", (7, 5): "FBX201600",
        (7, 6): "FBX201800", (7, 7): "FBX202000",
    }
    if v_major is not None:
        version_str = SCHEMA_TO_VERSION_STR.get((v_major, v_minor))
        if version_str is None:
            fail(f"unknown source FBX schema {v_major}.{v_minor} — refusing to silently downgrade/upgrade")
        if not exporter.SetFileExportVersion(version_str, fbx.FbxSceneRenamer.ERenamingMode.eNone):
            fail(f"SetFileExportVersion({version_str}) returned False")
        print(f"[info] export pinned to {version_str}", flush=True)

    for flag in (
        fbx.EXP_FBX_MATERIAL, fbx.EXP_FBX_TEXTURE, fbx.EXP_FBX_EMBEDDED,
        fbx.EXP_FBX_SHAPE, fbx.EXP_FBX_GOBO, fbx.EXP_FBX_ANIMATION,
        fbx.EXP_FBX_GLOBAL_SETTINGS,
    ):
        ios.SetBoolProp(flag, True)

    if not exporter.Export(scene):
        fail(f"exporter.Export failed: {exporter.GetStatus().GetErrorString()}")

    exporter.Destroy()
    manager.Destroy()

    # ---- Verify output really is binary ----
    with open(dst, "rb") as f:
        head = f.read(23)
    if not head.startswith(b"Kaydara FBX Binary"):
        fail(f"output {dst} does not have the Kaydara binary magic (head={head!r})")
    print(f"[ok]  wrote BINARY FBX: {dst} ({dst.stat().st_size:,} bytes)", flush=True)


def main(argv: list[str]) -> int:
    if len(argv) < 2 or len(argv) > 3:
        print(__doc__)
        return 64
    src = Path(argv[1]).resolve()
    dst = Path(argv[2]).resolve() if len(argv) == 3 else src.with_name(src.stem + "_bin.fbx")
    convert(src, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
