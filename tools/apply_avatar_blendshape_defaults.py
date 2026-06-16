"""Apply an avatar's baked blendshape defaults (data/<avatar>_blendshape_defaults.json)
to its ASCII FBX, idempotently.

Why this exists: curated avatars ship as a binary FBX whose source blendshape
DeformPercents are mostly 0, while the real "factory" weights live in the Unity
package. We transcribe those into a JSON config and bake them into the avatar's
ASCII (the file the pipeline actually splices). If that ASCII is ever regenerated
from the binary, the bake is lost — re-run this to restore it.

Usage:
    python tools/apply_avatar_blendshape_defaults.py [config.json] [avatar_ascii.fbx]

No args -> applies data/maya_blendshape_defaults.json to the 'maya' avatar's
registered ascii_fbx_path. Backs the target up as <name>.bak_<UTC> first.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from rigforge.ascii_fbx.edits import (
    apply_edits,
    iter_oo_connections,
    set_blend_shape_channel_deform_percent_edits,
)
from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract


def _channel_index(view):
    """(mesh_name, channel_name) -> channel_id, via channel->blendshape->geometry->mesh."""
    oo = list(iter_oo_connections(view))
    bs2geo = {s: d for s, d in oo if s in view.blend_shapes and d in view.geometries}
    geo_owner = view.geometry_owner_model
    mesh_name = {b.model_id: b.name for b in view.bones.values() if b.type_class == "Mesh"}
    idx = {}
    for ch in view.blend_shape_channels.values():
        mesh = mesh_name.get(geo_owner.get(bs2geo.get(ch.blend_shape_id)))
        idx[(mesh, ch.name)] = ch.channel_id
    return idx


def apply_defaults(config_path: Path, ascii_path: Path) -> int:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    raw = ascii_path.read_bytes()
    view = extract(parse(raw))
    idx = _channel_index(view)

    edits, missing = [], []
    for d in cfg["defaults"]:
        cid = idx.get((d["mesh"], d["channel"]))
        if cid is None:
            missing.append(f'{d["mesh"]} :: {d["channel"]} (channel not found)')
            continue
        e = set_blend_shape_channel_deform_percent_edits(view, cid, float(d["deform_percent"]))
        if not e:
            missing.append(f'{d["mesh"]} :: {d["channel"]} (no DeformPercent node)')
            continue
        edits.extend(e)
        print(f'  set {d["mesh"]} :: {d["channel"]} = {d["deform_percent"]}')
    if missing:
        print("WARN unmatched (skipped):")
        for m in missing:
            print("  " + m)

    backup = ascii_path.with_name(
        ascii_path.name + f".bak_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    backup.write_bytes(raw)
    ascii_path.write_bytes(apply_edits(raw, edits))
    print(f"applied {len(edits)} DeformPercent edits to {ascii_path}  (backup: {backup.name})")
    return 0


def main(argv: list[str]) -> int:
    repo = Path(__file__).resolve().parent.parent
    config = Path(argv[1]) if len(argv) > 1 else repo / "data" / "maya_blendshape_defaults.json"
    if len(argv) > 2:
        ascii_path = Path(argv[2])
    else:
        from rigforge.avatars.registry import AvatarRegistry
        avatar_id = json.loads(config.read_text(encoding="utf-8"))["avatar_id"]
        avatar = AvatarRegistry.load_default().get(avatar_id)
        if not avatar.ascii_fbx_path:
            print(f"FATAL: avatar {avatar_id!r} has no ascii_fbx_path", file=sys.stderr)
            return 2
        ascii_path = Path(avatar.ascii_fbx_path)
    if not ascii_path.is_file():
        print(f"FATAL: avatar ascii not found: {ascii_path}", file=sys.stderr)
        return 2
    return apply_defaults(config, ascii_path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
