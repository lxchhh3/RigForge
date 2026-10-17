"""Headless Blender: report where named bones sit, optionally render.

Runs INSIDE Blender's bundled Python. Invoked by
`rigforge.geometry.blender_edit.bone_report` as:

    blender --background --factory-startup --python blender_env/bone_report.py -- \
        --in <src.fbx> --report <report.json> --bones "A,B,C" \
        [--frame-mesh <mesh name>] [--preview-dir <dir>]

For each requested bone (matched by exact name, then suffix-tolerant) the
report carries:

    head_world       armature REST head — Blender builds the rest pose from
                     the skin bind pose (cluster TransformLink), so a rest-pose
                     bone move in the FBX does NOT show here
    pose_head_world  the bone's Model transform applied as the pose — what the
                     viewport, a render, and Unity all show; a bone move shows
                     up as a delta here

Same import settings as geometry_edit.py so the two agree on orientation.
No export, no geometry change.
"""
import json
import os
import sys
import traceback

import bpy
import mathutils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_edit import (  # noqa: E402
    _argv,
    _ensure_fbx_addon,
    _find_mesh,
    _import_fbx,
    _mesh_objects,
    _parse,
    _render_preview,
    _reset_scene,
)


def _find_bone(arm, name):
    if name in arm.data.bones:
        return arm.data.bones[name]
    for b in arm.data.bones:
        if b.name.split(".")[0] == name or b.name.lower() == name.lower():
            return b
    return None


def _largest_mesh():
    best, best_span = None, -1.0
    for o in _mesh_objects():
        corners = [o.matrix_world @ mathutils.Vector(c) for c in o.bound_box]
        span = max(
            max(c[i] for c in corners) - min(c[i] for c in corners) for i in range(3)
        )
        if span > best_span:
            best, best_span = o, span
    return best


def main():
    args = _parse(_argv())
    src = args.get("in")
    report_path = args.get("report")
    if not src or not report_path:
        raise SystemExit("bone_report: --in and --report are required")
    wanted = [n for n in str(args.get("bones", "")).split(",") if n]

    _ensure_fbx_addon()
    _reset_scene()
    _import_fbx(src)
    # Make sure pose evaluation is current before reading pose bones.
    bpy.context.view_layer.update()

    bones_out = {}
    armatures = []
    for arm in bpy.data.objects:
        if arm.type != "ARMATURE":
            continue
        armatures.append(arm.name)
        for name in wanted:
            if name in bones_out:
                continue
            b = _find_bone(arm, name)
            if b is None:
                continue
            pb = arm.pose.bones.get(b.name)
            head_world = arm.matrix_world @ b.head_local
            tail_world = arm.matrix_world @ b.tail_local
            pose_head_world = arm.matrix_world @ pb.head if pb else head_world
            bones_out[name] = {
                "blender_name": b.name,
                "armature": arm.name,
                "parent": b.parent.name if b.parent else None,
                "head_world": [round(v, 6) for v in head_world],
                "tail_world": [round(v, 6) for v in tail_world],
                "pose_head_world": [round(v, 6) for v in pose_head_world],
            }
    missing = [n for n in wanted if n not in bones_out]

    previews = {}
    preview_dir = args.get("preview-dir")
    if preview_dir:
        target = _find_mesh(args.get("frame-mesh", "")) if args.get("frame-mesh") else None
        if target is None:
            target = _largest_mesh()
        if target is not None:
            for view in ("front", "side"):
                try:
                    png = os.path.join(preview_dir, f"{view}.png")
                    _render_preview(target, png, view=view)
                    previews[view] = png
                except Exception as exc:  # best-effort
                    previews[f"{view}_error"] = repr(exc)
            previews["framed_on"] = target.name

    report = {
        "op": "bone_report",
        "source": src,
        "blender_version": bpy.app.version_string,
        "armatures": armatures,
        "bones": bones_out,
        "missing": missing,
        "previews": previews,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[bone_report] {len(bones_out)}/{len(wanted)} bones found, "
          f"missing={missing} -> {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
