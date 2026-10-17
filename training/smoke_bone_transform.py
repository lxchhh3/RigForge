"""Bone-move proof on a real file: raise Maya's own hat bone and watch it move.

The bone-move feature sets a bone's rest-pose local transform in the FBX (the
skin bind pose is untouched, so the mesh follows — same as dragging the bone
in Unity). This smoke proves the whole chain on the real Maya file:

  1. read the hat bone's current local transform (Unity frame, as the FE
     shows it) and ask for it `--offset` metres higher
  2. assemble Maya-as-clothing onto Maya (mock LLM, identity decisions,
     no API spend) with that as a TARGET-side bone move
  3. import the ORIGINAL and the OUTPUT binary in headless Blender and read
     the bone back: the posed head must rise by the offset (world Z, up in
     Blender), the rest head must not move (bind pose untouched)
  4. render front + side views of both so the lift can be eyeballed

Usage:
    PYTHONPATH=. python training/smoke_bone_transform.py [--bone "Hat Root"] [--offset 0.05]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rigforge.ascii_fbx.convert import bin_to_ascii_cached
from rigforge.ascii_fbx.lexer import parse as parse_fbx
from rigforge.ascii_fbx.sections import extract, model_local_transform
from rigforge.ascii_fbx.transform import LocalTransform, fbx_to_unity
from rigforge.avatars.registry import AvatarRegistry
from rigforge.canonical.decisions import Decision, DecisionSet
from rigforge.canonical.schema import CanonicalSchema
from rigforge.geometry.blender_edit import bone_report
from rigforge.llm.mock import MockLLMClient
from rigforge.pipeline.bone_transforms import BoneTransformEdit
from rigforge.pipeline.orchestrator import assemble

REPO_ROOT = Path(__file__).resolve().parent.parent
MAYA_FBX = Path("D:/2files/models/vrc/Maya/Maya_Ver1.02.2/Maya.fbx")


def _identity_decisions(view, registry) -> DecisionSet:
    av = registry.get("maya")
    display_to_role = {v: k for k, v in av.canonical_to_name.items()}
    bones = []
    for b in view.limb_bones():
        role = display_to_role.get(b.name)
        bones.append(Decision(
            model_id=b.model_id,
            role=role if role is not None else f"Secondary.{b.name}",
            verdict="keep",
        ))
    return DecisionSet(bones=bones, llm_model_id="smoke-bone-transform")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bone", default="Hat Root")
    ap.add_argument("--offset", type=float, default=0.05, help="metres, world up")
    ap.add_argument("--clothing", default=str(MAYA_FBX))
    args = ap.parse_args()

    src = Path(args.clothing)
    if not src.is_file():
        print(f"[skip] not found: {src}")
        return 2
    work = REPO_ROOT / "data" / "training" / "_bone_transform"
    work.mkdir(parents=True, exist_ok=True)
    out_fbx = work / f"{src.stem}_{args.bone.replace(' ', '_')}_up.fbx"

    registry = AvatarRegistry.load_default()
    schema = CanonicalSchema.load_default()

    # 1. current local transform of the bone, in the frame the FE shows
    ascii_path = bin_to_ascii_cached(src, work / "_fbx_ascii_cache")
    view = extract(parse_fbx(ascii_path.read_bytes()))
    ids = [b.model_id for b in view.limb_bones() if b.name == args.bone]
    if len(ids) != 1:
        print(f"[fail] bone {args.bone!r}: {len(ids)} matches")
        return 1
    cur_unity = fbx_to_unity(model_local_transform(view, ids[0]))
    px, py, pz = cur_unity.position
    new_pos = (px, py + args.offset, pz)
    print(f"[bone] {args.bone}: unity-local position {cur_unity.position} -> {new_pos}")

    # 2. assemble with the target-side move
    t0 = time.time()
    run = assemble(
        clothing_fbx=src,
        target_id="maya",
        out_fbx=out_fbx,
        registry=registry,
        schema=schema,
        llm_client=MockLLMClient(_identity_decisions(view, registry)),
        cache=None,
        work_dir=work / "work",
        ascii_cache_dir=work / "_fbx_ascii_cache",
        target_bone_transforms=[BoneTransformEdit(
            bone=args.bone, transform=LocalTransform(position=new_pos), frame="unity",
        )],
    )
    print(f"[assemble] {out_fbx.name} in {time.time() - t0:.0f}s")
    for n in run.notes:
        if "bone_transform" in n:
            print(f"   note: {n}")

    # 3 + 4. Blender: bone positions + renders, before and after
    before = bone_report(
        src, bones=[args.bone], preview_dir=work / "preview_before", frame_mesh="Body all",
    )
    after = bone_report(
        out_fbx, bones=[args.bone], preview_dir=work / "preview_after", frame_mesh="Body all",
    )
    b0, b1 = before["bones"].get(args.bone), after["bones"].get(args.bone)
    if not b0 or not b1:
        print(f"[fail] bone not found in Blender: before={bool(b0)} after={bool(b1)}")
        return 1
    d_pose = [round(a - b, 4) for a, b in zip(b1["pose_head_world"], b0["pose_head_world"])]
    d_rest = [round(a - b, 4) for a, b in zip(b1["head_world"], b0["head_world"])]
    print(f"[blender] posed head world: {b0['pose_head_world']} -> {b1['pose_head_world']}")
    print(f"[blender] delta posed = {d_pose}   delta rest = {d_rest}")
    print(f"[previews] {before['previews']}")
    print(f"[previews] {after['previews']}")

    # Blender is Z-up: the requested "up" offset must land on world Z of the
    # POSED head while the rest head (bind pose) stays put.
    ok_pose = abs(d_pose[2] - args.offset) < 1e-3 and abs(d_pose[0]) < 1e-3 and abs(d_pose[1]) < 1e-3
    ok_rest = all(abs(v) < 1e-6 for v in d_rest)
    summary = {
        "bone": args.bone, "offset": args.offset, "output": str(out_fbx),
        "delta_pose_world": d_pose, "delta_rest_world": d_rest,
        "ok_pose": ok_pose, "ok_rest": ok_rest,
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[result] pose moved as asked: {ok_pose}   bind pose untouched: {ok_rest}")
    return 0 if ok_pose else 1


if __name__ == "__main__":
    sys.exit(main())
