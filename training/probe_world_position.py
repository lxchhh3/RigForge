"""Prototype: extract world-space bone positions from cluster TransformLink.

Goal — validate that we can cheaply get world coords per bone by reading the
4x4 bind matrix already present in every skin cluster, with no transform-chain
math required.

FBX `Cluster.TransformLink: *16 { a: ... }` is a row-major 4x4 world matrix
(the bone's transform at bind time). Elements [12], [13], [14] are the
world-space translation column.

For bones with no cluster (Blender's `_end` marker bones, root nulls, etc.),
fall back to walking the parent chain and summing Lcl Translation (rough; OK
for T-pose anchor estimates).

Outputs:
  - For PicodraTech, dump a sorted-by-Y table to inspect the spatial layout
  - Confirm Spine ~ 0.8m, Hips ~ 0.7m, Head ~ 1.5m, feet ~ 0
  - Confirm OuterA/B/C/D distinguish front/back by Z sign

Run:
    PYTHONPATH=. python training/probe_world_position.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from rigforge.ascii_fbx.lexer import parse, FBXNode
from rigforge.ascii_fbx.sections import extract, parse_args, _num


CLOTHING = Path("D:/2files/models/vrc/ASCII_models/clothing/Maya_PicodraTech/PicodraTech_MAYA_ascii.fbx")


def _parse_transform_link(cluster_node: FBXNode, source: bytes) -> list[float] | None:
    """Read the 16 floats from `TransformLink: *16 { a: ... }`. None if missing."""
    tl = cluster_node.child("TransformLink")
    if tl is None:
        return None
    # The matrix values live inside an 'a' sub-node.
    a = tl.child("a")
    if a is None:
        return None
    args = parse_args(a.args_bytes(source))
    nums = [_num(arg) for arg in args if arg[0] == "num"]
    if len(nums) < 16:
        return None
    return [float(x) for x in nums[:16]]


def _world_xyz_from_matrix(m16: list[float]) -> tuple[float, float, float]:
    """FBX row-major 4x4: translation row is at indices 12..14."""
    return (m16[12], m16[13], m16[14])


def main() -> int:
    print(f"loading {CLOTHING.name}...")
    doc = parse(CLOTHING.read_bytes())
    view = extract(doc)
    src = doc.source

    # cluster_id -> world_xyz (from TransformLink)
    cluster_world: dict[int, tuple[float, float, float]] = {}
    for cid, c in view.clusters.items():
        m = _parse_transform_link(c.node_ref, src)
        if m is not None:
            cluster_world[cid] = _world_xyz_from_matrix(m)

    # bone_id -> world_xyz (taking the first cluster's TransformLink)
    bone_world: dict[int, tuple[float, float, float]] = {}
    for b in view.limb_bones():
        for cid in b.cluster_ids:
            if cid in cluster_world:
                bone_world[b.model_id] = cluster_world[cid]
                break

    # Fallback: walk up the parent chain summing Lcl Translation until we hit
    # a bone whose world we know. This ignores rotation but works for T-pose.
    by_id = {b.model_id: b for b in view.limb_bones()}
    def estimate_world(bid: int, seen=None) -> tuple[float, float, float] | None:
        if bid in bone_world:
            return bone_world[bid]
        seen = seen or set()
        if bid in seen:
            return None
        seen.add(bid)
        b = by_id.get(bid)
        if b is None or b.parent_id is None:
            return None
        parent_world = estimate_world(b.parent_id, seen)
        if parent_world is None:
            return None
        # Add local translation (rotation ignored — rough)
        tx, ty, tz = b.translation_xyz
        wx, wy, wz = parent_world
        result = (wx + tx, wy + ty, wz + tz)
        bone_world[bid] = result
        return result

    for b in view.limb_bones():
        estimate_world(b.model_id)

    # Coverage stats
    bones_total = len(view.limb_bones())
    bones_with_world = sum(1 for b in view.limb_bones() if b.model_id in bone_world)
    print(f"\ncoverage: {bones_with_world}/{bones_total} bones have a world_xyz "
          f"({100*bones_with_world/bones_total:.1f}%)")

    # Y bounds (anchor for height_pct)
    ys = [bone_world[bid][1] for bid in bone_world]
    y_min, y_max = min(ys), max(ys)
    print(f"Y range: [{y_min:.3f}, {y_max:.3f}] (character height in scene units)")

    # Sanity table: known canonical bones at expected heights
    print("\n=== CANONICAL ANCHORS (sanity check) ===")
    anchors_to_check = [
        "Hips", "Spine", "Chest", "Neck", "Head",
        "UpperLeg.L", "UpperLeg.R", "LowerLeg.L", "LowerLeg.R", "Foot.L", "Foot.R",
        "Left elbow", "Right elbow",
    ]
    name_to_bone = {b.name: b for b in view.limb_bones()}
    print(f"{'bone':<22} {'world_xyz':<28} {'height_pct':<10}")
    for name in anchors_to_check:
        b = name_to_bone.get(name)
        if b is None or b.model_id not in bone_world:
            print(f"  {name:<20} (not present or no world)")
            continue
        wx, wy, wz = bone_world[b.model_id]
        pct = (wy - y_min) / (y_max - y_min) if y_max > y_min else 0
        print(f"  {name:<20} ({wx:+.3f}, {wy:+.3f}, {wz:+.3f})     {pct:.2f}")

    # The critical question: do OuterA/B/C/D distinguish by Z sign?
    print("\n=== OUTER SKIRT FAMILY (front/back disambiguation) ===")
    print(f"{'bone':<22} {'wx':>8} {'wy':>8} {'wz':>8}  inferred_side")
    outer_bones = sorted(
        [b for b in view.limb_bones() if b.name.startswith("Outer")
         and "." not in b.name.split("_")[0]],
        key=lambda b: b.name,
    )
    for b in outer_bones[:20]:
        if b.model_id not in bone_world:
            continue
        wx, wy, wz = bone_world[b.model_id]
        side = "L" if wx > 0.02 else "R" if wx < -0.02 else "C"
        fb = "F" if wz > 0.02 else "B" if wz < -0.02 else "C"
        print(f"  {b.name:<20} {wx:+.4f}  {wy:+.4f}  {wz:+.4f}  side={side} fb={fb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
