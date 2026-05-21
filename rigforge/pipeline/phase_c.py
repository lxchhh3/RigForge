"""Phase C — Armature merge (deterministic only).

Two paths:

  (a) donor_id == target_id  →  PASS-THROUGH
      Phase B already renamed donor-side names to canonical (= target-side)
      and dropped aux bones. Apply those edits and return the bytes; no
      structural merge is needed.

  (b) donor_id != target_id  →  CROSS-AVATAR MERGE
      Per PLAN.md Phase C steps 2–5:
        2. drop clothing's bone Models + armature root
        3. re-point cluster→bone connections to the target avatar's bone ids
           (by canonical role lookup)
        4. splice surviving clothing nodes into the target's Objects +
           Connections sections
        5. renumber clothing object ids with a safe offset to avoid id
           collisions with target ids

      The strip and id-offset passes both rewrite the clothing ASCII; we
      apply them in two passes (re-parsing between) so their edits cannot
      overlap on the same byte ranges. The splice is the final pass on the
      target ASCII.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rigforge.ascii_fbx.edits import (
    TextEdit,
    apply_edits,
    drop_bone_edits,
    drop_mesh_edits,
    rename_bone_edits,
    reparent_bone_edits,
)
from rigforge.ascii_fbx.lexer import parse as parse_fbx
from rigforge.ascii_fbx.merge import (
    bone_keep_strip_edits,
    id_offset_edits,
    splice_into_section_edit,
)
from rigforge.ascii_fbx.sections import SectionView, extract
from rigforge.avatars.registry import AvatarRegistry, CuratedAvatar
from rigforge.canonical.decisions import DecisionSet

from .edit_plan import EditPlan


class PhaseCError(RuntimeError):
    pass


@dataclass
class PhaseCResult:
    merged_ascii: bytes
    notes: list[str]


def run_phase_c(
    *,
    clothing_ascii: bytes,
    clothing_view: SectionView,
    donor_id: str,
    target_id: str,
    registry: AvatarRegistry,
    edit_plan: EditPlan,
    decisions: Optional[DecisionSet] = None,
    target_drop_bone_ids: Optional[set[int]] = None,
    target_drop_mesh_ids: Optional[set[int]] = None,
    drop_mesh_ids: Optional[set[int]] = None,
) -> PhaseCResult:
    """Apply EditPlan to clothing ASCII, then merge with target if needed.

    `clothing_view` was extracted from `clothing_ascii`; we keep it alongside
    to avoid re-parsing here.
    `decisions` is required for the cross-avatar branch.
    `target_drop_bone_ids` / `target_drop_mesh_ids` strip from the TARGET
    ASCII before splice (cross-merge only; ignored in pass-through, where
    the output IS the clothing).
    `drop_mesh_ids` strips meshes from the CLOTHING (both branches): the
    Mesh-Model + its Geometry + Skin + Clusters. The FE drives this — a
    modder uncheckes mesh boxes in the outliner, the FE sends ids.
    """
    notes: list[str] = []

    if donor_id == target_id:
        edits = _build_passthrough_edits(clothing_view, edit_plan, notes=notes)
        # Clothing-side mesh drops apply equally in pass-through.
        if drop_mesh_ids:
            for mid in drop_mesh_ids:
                edits.extend(drop_mesh_edits(clothing_view, mid))
            notes.append(f"drop_meshes: removed {len(drop_mesh_ids)} clothing meshes")
        edits = _dedupe_edits(edits)
        edited = apply_edits(clothing_ascii, edits)
        notes.append("merge: pass-through (donor==target)")
        if target_drop_bone_ids:
            notes.append(
                f"ignore: target_drop_bone_ids ({len(target_drop_bone_ids)}) "
                f"not applicable in pass-through"
            )
        if target_drop_mesh_ids:
            notes.append(
                f"ignore: target_drop_mesh_ids ({len(target_drop_mesh_ids)}) "
                f"not applicable in pass-through"
            )
        return PhaseCResult(merged_ascii=edited, notes=notes)

    if decisions is None:
        raise PhaseCError(
            "cross-avatar merge requires the DecisionSet (for canonical role "
            "lookup); orchestrator must pass `decisions=`"
        )

    target_avatar = registry.get(target_id)
    merged = _run_cross_merge(
        clothing_ascii=clothing_ascii,
        clothing_view=clothing_view,
        edit_plan=edit_plan,
        decisions=decisions,
        target_avatar=target_avatar,
        target_drop_bone_ids=target_drop_bone_ids or set(),
        target_drop_mesh_ids=target_drop_mesh_ids or set(),
        drop_mesh_ids=drop_mesh_ids or set(),
        notes=notes,
    )
    return PhaseCResult(merged_ascii=merged, notes=notes)


# ---------------------------------------------------------------------------
# Pass-through branch
# ---------------------------------------------------------------------------


def _build_passthrough_edits(
    view: SectionView,
    plan: EditPlan,
    *,
    notes: list[str],
) -> list[TextEdit]:
    edits: list[TextEdit] = []
    for bone_id in plan.drops:
        bone = view.bones.get(bone_id)
        if bone is None:
            notes.append(f"skip: drop target bone id={bone_id} not in view")
            continue
        edits.extend(drop_bone_edits(view, bone))
    for bone_id, new_name in plan.renames.items():
        bone = view.bones.get(bone_id)
        if bone is None:
            notes.append(f"skip: rename target bone id={bone_id} not in view")
            continue
        edits.extend(rename_bone_edits(view, bone, new_name))
    for bone_id, new_parent_id in plan.reparents.items():
        bone = view.bones.get(bone_id)
        if bone is None:
            notes.append(f"skip: reparent target bone id={bone_id} not in view")
            continue
        edits.extend(reparent_bone_edits(view, bone, new_parent_id))
    return edits


# ---------------------------------------------------------------------------
# Cross-avatar merge branch
# ---------------------------------------------------------------------------


def _run_cross_merge(
    *,
    clothing_ascii: bytes,
    clothing_view: SectionView,
    edit_plan: EditPlan,
    decisions: DecisionSet,
    target_avatar: CuratedAvatar,
    target_drop_bone_ids: set[int],
    target_drop_mesh_ids: set[int],
    drop_mesh_ids: set[int],
    notes: list[str],
) -> bytes:
    # 1. Build clothing_bone_id -> target_bone_id repoint table.
    #    Source: Phase B decisions where verdict=keep AND target has the role.
    repoint_table: dict[int, int] = {}
    kept_bone_ids: set[int] = set()
    for d in decisions.bones:
        if d.verdict != "keep":
            continue
        if d.role not in target_avatar.canonical_to_name:
            # Target doesn't carry this role (e.g. a secondary chain the
            # target lacks). Skip — its bone + clusters will be dropped at
            # offset time because no cluster repoint applies.
            continue
        try:
            tid = target_avatar.target_bone_id(d.role)
        except KeyError as e:
            notes.append(f"warn: target_bone_id({d.role!r}) lookup failed: {e}")
            continue
        repoint_table[d.model_id] = tid
        kept_bone_ids.add(d.model_id)

    # 2. Strip pass: drop kept-bone Models + their own upward parent refs.
    #    Connections where these bones are referenced AS PARENT stay so they
    #    can be repointed to target bone ids in the id_offset pass — that's
    #    how secondary chains (Hair, Breast, ornament bones) get reparented
    #    onto the target armature.
    strip_edits: list[TextEdit] = []
    for bid in kept_bone_ids:
        bone = clothing_view.bones.get(bid)
        if bone is None:
            notes.append(f"skip: kept bone id={bid} not in clothing view")
            continue
        strip_edits.extend(bone_keep_strip_edits(clothing_view, bone))
    for bid in edit_plan.drops:
        if bid in kept_bone_ids:
            continue  # already handled above
        bone = clothing_view.bones.get(bid)
        if bone is None:
            notes.append(f"skip: drop bone id={bid} not in clothing view")
            continue
        strip_edits.extend(drop_bone_edits(clothing_view, bone))
    # Strip clothing's armature root (Type=Null Models). Repoint children
    # pointing at it to scene RootNode (0) so the surviving sub-tree stays
    # attached to the scene after the armature root vanishes.
    for b in clothing_view.bones.values():
        if b.type_class == "Null":
            strip_edits.extend(bone_keep_strip_edits(clothing_view, b))
            repoint_table[b.model_id] = 0

    # Apply user mesh drops on the clothing side (e.g., outfit ships with a
    # cape mesh the modder doesn't want). Strips Mesh-Model + Geometry + Skin
    # + Clusters as a unit so no orphan refs survive into the splice.
    for mid in drop_mesh_ids:
        strip_edits.extend(drop_mesh_edits(clothing_view, mid))
    if drop_mesh_ids:
        notes.append(f"drop_meshes: removed {len(drop_mesh_ids)} clothing meshes")

    strip_edits = _dedupe_edits(strip_edits)
    stripped = apply_edits(clothing_ascii, strip_edits)
    notes.append(f"strip: dropped {len(kept_bone_ids)} kept-bone models + "
                 f"{len(edit_plan.drops) - len(kept_bone_ids & set(edit_plan.drops))} dropped bones")

    # 3. Offset pass: re-parse stripped, then offset every remaining clothing
    #    id by a safe constant, using repoint to redirect bone references.
    stripped_doc = parse_fbx(stripped)
    stripped_view = extract(stripped_doc)
    offset = _compute_safe_offset(target_avatar)
    notes.append(f"offset: shifting clothing ids by {offset}")
    offset_edits = id_offset_edits(stripped_view, offset=offset, repoint=repoint_table)
    offset_ascii = apply_edits(stripped, offset_edits)

    # 4. Splice pass: extract clothing's surviving Objects + Connections
    #    bodies, splice into target's same-named sections.
    final_doc = parse_fbx(offset_ascii)
    objects_payload = _section_body_bytes(final_doc, "Objects")
    connections_payload = _section_body_bytes(final_doc, "Connections")
    notes.append(
        f"splice: objects_payload={len(objects_payload)}B "
        f"connections_payload={len(connections_payload)}B"
    )

    target_ascii = target_avatar.load_ascii_bytes()
    # Apply user's target-side drops BEFORE the splice. Two flavors:
    #   - target_drop_bone_ids: kept for power users / older callers
    #   - target_drop_mesh_ids: what the FE outliner sends — strip the
    #     target's bundled outfit meshes (Maya ships with Cloth, Shoes, Hat, ...)
    if target_drop_bone_ids or target_drop_mesh_ids:
        target_view = target_avatar.load_ascii_view()
        target_drop_edits: list[TextEdit] = []
        for bid in target_drop_bone_ids:
            bone = target_view.bones.get(bid)
            if bone is None:
                notes.append(f"skip: target_drop bone id={bid} not in target view")
                continue
            target_drop_edits.extend(drop_bone_edits(target_view, bone))
        for mid in target_drop_mesh_ids:
            mesh = target_view.bones.get(mid)
            if mesh is None or mesh.type_class != "Mesh":
                notes.append(f"skip: target_drop_mesh id={mid} not a target mesh")
                continue
            target_drop_edits.extend(drop_mesh_edits(target_view, mid))
        target_drop_edits = _dedupe_edits(target_drop_edits)
        target_ascii = apply_edits(target_ascii, target_drop_edits)
        notes.append(
            f"target_drop: removed {len(target_drop_bone_ids)} bones + "
            f"{len(target_drop_mesh_ids)} meshes before splice"
        )

    target_doc = parse_fbx(target_ascii)
    splice_edits = [
        splice_into_section_edit(target_doc, "Objects", objects_payload),
        splice_into_section_edit(target_doc, "Connections", connections_payload),
    ]
    merged = apply_edits(target_ascii, splice_edits)
    return merged


def _compute_safe_offset(target_avatar: CuratedAvatar) -> int:
    """Pick an id offset larger than any id present in the target ASCII.

    Scans the target's Objects for max numeric id (cheap — we already cache
    the SectionView at registry-load).
    """
    view = target_avatar.load_ascii_view()
    candidates = [b.model_id for b in view.bones.values()]
    candidates.extend(view.clusters.keys())
    # Also scan ALL Objects (NodeAttributes, Geometries, Materials, Textures,
    # etc.) so the offset clears every id in the target file.
    objects = view.document.root("Objects")
    if objects is not None:
        from rigforge.ascii_fbx.merge import positioned_args
        source = view.document.source
        for obj in objects.children:
            for tok in positioned_args(obj.args_bytes(source)):
                if tok.kind == "num":
                    candidates.append(int(tok.value))
                    break
    max_id = max(candidates) if candidates else 0
    # Round up to a clean power-of-10 boundary above the max.
    boundary = 10 ** (len(str(max_id)) + 1)
    return boundary


def _dedupe_edits(edits: list[TextEdit]) -> list[TextEdit]:
    """Remove duplicate (start, end, replacement) triples.

    Many strip-pass edit-builders independently emit the same connection-drop
    when both endpoints are being stripped (e.g. parent-child between two kept
    bones; the strip is requested once per bone). Apply_edits rejects exact
    overlaps as a safety net — dedupe first so legitimate duplicates pass.
    """
    seen: set[tuple[int, int, bytes]] = set()
    out: list[TextEdit] = []
    for e in edits:
        key = (e.start, e.end, e.replacement)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _section_body_bytes(doc, section_name: str) -> bytes:
    """Return the bytes BETWEEN '{' and '}' of a top-level section (exclusive
    of the braces themselves)."""
    section = doc.root(section_name)
    if section is None:
        return b""
    if section.body_open is None or section.body_close is None:
        return b""
    return doc.source[section.body_open + 1 : section.body_close]
