"""Phase C — Armature merge (deterministic only).

Always produces `target_avatar.ascii + clothing's meshes/clusters spliced in`.
The shape is identical in both branches; what differs is how clothing bones
get mapped onto target bones for the cluster repoint pass:

  (a) donor_id == target_id  →  NAME-BASED REPOINT
      Each clothing bone is matched to the target bone with the same name.
      Clothing-specific extras (ornament bones the target doesn't carry)
      survive via id_offset and ride along.

  (b) donor_id != target_id  →  ROLE-BASED REPOINT (LLM-driven)
      Phase B's DecisionSet tells us each clothing bone's canonical role;
      we look up the target's bone for that role in the avatar registry.

Steps shared by both branches:
  1. Strip clothing's redundant bone Models (the ones we're repointing).
  2. Apply user's clothing-side drops (LLM drops + UI mesh/channel drops +
     DeformPercent overrides). Dedup donor materials/blendshapes by name.
  3. Offset all remaining clothing object ids by a safe constant, using
     `repoint_table` to redirect specific ids to their target counterparts.
  4. Apply user's target-side drops + DeformPercent overrides on the target
     ASCII before splice.
  5. Splice clothing's surviving Objects + Connections into the target's
     same-named sections.

The strip + id-offset passes rewrite the clothing ASCII in two passes
(re-parsing between) so their edits cannot overlap on the same byte ranges.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rigforge.ascii_fbx.edits import (
    TextEdit,
    apply_edits,
    drop_blend_shape_channel_edits,
    drop_bone_edits,
    drop_mesh_edits,
    drop_node_edit,
    set_blend_shape_channel_deform_percent_edits,
)
from rigforge.ascii_fbx.lexer import parse as parse_fbx
from rigforge.ascii_fbx.merge import (
    bone_keep_strip_edits,
    id_offset_edits,
    splice_into_section_edit,
    zero_weight_leaf_bone_ids,
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
    drop_blend_shape_channel_ids: Optional[set[int]] = None,
    target_drop_blend_shape_channel_ids: Optional[set[int]] = None,
    blend_shape_channel_overrides: Optional[dict[int, float]] = None,
    target_blend_shape_channel_overrides: Optional[dict[int, float]] = None,
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
    `drop_blend_shape_channel_ids` / `target_drop_blend_shape_channel_ids`
    strip individual BlendShapeChannel SubDeformers (morphs) on the clothing
    and target sides respectively. The FE surfaces these so the modder can
    cull morphs they don't want, mirroring the mesh-drop pattern.
    `blend_shape_channel_overrides` / `target_blend_shape_channel_overrides`
    set the channel's DeformPercent (0-100), baking a baseline morph value
    into the assembled FBX so the modder gets, e.g., "always 30% BigBust"
    without needing an animation track to drive it. Channels in the drop
    set are dropped before overrides apply — override on a dropped id is a
    no-op (the node is gone).
    """
    notes: list[str] = []

    target_avatar = registry.get(target_id)
    target_view = target_avatar.load_ascii_view()

    # Build the donor→target bone repoint table. Two strategies depending on
    # whether the clothing was rigged for THIS curated avatar already.
    if donor_id == target_id:
        repoint_table, kept_bone_ids = _build_name_based_repoint(
            clothing_view, target_view, edit_plan, notes,
        )
        notes.append("merge: name-based repoint (donor==target)")
    else:
        if decisions is None:
            raise PhaseCError(
                "cross-avatar merge requires the DecisionSet (for canonical "
                "role lookup); orchestrator must pass `decisions=`"
            )
        repoint_table, kept_bone_ids = _build_role_based_repoint(
            decisions, target_avatar, notes,
        )
        notes.append("merge: role-based repoint (cross-avatar)")

    merged = _run_merge(
        clothing_ascii=clothing_ascii,
        clothing_view=clothing_view,
        edit_plan=edit_plan,
        target_avatar=target_avatar,
        target_view=target_view,
        repoint_table=repoint_table,
        kept_bone_ids=kept_bone_ids,
        target_drop_bone_ids=target_drop_bone_ids or set(),
        target_drop_mesh_ids=target_drop_mesh_ids or set(),
        drop_mesh_ids=drop_mesh_ids or set(),
        drop_blend_shape_channel_ids=drop_blend_shape_channel_ids or set(),
        target_drop_blend_shape_channel_ids=target_drop_blend_shape_channel_ids or set(),
        blend_shape_channel_overrides=blend_shape_channel_overrides or {},
        target_blend_shape_channel_overrides=target_blend_shape_channel_overrides or {},
        notes=notes,
    )
    return PhaseCResult(merged_ascii=merged, notes=notes)


def _build_name_based_repoint(
    clothing_view: SectionView,
    target_view: SectionView,
    edit_plan: EditPlan,
    notes: list[str],
) -> tuple[dict[int, int], set[int]]:
    """For each clothing bone whose name matches a target bone, map the
    clothing's id to the target's id. Used in the donor==target case where
    the clothing is rigged for THIS curated avatar already — no LLM
    classification needed.

    Bones marked for drop by `edit_plan.drops` (user_drop or LLM verdict=drop)
    are EXCLUDED from the repoint table so the strip pass routes them
    through drop_bone_edits (full removal) rather than bone_keep_strip_edits
    (just the Model, keep cluster connections for repoint).

    Returns (repoint_table, kept_bone_ids). kept_bone_ids is the subset of
    clothing bones whose Models get stripped during merge (they're
    redundant; target owns the canonical copy). Clothing bones with no
    matching name (clothing-specific ornament bones) stay — they get an
    id offset and ride along into the merged output.
    """
    target_by_name: dict[str, int] = {}
    for b in target_view.bones.values():
        if b.name and b.type_class == "LimbNode":
            target_by_name[b.name] = b.model_id
    dropped = set(edit_plan.drops)
    repoint_table: dict[int, int] = {}
    kept_bone_ids: set[int] = set()
    matched = 0
    total = 0
    for b in clothing_view.bones.values():
        if b.type_class != "LimbNode":
            continue
        total += 1
        if b.model_id in dropped:
            continue
        tid = target_by_name.get(b.name)
        if tid is None:
            continue
        repoint_table[b.model_id] = tid
        kept_bone_ids.add(b.model_id)
        matched += 1
    notes.append(f"name-repoint: matched {matched}/{total} clothing bones to target")
    return repoint_table, kept_bone_ids


def _build_role_based_repoint(
    decisions: DecisionSet,
    target_avatar: CuratedAvatar,
    notes: list[str],
) -> tuple[dict[int, int], set[int]]:
    """For each kept bone whose LLM-assigned role exists in the target's
    canonical_to_name, map clothing's id to target's id. Used in the
    cross-avatar case where bone names differ between donor and target."""
    repoint_table: dict[int, int] = {}
    kept_bone_ids: set[int] = set()
    for d in decisions.bones:
        if d.verdict != "keep":
            continue
        if d.role not in target_avatar.canonical_to_name:
            # Target doesn't carry this role (e.g. a secondary chain the
            # target lacks). Skip — the clothing bone will get an id offset
            # and survive in the merged output unattached to a target bone.
            continue
        try:
            tid = target_avatar.target_bone_id(d.role)
        except KeyError as e:
            notes.append(f"warn: target_bone_id({d.role!r}) lookup failed: {e}")
            continue
        repoint_table[d.model_id] = tid
        kept_bone_ids.add(d.model_id)
    return repoint_table, kept_bone_ids


# ---------------------------------------------------------------------------
# Merge body (shared by both branches)
# ---------------------------------------------------------------------------


def _run_merge(
    *,
    clothing_ascii: bytes,
    clothing_view: SectionView,
    edit_plan: EditPlan,
    target_avatar: CuratedAvatar,
    target_view: SectionView,
    repoint_table: dict[int, int],
    kept_bone_ids: set[int],
    target_drop_bone_ids: set[int],
    target_drop_mesh_ids: set[int],
    drop_mesh_ids: set[int],
    drop_blend_shape_channel_ids: set[int],
    target_drop_blend_shape_channel_ids: set[int],
    blend_shape_channel_overrides: dict[int, float],
    target_blend_shape_channel_overrides: dict[int, float],
    notes: list[str],
) -> bytes:
    # `repoint_table` + `kept_bone_ids` come pre-built from one of the two
    # repoint-table builders (name-based or role-based). Everything below is
    # strategy-agnostic — it just needs the donor→target id map.

    # 1. Strip pass: drop kept-bone Models + their own upward parent refs.
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

    # Material dedup. Donor materials whose `name` collides with the target
    # are stripped from the clothing side; their donor ids land in the
    # repoint table so id_offset_edits rewrites surviving clothing
    # connections (e.g. Geometry→Material) to point at the target's
    # already-present equivalents. Materials are geometry-independent so this
    # is safe — see _compute_dedup_repoint for why blendshapes are NOT deduped
    # this way.
    dedup_strip, dedup_repoint = _compute_dedup_repoint(
        donor=clothing_view, target=target_view, notes=notes
    )
    strip_edits.extend(dedup_strip)
    repoint_table.update(dedup_repoint)

    # Apply user channel drops on the clothing side. Each channel is dropped
    # independently — its owning BlendShape Deformer survives (it may own
    # other channels we're keeping).
    for cid in drop_blend_shape_channel_ids:
        strip_edits.extend(drop_blend_shape_channel_edits(clothing_view, cid))
    if drop_blend_shape_channel_ids:
        notes.append(
            f"drop_blend_shape_channels: removed "
            f"{len(drop_blend_shape_channel_ids)} clothing channels"
        )

    # Apply DeformPercent overrides on the clothing side. Skip channels in
    # the drop set — the node is about to vanish; overriding a doomed node
    # would produce overlapping edits.
    if blend_shape_channel_overrides:
        applied = 0
        for cid, pct in blend_shape_channel_overrides.items():
            if cid in drop_blend_shape_channel_ids:
                continue
            channel_edits = set_blend_shape_channel_deform_percent_edits(
                clothing_view, cid, pct,
            )
            if channel_edits:
                strip_edits.extend(channel_edits)
                applied += 1
        if applied:
            notes.append(
                f"override: set DeformPercent on {applied} clothing channels"
            )

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
    if (
        target_drop_bone_ids
        or target_drop_mesh_ids
        or target_drop_blend_shape_channel_ids
        or target_blend_shape_channel_overrides
    ):
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
        for cid in target_drop_blend_shape_channel_ids:
            ch = target_view.blend_shape_channels.get(cid)
            if ch is None:
                notes.append(
                    f"skip: target_drop_blend_shape_channel id={cid} not a target channel"
                )
                continue
            target_drop_edits.extend(drop_blend_shape_channel_edits(target_view, cid))
        target_override_applied = 0
        for cid, pct in target_blend_shape_channel_overrides.items():
            if cid in target_drop_blend_shape_channel_ids:
                continue
            channel_edits = set_blend_shape_channel_deform_percent_edits(
                target_view, cid, pct,
            )
            if channel_edits:
                target_drop_edits.extend(channel_edits)
                target_override_applied += 1
        target_drop_edits = _dedupe_edits(target_drop_edits)
        target_ascii = apply_edits(target_ascii, target_drop_edits)
        notes.append(
            f"target_drop: removed {len(target_drop_bone_ids)} bones + "
            f"{len(target_drop_mesh_ids)} meshes + "
            f"{len(target_drop_blend_shape_channel_ids)} channels + "
            f"override {target_override_applied} channels before splice"
        )

    target_doc = parse_fbx(target_ascii)
    splice_edits = [
        splice_into_section_edit(target_doc, "Objects", objects_payload),
        splice_into_section_edit(target_doc, "Connections", connections_payload),
    ]
    merged = apply_edits(target_ascii, splice_edits)

    # 6. Zero-weight bone sweep. Collaborator's verdict: cleanup of the
    #    merged armature should aggressively drop bones that contribute no
    #    skin weight to any vertex. Leaf-prune iteratively so dead chains
    #    (e.g. `_end` markers, unused IK helpers) collapse end-first without
    #    orphaning weighted descendants. Re-parses the merged bytes once.
    merged_view = extract(parse_fbx(merged))
    zero_weight_ids = zero_weight_leaf_bone_ids(merged_view)
    if zero_weight_ids:
        sweep_edits: list[TextEdit] = []
        for bid in zero_weight_ids:
            bone = merged_view.bones.get(bid)
            if bone is None:
                continue
            sweep_edits.extend(drop_bone_edits(merged_view, bone))
        sweep_edits = _dedupe_edits(sweep_edits)
        merged = apply_edits(merged, sweep_edits)
        notes.append(
            f"zero_weight_sweep: dropped {len(zero_weight_ids)} bones with "
            f"no skin influence (leaf-prune, iterative)"
        )
    else:
        notes.append("zero_weight_sweep: nothing to drop")

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


def _compute_dedup_repoint(
    *,
    donor: SectionView,
    target: SectionView,
    notes: list[str],
) -> tuple[list[TextEdit], dict[int, int]]:
    """Dedup donor materials and blendshape (channels + owners) against the
    target by NAME. Returns (strip_edits, repoint_table).

    For each collision the donor node is dropped (the target's stays) and the
    donor id maps to the target id so connections referencing the donor are
    redirected at offset time. No-collision donor entities are untouched —
    they ride along in the wholesale Objects splice and pick up the id offset.

    Symmetric with [[bone_keep_strip_edits]]: drop the donor node, leave the
    referencing connections for the offset pass to repoint.
    """
    strip_edits: list[TextEdit] = []
    repoint: dict[int, int] = {}
    source = donor.document.source

    # Materials by short name.
    target_mat_by_name: dict[str, int] = {m.name: mid for mid, m in target.materials.items()}
    mat_hits = 0
    for mid, mat in donor.materials.items():
        tid = target_mat_by_name.get(mat.name)
        if tid is None:
            continue
        strip_edits.append(drop_node_edit(mat.node_ref, source))
        repoint[mid] = tid
        mat_hits += 1
    if mat_hits:
        notes.append(f"dedup: dropped {mat_hits} donor materials colliding with target")

    # BlendShape deformers and channels are deliberately NOT deduped by name.
    # A BlendShape deformer is tied to a specific Geometry (it deforms that
    # mesh's vertices via per-index displacements). Donor's "BlendShape" on
    # the donor's Body mesh and target's "BlendShape" on the target's Body
    # mesh are different objects even when same-named — their indices reference
    # different vertex layouts. Conflating them by name produced two distinct
    # Blender import failures:
    #   (a) one target channel ending up with multiple shape geoms attached
    #       → "FBX in-between Shapes are not currently supported"
    #   (b) donor shape geom indices interpreted against target's vertex count
    #       → "IndexError: index N out of bounds for axis 0 with size M"
    # Letting each mesh keep its own blendshape graph fixes both. If the modder
    # wants to remove a redundant clothing mesh (and its morphs), they drop it
    # via the FE mesh-picker — drop_mesh_edits cascades the cleanup.

    return strip_edits, repoint


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
