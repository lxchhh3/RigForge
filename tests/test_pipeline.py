"""Tests for individual pipeline phases.

The orchestrator + end-to-end smoke test lives in test_e2e.py (task #14) and
exercises the converter subprocess. Tests here stay in-memory (parse → phase →
assert) so they run fast without invoking fbx_env.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.ascii_fbx.edits import apply_edits
from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.avatars.registry import (
    AvatarRegistry,
    DonorIdentificationError,
)
from rigforge.canonical.decisions import Decision, DecisionSet
from rigforge.canonical.schema import CanonicalSchema
from rigforge.llm.mock import MockLLMClient
from rigforge.pipeline.edit_plan import EditPlan
from rigforge.pipeline.phase_a import run_phase_a
from rigforge.pipeline.phase_b import PhaseBError, run_phase_b
from rigforge.pipeline.phase_c import PhaseCError, run_phase_c


@pytest.fixture(scope="module")
def schema() -> CanonicalSchema:
    return CanonicalSchema.load_default()


@pytest.fixture(scope="module")
def registry() -> AvatarRegistry:
    return AvatarRegistry.load_default()


# ---------------------------------------------------------------------------
# Phase A
# ---------------------------------------------------------------------------


def test_phase_a_identifies_maya(maya_fbx_ascii: Path, registry, tmp_path: Path):
    result = run_phase_a(maya_fbx_ascii, registry, work_dir=tmp_path)
    assert result.donor_id == "maya"
    assert result.score == 1.0
    assert result.view.limb_bones(), "Phase A should produce a section view"


def test_phase_a_rejects_unfamiliar_rig(registry, tmp_path: Path):
    """A clothing FBX rigged for an unknown booth base → hard-fail with
    top-3 candidates."""
    fake = tmp_path / "fake.fbx"
    # Minimal ASCII FBX with completely foreign bone names
    fake.write_bytes(b"""\
; FBX 7.4.0 project file
Objects:  {
\tModel: 1, "Model::J_Bip_C_Hips", "LimbNode" {
\t}
\tModel: 2, "Model::J_Bip_C_Spine", "LimbNode" {
\t}
}
Connections:  {
\tC: "OO",2,1
\tC: "OO",1,0
}
""")
    with pytest.raises(DonorIdentificationError) as exc:
        run_phase_a(fake, registry, work_dir=tmp_path)
    assert exc.value.top_candidates


# ---------------------------------------------------------------------------
# Phase B
# ---------------------------------------------------------------------------


def _maya_identity_decisions(view) -> DecisionSet:
    """Build a decision set that maps each Maya bone to its canonical role
    (where applicable) with verdict=keep. Other bones go aux/drop."""
    schema = CanonicalSchema.load_default()
    reg = AvatarRegistry.load_default()
    av = reg.get("maya")
    # Maya already uses these display names → canonical_to_name maps role→display
    # Build display_name → canonical_role
    display_to_role = {v: k for k, v in av.canonical_to_name.items()}

    bones = []
    for b in view.limb_bones():
        role = display_to_role.get(b.name)
        if role is not None:
            bones.append(Decision(model_id=b.model_id, role=role, verdict="keep"))
        else:
            # Treat anything we don't have a canonical mapping for as a
            # secondary kept bone (keep its own name as the role; passes
            # unique_role since secondaries can repeat).
            bones.append(Decision(model_id=b.model_id, role=f"Secondary.{b.name}",
                                   verdict="keep"))
    return DecisionSet(bones=bones, llm_model_id="test-identity")


def test_phase_b_clean_run_produces_no_renames(maya_fbx_ascii: Path,
                                                 registry, schema):
    view = extract(parse(maya_fbx_ascii.read_bytes()))
    decisions = _maya_identity_decisions(view)
    mock = MockLLMClient(decisions, model_id="test-identity")

    result = run_phase_b(
        view=view,
        donor_id="maya",
        target_id="maya",
        registry=registry,
        schema=schema,
        llm_client=mock,
    )
    # Bones already named per Maya's canonical_to_name → renames map is empty
    assert result.edit_plan.renames == {}
    assert not result.edit_plan.drops
    assert result.cache_hit is False  # no cache passed


def test_phase_b_cache_round_trip(maya_fbx_ascii: Path, registry, schema, tmp_path):
    from rigforge.cache.store import DecisionCache

    view = extract(parse(maya_fbx_ascii.read_bytes()))
    decisions = _maya_identity_decisions(view)
    mock = MockLLMClient(decisions, model_id="test-identity")
    cache = DecisionCache(root=tmp_path)

    r1 = run_phase_b(view=view, donor_id="maya", target_id="maya",
                     registry=registry, schema=schema,
                     llm_client=mock, cache=cache)
    assert r1.cache_hit is False
    # Second run: should hit cache
    r2 = run_phase_b(view=view, donor_id="maya", target_id="maya",
                     registry=registry, schema=schema,
                     llm_client=mock, cache=cache)
    assert r2.cache_hit is True
    assert r2.cache_key == r1.cache_key


def test_phase_b_user_drop_overrides_keep(maya_fbx_ascii: Path, registry, schema):
    """A bone in user_drop_bone_ids gets verdict='drop' with
    drop_category='user_dropped', overriding what the LLM said."""
    view = extract(parse(maya_fbx_ascii.read_bytes()))
    decisions = _maya_identity_decisions(view)
    mock = MockLLMClient(decisions, model_id="test-identity")

    # Pick a non-canonical secondary bone the user wants to drop
    secondary = next(b for b in view.limb_bones() if b.name == "Tail")
    result = run_phase_b(
        view=view, donor_id="maya", target_id="maya",
        registry=registry, schema=schema, llm_client=mock,
        user_drop_bone_ids={secondary.model_id},
    )
    overridden = result.decisions.by_id()[secondary.model_id]
    assert overridden.verdict == "drop"
    assert overridden.drop_category == "user_dropped"
    assert secondary.model_id in result.edit_plan.drops


def test_phase_b_user_drop_cascades_to_subtree(maya_fbx_ascii: Path, registry, schema):
    """Dropping a chain root drops every descendant too — otherwise the
    descendants would dangle without a parent in the assembled rig."""
    view = extract(parse(maya_fbx_ascii.read_bytes()))
    decisions = _maya_identity_decisions(view)
    mock = MockLLMClient(decisions, model_id="test-identity")

    tail_root = next(b for b in view.limb_bones() if b.name == "Tail")
    # Tail has children (Tail.001 .. Tail.011) — find at least one
    assert tail_root.children_ids, "fixture broken: Tail should have children"
    descendant_id = tail_root.children_ids[0]

    result = run_phase_b(
        view=view, donor_id="maya", target_id="maya",
        registry=registry, schema=schema, llm_client=mock,
        user_drop_bone_ids={tail_root.model_id},
    )
    desc_decision = result.decisions.by_id()[descendant_id]
    assert desc_decision.verdict == "drop"
    assert desc_decision.drop_category == "user_dropped"


def test_phase_b_user_drop_doesnt_break_canonical(maya_fbx_ascii: Path, registry, schema):
    """User can drop a non-required bone but required canonical bones must
    survive — required_roles_present would otherwise fire and the user has
    no business removing Hips/Spine/etc. We don't enforce this defensively
    in the pre-filter; we trust the FE to gate it. But the smoke here is
    that dropping a SAFE secondary chain doesn't break the validators."""
    view = extract(parse(maya_fbx_ascii.read_bytes()))
    decisions = _maya_identity_decisions(view)
    mock = MockLLMClient(decisions, model_id="test-identity")

    # Drop a hair chain — safe
    ahoge_root = next(b for b in view.limb_bones() if b.name == "Ahoge.001")
    result = run_phase_b(
        view=view, donor_id="maya", target_id="maya",
        registry=registry, schema=schema, llm_client=mock,
        user_drop_bone_ids={ahoge_root.model_id},
    )
    # Required roles unaffected
    kept = result.decisions.kept_by_role()
    for r in schema.required_roles:
        assert r in kept and kept[r], f"required role {r} missing after user drop"


def test_phase_b_validation_failure_raises(maya_fbx_ascii: Path, registry, schema):
    """A decision set that violates unique_role (Hips assigned twice) must
    surface as PhaseBError after the re-prompt fails."""
    view = extract(parse(maya_fbx_ascii.read_bytes()))
    limbs = view.limb_bones()
    # Force a unique_role violation: assign two bones the same canonical role
    decisions = DecisionSet(bones=[
        Decision(model_id=limbs[0].model_id, role="Hips", verdict="keep"),
        Decision(model_id=limbs[1].model_id, role="Hips", verdict="keep"),
        *[Decision(model_id=b.model_id, role=f"Secondary.{b.name}", verdict="keep")
          for b in limbs[2:]],
    ])
    mock = MockLLMClient(decisions, model_id="bad-fixture")
    with pytest.raises(PhaseBError) as exc:
        run_phase_b(view=view, donor_id="maya", target_id="maya",
                    registry=registry, schema=schema, llm_client=mock)
    assert exc.value.violations


# ---------------------------------------------------------------------------
# Phase C
# ---------------------------------------------------------------------------


def test_phase_c_pass_through_when_donor_equals_target(maya_fbx_ascii: Path, registry):
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    plan = EditPlan()  # no-op
    result = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="maya",
        registry=registry,
        edit_plan=plan,
    )
    assert result.merged_ascii == raw
    assert any("pass-through" in n for n in result.notes)


def test_phase_c_applies_renames_and_drops(maya_fbx_ascii: Path, registry):
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    hips = next(b for b in view.limb_bones() if b.name == "Hips")
    plan = EditPlan(drops=[], renames={hips.model_id: "ZZZ_RENAMED"})

    result = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="maya",
        registry=registry,
        edit_plan=plan,
    )
    assert b'"Model::ZZZ_RENAMED"' in result.merged_ascii
    assert b'"Model::Hips"' not in result.merged_ascii


# --- EditPlan.from_decisions reparent wiring (v2) --------------------------


def _maya_bone_id_by_name(view, name: str) -> int:
    return next(b.model_id for b in view.limb_bones() if b.name == name)


def test_edit_plan_carries_reparents_from_new_parent_role(maya_fbx_ascii: Path, registry):
    """A Decision with new_parent_role resolves to (bone_id -> target_bone_id) in
    EditPlan.reparents. The target id is the kept bone holding that role."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    chest_id = _maya_bone_id_by_name(view, "Chest")
    spine_id = _maya_bone_id_by_name(view, "Spine")

    decisions = DecisionSet(bones=[
        Decision(model_id=spine_id, role="Spine", verdict="keep"),
        Decision(model_id=chest_id, role="Chest", verdict="keep"),
        Decision(model_id=_maya_bone_id_by_name(view, "Left shoulder"),
                 role="Shoulder.L", verdict="keep",
                 new_parent_role="Chest"),
    ])
    plan = EditPlan.from_decisions(decisions, view, registry.get("maya"))
    shoulder_id = _maya_bone_id_by_name(view, "Left shoulder")
    assert shoulder_id in plan.reparents
    assert plan.reparents[shoulder_id] == chest_id


def test_edit_plan_no_reparents_when_decisions_omit_field(maya_fbx_ascii: Path, registry):
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    decisions = DecisionSet(bones=[
        Decision(model_id=_maya_bone_id_by_name(view, "Hips"),
                 role="Hips", verdict="keep"),
    ])
    plan = EditPlan.from_decisions(decisions, view, registry.get("maya"))
    assert plan.reparents == {}


def test_phase_c_passthrough_applies_reparent(maya_fbx_ascii: Path, registry):
    """Pass-through path emits reparent edits — the bone's parent_id in the
    Connections section changes."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    shoulder_id = _maya_bone_id_by_name(view, "Left shoulder")
    spine_id = _maya_bone_id_by_name(view, "Spine")

    # Reparent Left shoulder onto Spine (synthetic — just to verify wiring).
    plan = EditPlan(reparents={shoulder_id: spine_id})
    result = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="maya",
        registry=registry,
        edit_plan=plan,
    )
    out_view = extract(parse(result.merged_ascii))
    assert out_view.bones[shoulder_id].parent_id == spine_id


def test_phase_c_cross_avatar_requires_decisions():
    """Cross-avatar merge needs DecisionSet for canonical role lookup."""
    from rigforge.ascii_fbx.lexer import parse as _parse
    from rigforge.ascii_fbx.sections import extract as _extract
    reg = AvatarRegistry.load_default()
    raw = b"Objects: {\n}\nConnections: {\n}\n"
    view = _extract(_parse(raw))
    plan = EditPlan()
    with pytest.raises(PhaseCError, match="DecisionSet"):
        run_phase_c(
            clothing_ascii=raw,
            clothing_view=view,
            donor_id="maya",
            target_id="moe",
            registry=reg,
            edit_plan=plan,
            decisions=None,
        )


def test_phase_c_cross_avatar_produces_structurally_valid_ascii(maya_fbx_ascii: Path, registry):
    """End-to-end cross-merge on real avatars: treat Maya as clothing, target Moe.
    Verify the output ASCII re-parses cleanly + clothing bone Models are gone
    while clothing mesh content is preserved."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))

    # Build a decision set: every Maya bone whose display name matches Moe's
    # canonical_to_name gets verdict=keep with that role. Everything else is
    # secondary keep so the validator doesn't complain.
    moe = registry.get("moe")
    display_to_role = {v: k for k, v in moe.canonical_to_name.items()}
    bones = []
    for b in view.limb_bones():
        role = display_to_role.get(b.name)
        if role is not None:
            bones.append(Decision(model_id=b.model_id, role=role, verdict="keep"))
        else:
            bones.append(Decision(model_id=b.model_id,
                                   role=f"Secondary.{b.name}", verdict="keep"))
    decisions = DecisionSet(bones=bones, llm_model_id="test-cross-merge")
    plan = EditPlan()  # no renames/drops — clothing bones get stripped by Phase C

    result = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="moe",
        registry=registry,
        edit_plan=plan,
        decisions=decisions,
    )
    merged = result.merged_ascii

    # 1) The merged ASCII must re-parse cleanly (structural sanity gate)
    merged_doc = parse(merged)
    merged_view = extract(merged_doc)

    # 2) Moe's bones must still be present (target armature intact)
    moe_names = {b.name for b in merged_view.limb_bones()}
    for canon, moe_bone in moe.canonical_to_name.items():
        assert moe_bone in moe_names, f"target bone {moe_bone!r} ({canon}) missing"

    # 3) Maya's clothing-side bone Models must be gone
    #    (they were stripped — target has its own canonical bones)
    maya_only_names = {"Left arm", "Right arm"}  # exist in both convention-wise
    # Specifically Maya's "Twin tail_L.001" doesn't exist in Moe — it's secondary
    # so it should have survived as a kept "Secondary." role. Just verify the
    # canonical bone Models were dropped:
    canonical_clothing_bones = ["Left arm", "Right arm", "Left leg", "Right leg"]
    for cb in canonical_clothing_bones:
        # These appear in BOTH Maya and Moe; after merge, only ONE copy should
        # exist (Moe's). We verify by counting.
        count = merged.count(f'"Model::{cb}"'.encode("ascii"))
        assert count == 1, f"expected 1 Model::{cb} after merge, got {count}"


def test_phase_c_cross_avatar_applies_target_drop_bone_ids(
    maya_fbx_ascii: Path, registry,
):
    """Cross-merge must strip user-selected TARGET bones from the target's
    ASCII before splicing clothing in — that's how the modder removes the
    target's bundled clothing (Maya ships with a default outfit on Moe).

    Setup: treat Maya as donor, target Moe, drop one Moe-specific bone by id
    and verify it's gone from the merged ASCII while neighboring bones remain.
    """
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))

    moe = registry.get("moe")
    moe_view = moe.load_ascii_view()
    # Pick a real Moe secondary bone (not in canonical_to_name so dropping it
    # doesn't break the splice). Find any LimbNode whose name doesn't appear
    # in canonical_to_name values.
    canonical_targets = set(moe.canonical_to_name.values())
    droppable = next(
        b for b in moe_view.bones.values()
        if b.type_class == "LimbNode" and b.name and b.name not in canonical_targets
    )
    dropped_name = droppable.name

    display_to_role = {v: k for k, v in moe.canonical_to_name.items()}
    bones = []
    for b in view.limb_bones():
        role = display_to_role.get(b.name)
        if role is not None:
            bones.append(Decision(model_id=b.model_id, role=role, verdict="keep"))
        else:
            bones.append(Decision(model_id=b.model_id,
                                   role=f"Secondary.{b.name}", verdict="keep"))
    decisions = DecisionSet(bones=bones, llm_model_id="test-target-drop")
    plan = EditPlan()

    # Baseline: without target_drop, the bone should be present
    baseline = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="moe",
        registry=registry,
        edit_plan=plan,
        decisions=decisions,
    )
    assert f'"Model::{dropped_name}"'.encode("ascii") in baseline.merged_ascii, (
        f"baseline merge should still contain Model::{dropped_name}"
    )

    # With target_drop_bone_ids: bone should be gone
    result = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="moe",
        registry=registry,
        edit_plan=plan,
        decisions=decisions,
        target_drop_bone_ids={droppable.model_id},
    )
    assert f'"Model::{dropped_name}"'.encode("ascii") not in result.merged_ascii, (
        f"target_drop_bone_ids should have removed Model::{dropped_name}"
    )
    # And the merged ASCII still re-parses cleanly
    extract(parse(result.merged_ascii))


def test_phase_c_cross_avatar_applies_target_drop_mesh_ids(
    maya_fbx_ascii: Path, registry,
):
    """Cross-merge must drop target meshes (Mesh-Model + Geometry + Skin +
    Clusters) when target_drop_mesh_ids is set — this is how the modder
    strips Maya's bundled outfit meshes (Cloth, Shoes, Hat, etc.)."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))

    moe = registry.get("moe")
    moe_view = moe.load_ascii_view()
    # Pick any Mesh in the target. Whatever the name is, dropping it should
    # remove its Model + Geometry from the merged output.
    mesh = next(b for b in moe_view.bones.values() if b.type_class == "Mesh")
    mesh_name = mesh.name

    display_to_role = {v: k for k, v in moe.canonical_to_name.items()}
    bones = []
    for b in view.limb_bones():
        role = display_to_role.get(b.name)
        if role is not None:
            bones.append(Decision(model_id=b.model_id, role=role, verdict="keep"))
        else:
            bones.append(Decision(model_id=b.model_id,
                                   role=f"Secondary.{b.name}", verdict="keep"))
    decisions = DecisionSet(bones=bones, llm_model_id="test-target-mesh-drop")

    result = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="moe",
        registry=registry,
        edit_plan=EditPlan(),
        decisions=decisions,
        target_drop_mesh_ids={mesh.model_id},
    )
    # The target Mesh-Model must be gone
    assert f'"Model::{mesh_name}"'.encode("ascii") not in result.merged_ascii, (
        f"target_drop_mesh_ids should have removed Model::{mesh_name}"
    )
    # Output re-parses cleanly
    extract(parse(result.merged_ascii))


def test_phase_c_passthrough_drops_clothing_meshes(
    maya_fbx_ascii: Path, registry,
):
    """In pass-through mode, drop_mesh_ids applies to the clothing itself.
    The FE sends mesh ids of unwanted clothing meshes (e.g., the modder
    doesn't want the cape that ships with the outfit)."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    mesh = next(b for b in view.bones.values() if b.type_class == "Mesh")
    plan = EditPlan()
    result = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="maya",
        registry=registry,
        edit_plan=plan,
        drop_mesh_ids={mesh.model_id},
    )
    assert f'"Model::{mesh.name}"'.encode("ascii") not in result.merged_ascii
    extract(parse(result.merged_ascii))


def test_phase_c_passthrough_ignores_target_drop_bone_ids(
    maya_fbx_ascii: Path, registry,
):
    """In pass-through (donor==target), target_drop is meaningless — the
    output IS the clothing FBX. Should silently no-op (or note) rather than
    crash, so the FE doesn't have to special-case the pass-through path."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    plan = EditPlan()
    result = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="maya",
        registry=registry,
        edit_plan=plan,
        target_drop_bone_ids={1, 2, 3},
    )
    # Output must re-parse cleanly
    extract(parse(result.merged_ascii))
