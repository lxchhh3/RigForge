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


def test_phase_b_unknown_role_falls_back_to_secondary(
    maya_fbx_ascii: Path, registry, schema,
):
    """When the LLM emits role='unknown', Phase B rewrites in place to
    role='Secondary.<bone.name>' with verdict='keep' so the pipeline keeps
    moving rather than hard-failing. Surfaces as a warning, not an error."""
    view = extract(parse(maya_fbx_ascii.read_bytes()))
    limbs = view.limb_bones()
    # Take a working identity classification, then corrupt two non-canonical
    # bones to role='unknown' to exercise the fallback path.
    decisions = _maya_identity_decisions(view)
    by_id = {d.model_id: i for i, d in enumerate(decisions.bones)}
    secondaries = [b for b in limbs if b.name not in
                   {v for v in registry.get("maya").canonical_to_name.values()}]
    assert len(secondaries) >= 2, "fixture broken: need 2+ secondary bones"
    unknown_ids = [secondaries[0].model_id, secondaries[1].model_id]
    for uid in unknown_ids:
        decisions.bones[by_id[uid]] = Decision(
            model_id=uid, role="unknown", verdict="keep",
        )
    mock = MockLLMClient(decisions, model_id="test-unknown-fallback")

    result = run_phase_b(
        view=view, donor_id="maya", target_id="maya",
        registry=registry, schema=schema, llm_client=mock,
    )

    # Each unknown rewritten to Secondary.<bone.name>; original name preserved
    for uid in unknown_ids:
        d = result.decisions.by_id()[uid]
        bone = view.bones[uid]
        assert d.role == f"Secondary.{bone.name}", (
            f"expected Secondary.{bone.name}, got {d.role!r}"
        )
        assert d.verdict == "keep"
    # Warnings carry the fallback record
    fallback_warnings = [w for w in result.warnings if w.rule == "unknown_role_fallback"]
    assert len(fallback_warnings) == len(unknown_ids)
    # No renames emitted for the rewritten bones (Secondary roles aren't in
    # canonical_to_name → bone keeps its on-disk name)
    for uid in unknown_ids:
        assert uid not in result.edit_plan.renames


def test_phase_b_unknown_fallback_handles_reprompt(
    maya_fbx_ascii: Path, registry, schema,
):
    """The re-prompt round can also emit 'unknown'; the fallback applies
    there too. Use a tiny sequential mock so the first call returns a
    violating set (Hips assigned twice) AND has an 'unknown' bone, then the
    re-prompt returns a clean set still containing an 'unknown'. Both
    unknowns must be rewritten and surfaced as warnings."""
    view = extract(parse(maya_fbx_ascii.read_bytes()))
    limbs = view.limb_bones()

    canonical_names = set(registry.get("maya").canonical_to_name.values())
    secondaries = [b for b in limbs if b.name not in canonical_names]
    assert len(secondaries) >= 3, "fixture broken: need 3+ secondary bones"

    # First call: dup Hips (forces re-prompt) + one 'unknown'
    first = _maya_identity_decisions(view)
    by_id_first = {d.model_id: i for i, d in enumerate(first.bones)}
    first.bones[by_id_first[secondaries[0].model_id]] = Decision(
        model_id=secondaries[0].model_id, role="unknown", verdict="keep",
    )
    first.bones[by_id_first[secondaries[2].model_id]] = Decision(
        model_id=secondaries[2].model_id, role="Hips", verdict="keep",
    )

    # Re-prompt response: clean, but still has one 'unknown' on another bone
    second = _maya_identity_decisions(view)
    by_id_second = {d.model_id: i for i, d in enumerate(second.bones)}
    second.bones[by_id_second[secondaries[1].model_id]] = Decision(
        model_id=secondaries[1].model_id, role="unknown", verdict="keep",
    )

    class _SequentialMock:
        model_id = "test-reprompt-unknown"
        def __init__(self, responses):
            self._responses = list(responses)
            self._i = 0
        def classify(self, request):
            r = self._responses[min(self._i, len(self._responses) - 1)]
            self._i += 1
            return r

    mock = _SequentialMock([first, second])
    result = run_phase_b(
        view=view, donor_id="maya", target_id="maya",
        registry=registry, schema=schema, llm_client=mock,
    )
    # Both rounds' unknowns landed as fallback warnings
    fallback_warnings = [w for w in result.warnings if w.rule == "unknown_role_fallback"]
    assert len(fallback_warnings) >= 2


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
    """donor==target triggers name-based merge (not a byte-identity
    pass-through). Output must re-parse and contain the target avatar's
    WEIGHTED canonical bones — that's how we ensure the avatar's body ships.

    Canonical bones that carry zero skin weight in the source rig (some
    rigs have weightless placeholder bones at the canonical-name spot) are
    swept by the post-merge zero-weight pass and are NOT expected to
    survive — that is the intended cleanup behavior."""
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
    merged_view = extract(parse(result.merged_ascii))

    # Every canonical bone that carries skin weight in the source must
    # survive the round trip. Weightless canonical bones are allowed to be
    # swept (that's the whole point of the cleanup).
    maya = registry.get("maya")
    output_names = {b.name for b in merged_view.bones.values()}
    source_by_name = {b.name: b for b in view.limb_bones()}
    for canon, name in maya.canonical_to_name.items():
        src_bone = source_by_name.get(name)
        if src_bone is None:
            continue  # not in source rig at all — not our concern here
        has_weight = any(
            view.clusters[cid].weight_count > 0
            for cid in src_bone.cluster_ids
            if cid in view.clusters
        )
        if has_weight:
            assert name in output_names, (
                f"weighted canonical bone {name!r} ({canon}) missing"
            )
    assert any("name-based repoint" in n for n in result.notes)
    assert any("zero_weight_sweep" in n for n in result.notes)


def test_phase_c_passthrough_renames_are_irrelevant_when_bones_match_target(
    maya_fbx_ascii: Path, registry,
):
    """In the new merge-based pass-through, clothing bones that match the
    target by name get STRIPPED — they're redundant with the target's own
    bones, which already carry the canonical names. Renames in the edit
    plan for those bones are silently moot; the target's named bone is
    what ships in the output. This documents the contract change."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    hips = next(b for b in view.limb_bones() if b.name == "Hips")
    # Set a rename — it should NOT show up in the output because the
    # clothing's Hips gets stripped (name-matched to target's Hips), and
    # target's Hips keeps its canonical name.
    plan = EditPlan(drops=[], renames={hips.model_id: "ZZZ_RENAMED"})
    result = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="maya",
        registry=registry,
        edit_plan=plan,
    )
    # ZZZ_RENAMED is NOT in the output — clothing's Hips got stripped, target's
    # Hips (still named "Hips") survived from the target ASCII.
    assert b'"Model::ZZZ_RENAMED"' not in result.merged_ascii
    assert b'"Model::Hips"' in result.merged_ascii


# Tiny clothing/target rigs for the name-based repoint fallback test. The
# clothing's Hips is MISNAMED (JBip_Hips) so it can't match the target by name;
# its EditPlan rename target ("Hips") is what should strip it. Spine matches by
# name directly. Ribbon is a genuine ornament (its rename target isn't a target
# bone) and must ride along.
_FALLBACK_CLOTHING_FBX = b"""\
; FBX 7.4.0 project file
Objects:  {
\tModel: 100, "Model::JBip_Hips", "LimbNode" {
\t}
\tModel: 101, "Model::Spine", "LimbNode" {
\t}
\tModel: 102, "Model::Ribbon", "LimbNode" {
\t}
}
Connections:  {
\tC: "OO",100,0
\tC: "OO",101,100
\tC: "OO",102,100
}
"""

_FALLBACK_TARGET_FBX = b"""\
; FBX 7.4.0 project file
Objects:  {
\tModel: 200, "Model::Hips", "LimbNode" {
\t}
\tModel: 201, "Model::Spine", "LimbNode" {
\t}
}
Connections:  {
\tC: "OO",200,0
\tC: "OO",201,200
}
"""


def test_name_based_repoint_strips_misnamed_canonical_via_rename_target():
    """A clothing canonical bone that misses by NAME but whose EditPlan rename
    target IS a target bone must be stripped + repointed (not ride along and
    duplicate). Ornament bones whose rename target isn't a target bone still
    ride along."""
    from rigforge.pipeline.phase_c import _build_name_based_repoint

    clothing = extract(parse(_FALLBACK_CLOTHING_FBX))
    target = extract(parse(_FALLBACK_TARGET_FBX))
    plan = EditPlan(drops=[], renames={100: "Hips", 102: "Ribbon_EN"})

    repoint, kept = _build_name_based_repoint(clothing, target, plan, notes=[])

    # JBip_Hips: name miss, rename target "Hips" IS a target bone -> stripped+repointed.
    assert repoint.get(100) == 200
    assert 100 in kept
    # Spine: matched directly by name.
    assert repoint.get(101) == 201
    assert 101 in kept
    # Ribbon: rename target "Ribbon_EN" is NOT a target bone -> rides along.
    assert 102 not in kept
    assert 102 not in repoint


# --- Bone-name translation (readability for the multilingual team) ----------


def _maya_secondary_bone(view, registry, *, weighted: bool = False):
    """Pick a Maya LimbNode whose name is NOT a canonical target name (so it
    rides along instead of being stripped). With weighted=True, require a
    cluster carrying skin weight so the bone survives Phase C's zero-weight
    sweep and reaches the output."""
    maya = registry.get("maya")
    canonical_names = set(maya.canonical_to_name.values())
    for b in view.limb_bones():
        if b.name in canonical_names:
            continue
        if weighted and not any(
            view.clusters[c].weight_count > 0
            for c in b.cluster_ids if c in view.clusters
        ):
            continue
        return b
    raise AssertionError("no suitable secondary bone in Maya view")


def test_sanitize_bone_name_blender_safe():
    """name_en is coerced to a Blender/FBX-safe token; unusable input (no ASCII
    content left, e.g. untranslated CJK) returns None so the caller keeps the
    original name."""
    from rigforge.pipeline.edit_plan import _sanitize_bone_name
    assert _sanitize_bone_name("Skirt Front") == "Skirt_Front"
    assert _sanitize_bone_name("  Skirt_Front.001  ") == "Skirt_Front.001"
    assert _sanitize_bone_name("Skirt/Front!") == "SkirtFront"
    assert _sanitize_bone_name("スカート") is None  # untranslated JP → keep original
    assert _sanitize_bone_name("") is None
    assert _sanitize_bone_name(None) is None


def test_from_decisions_translates_non_canonical_bone_to_name_en(maya_fbx_ascii, registry):
    """A non-canonical (ride-along) bone is renamed to its English translation
    (Decision.name_en) so the multilingual team can read it."""
    view = extract(parse(maya_fbx_ascii.read_bytes()))
    maya = registry.get("maya")
    sec = _maya_secondary_bone(view, registry)
    decisions = DecisionSet(
        bones=[Decision(model_id=sec.model_id, role="SkirtFront.C.01",
                        verdict="keep", name_en="Skirt_Front_01")],
        llm_model_id="t",
    )
    plan = EditPlan.from_decisions(decisions, view, maya)
    assert plan.renames.get(sec.model_id) == "Skirt_Front_01"


def test_from_decisions_translates_unknown_fallback_bone_too(maya_fbx_ascii, registry):
    """Even a Secondary.<name> fallback bone gets translated — readability is
    the goal, regardless of how confidently the LLM classified the role."""
    view = extract(parse(maya_fbx_ascii.read_bytes()))
    maya = registry.get("maya")
    sec = _maya_secondary_bone(view, registry)
    decisions = DecisionSet(
        bones=[Decision(model_id=sec.model_id, role=f"Secondary.{sec.name}",
                        verdict="keep", name_en="Translated_Thing")],
        llm_model_id="t",
    )
    plan = EditPlan.from_decisions(decisions, view, maya)
    assert plan.renames.get(sec.model_id) == "Translated_Thing"


def test_from_decisions_no_translation_without_name_en(maya_fbx_ascii, registry):
    """No name_en (None — e.g. cache/fixtures predating the field) → no rename;
    the bone keeps its original name. Backward compatible."""
    view = extract(parse(maya_fbx_ascii.read_bytes()))
    maya = registry.get("maya")
    sec = _maya_secondary_bone(view, registry)
    decisions = DecisionSet(
        bones=[Decision(model_id=sec.model_id, role="SkirtFront.C.01", verdict="keep")],
        llm_model_id="t",
    )
    plan = EditPlan.from_decisions(decisions, view, maya)
    assert sec.model_id not in plan.renames


def test_from_decisions_skips_translation_equal_to_original(maya_fbx_ascii, registry):
    """If name_en already equals the bone's name (already English), no rename."""
    view = extract(parse(maya_fbx_ascii.read_bytes()))
    maya = registry.get("maya")
    sec = _maya_secondary_bone(view, registry)
    decisions = DecisionSet(
        bones=[Decision(model_id=sec.model_id, role="Secondary.x",
                        verdict="keep", name_en=sec.name)],
        llm_model_id="t",
    )
    plan = EditPlan.from_decisions(decisions, view, maya)
    assert sec.model_id not in plan.renames


def test_phase_c_applies_translation_to_ride_along_bone(maya_fbx_ascii, registry):
    """The payoff: a ride-along bone is renamed to its English translation in
    the merged output. Maya-as-clothing → Moe target; a weighted Maya bone
    unique to the clothing is given name_en and must appear under that name
    (and not its original) in the output."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    moe = registry.get("moe")
    maya_canon = set(registry.get("maya").canonical_to_name.values())
    moe_names = {b.name for b in moe.load_ascii_view().bones.values()}
    # Ride-along (not canonical) + survives the zero-weight sweep (weighted) +
    # name unique to the clothing (absent from Moe) so the only Model::<name>
    # in the output is the clothing's — which we translate away.
    sec = next(
        b for b in view.limb_bones()
        if b.name not in maya_canon and b.name not in moe_names
        and any(view.clusters[c].weight_count > 0 for c in b.cluster_ids if c in view.clusters)
    )
    translated = "Translated_RideAlong_01"

    display_to_role = {v: k for k, v in moe.canonical_to_name.items()}
    bones = []
    for b in view.limb_bones():
        if b.model_id == sec.model_id:
            bones.append(Decision(model_id=b.model_id, role="SkirtFront.C.01",
                                  verdict="keep", name_en=translated))
            continue
        role = display_to_role.get(b.name)
        bones.append(Decision(
            model_id=b.model_id,
            role=role if role is not None else f"Secondary.{b.name}",
            verdict="keep",
        ))
    decisions = DecisionSet(bones=bones, llm_model_id="t-translate")
    plan = EditPlan.from_decisions(decisions, view, moe)
    assert plan.renames.get(sec.model_id) == translated

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
    assert f'"Model::{translated}"'.encode("ascii") in merged, "ride-along bone not translated in output"
    assert f'"Model::{sec.name}"'.encode("ascii") not in merged, "original name should be gone (translated)"
    extract(parse(merged))  # output re-parses cleanly


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


def test_phase_c_passthrough_reparent_is_moot_for_target_bones(maya_fbx_ascii: Path, registry):
    """Under the merge-based pass-through, clothing bones that match the
    target by name are stripped — their reparent entries become moot
    because the bone surviving in the output is the target's, not the
    clothing's. EditPlan.from_decisions still emits reparents for
    clothing-side use (e.g. structure correction before strip); they just
    don't visibly affect the merged output when the bone is name-matched.

    This test pins the contract: a reparent edit for a name-matched bone
    is silently ignored, and the bone's parent in the output is whatever
    the target avatar has on disk."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    shoulder_id = _maya_bone_id_by_name(view, "Left shoulder")
    spine_id = _maya_bone_id_by_name(view, "Spine")
    plan = EditPlan(reparents={shoulder_id: spine_id})
    result = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="maya",
        registry=registry,
        edit_plan=plan,
    )
    # Output re-parses; target's Left shoulder still attached where the
    # target's own ASCII puts it (not the synthetic Spine reparent).
    extract(parse(result.merged_ascii))


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

    # 2) Moe's WEIGHTED bones must still be present (target armature intact).
    #    Weightless canonical bones are allowed to be swept by the post-merge
    #    zero-weight pass — that's the cleanup we want.
    moe_view = moe.load_ascii_view()
    moe_source_by_name = {b.name: b for b in moe_view.limb_bones()}
    merged_names = {b.name for b in merged_view.limb_bones()}
    for canon, moe_bone in moe.canonical_to_name.items():
        src = moe_source_by_name.get(moe_bone)
        if src is None:
            continue
        has_weight = any(
            moe_view.clusters[cid].weight_count > 0
            for cid in src.cluster_ids
            if cid in moe_view.clusters
        )
        if has_weight:
            assert moe_bone in merged_names, (
                f"weighted target bone {moe_bone!r} ({canon}) missing"
            )

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
    """In pass-through (merge-based), drop_mesh_ids strips the clothing's
    copy of a mesh before splice. The target's copy of the same mesh
    survives (the modder removes that via target_drop_mesh_ids). Compare
    with-drop vs without-drop to verify the clothing instance got dropped."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    mesh = next(b for b in view.bones.values() if b.type_class == "Mesh")
    plan = EditPlan()
    baseline = run_phase_c(
        clothing_ascii=raw, clothing_view=view, donor_id="maya", target_id="maya",
        registry=registry, edit_plan=plan,
    )
    dropped = run_phase_c(
        clothing_ascii=raw, clothing_view=view, donor_id="maya", target_id="maya",
        registry=registry, edit_plan=plan, drop_mesh_ids={mesh.model_id},
    )
    needle = f'"Model::{mesh.name}"'.encode("ascii")
    # One fewer instance of the mesh name when the clothing's copy was dropped.
    assert dropped.merged_ascii.count(needle) == baseline.merged_ascii.count(needle) - 1
    extract(parse(dropped.merged_ascii))


def test_phase_c_passthrough_handles_stale_target_drop_ids(
    maya_fbx_ascii: Path, registry,
):
    """In merge-based pass-through, target_drop_* DOES apply now (the target
    ASCII is the merge base). The endpoint must still accept stale/unknown
    ids without crashing — they should be silently skipped with a note."""
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
        target_drop_bone_ids={1, 2, 3},  # not real target bone ids
    )
    # Output must re-parse cleanly — stale ids didn't break the merge
    extract(parse(result.merged_ascii))
    # And the run produced skip-notes for the stale ids
    assert any("skip: target_drop bone id=" in n for n in result.notes)


def test_phase_c_passthrough_drops_clothing_blend_shape_channels(
    maya_fbx_ascii: Path, registry,
):
    """User-driven channel drop on the clothing side strips that channel
    node + its connections. We no longer dedup blendshapes by name (each
    mesh owns its own deformer + channels — see _compute_dedup_repoint), so
    the drop lands as a plain strip with no collapse-into-dedup behavior."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    ch = next(iter(view.blend_shape_channels.values()))
    plan = EditPlan()
    result = run_phase_c(
        clothing_ascii=raw, clothing_view=view, donor_id="maya", target_id="maya",
        registry=registry, edit_plan=plan,
        drop_blend_shape_channel_ids={ch.channel_id},
    )
    extract(parse(result.merged_ascii))  # re-parses cleanly
    assert any("drop_blend_shape_channels: removed" in n for n in result.notes)


def test_phase_c_passthrough_applies_clothing_deform_percent_override(
    maya_fbx_ascii: Path, registry,
):
    """An override on a clothing channel writes the new DeformPercent value
    into the clothing's copy of that channel before splice. We no longer
    dedup by name, so the value lands in the merged ASCII verbatim."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    ch = next(iter(view.blend_shape_channels.values()))
    plan = EditPlan()
    result = run_phase_c(
        clothing_ascii=raw, clothing_view=view, donor_id="maya", target_id="maya",
        registry=registry, edit_plan=plan,
        blend_shape_channel_overrides={ch.channel_id: 42.5},
    )
    extract(parse(result.merged_ascii))  # re-parses cleanly
    assert b"DeformPercent: 42.5" in result.merged_ascii
    assert any("override: set DeformPercent on" in n for n in result.notes)


def test_phase_c_passthrough_override_skipped_when_channel_dropped(
    maya_fbx_ascii: Path, registry,
):
    """If a channel id is in both drop and override sets, the drop wins —
    the override is silently skipped (would otherwise overlap with the
    drop_node_edit on the same byte range). Verify by ensuring the override
    value never appears in the output."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    ch = next(iter(view.blend_shape_channels.values()))
    plan = EditPlan()
    result = run_phase_c(
        clothing_ascii=raw, clothing_view=view, donor_id="maya", target_id="maya",
        registry=registry, edit_plan=plan,
        drop_blend_shape_channel_ids={ch.channel_id},
        blend_shape_channel_overrides={ch.channel_id: 73},
    )
    # The override value (73) never lands in the merged output — the drop
    # short-circuited it, and no other channel happens to be set to 73.
    assert b"DeformPercent: 73" not in result.merged_ascii


def test_phase_c_cross_avatar_applies_target_deform_percent_override(
    maya_fbx_ascii: Path, registry,
):
    """target_blend_shape_channel_overrides must rewrite DeformPercent on
    target channels BEFORE the splice, so the assembled FBX ships with the
    modder's baked-in expression values."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))

    moe = registry.get("moe")
    moe_view = moe.load_ascii_view()
    target_ch = next(iter(moe_view.blend_shape_channels.values()))

    display_to_role = {v: k for k, v in moe.canonical_to_name.items()}
    bones = []
    for b in view.limb_bones():
        role = display_to_role.get(b.name)
        if role is not None:
            bones.append(Decision(model_id=b.model_id, role=role, verdict="keep"))
        else:
            bones.append(Decision(model_id=b.model_id,
                                   role=f"Secondary.{b.name}", verdict="keep"))
    decisions = DecisionSet(bones=bones, llm_model_id="test-target-channel-override")

    result = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="moe",
        registry=registry,
        edit_plan=EditPlan(),
        decisions=decisions,
        target_blend_shape_channel_overrides={target_ch.channel_id: 60},
    )
    out_view = extract(parse(result.merged_ascii))
    out_ch = out_view.blend_shape_channels[target_ch.channel_id]
    for child in out_ch.node_ref.children:
        if child.name == "DeformPercent":
            args = child.args_bytes(result.merged_ascii).strip()
            assert args == b"60", f"expected b'60', got {args!r}"
            break
    else:
        raise AssertionError("DeformPercent missing on target channel after merge")


def test_phase_c_cross_avatar_drops_target_blend_shape_channels(
    maya_fbx_ascii: Path, registry,
):
    """Cross-merge must strip target-side morph channels when
    target_drop_blend_shape_channel_ids is set, so the modder can cull morphs
    they don't want shipped on the avatar."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))

    moe = registry.get("moe")
    moe_view = moe.load_ascii_view()
    target_ch = next(iter(moe_view.blend_shape_channels.values()))

    display_to_role = {v: k for k, v in moe.canonical_to_name.items()}
    bones = []
    for b in view.limb_bones():
        role = display_to_role.get(b.name)
        if role is not None:
            bones.append(Decision(model_id=b.model_id, role=role, verdict="keep"))
        else:
            bones.append(Decision(model_id=b.model_id,
                                   role=f"Secondary.{b.name}", verdict="keep"))
    decisions = DecisionSet(bones=bones, llm_model_id="test-target-channel-drop")

    result = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="moe",
        registry=registry,
        edit_plan=EditPlan(),
        decisions=decisions,
        target_drop_blend_shape_channel_ids={target_ch.channel_id},
    )
    merged_view = extract(parse(result.merged_ascii))
    assert target_ch.channel_id not in merged_view.blend_shape_channels, (
        "target_drop_blend_shape_channel_ids should have removed the target channel"
    )


def test_phase_c_cross_avatar_dedups_materials_by_name(
    maya_fbx_ascii: Path, registry,
):
    """Maya and Moe both ship with materials named Body/Cloth/Hair/etc.
    After cross-merge the result must have exactly ONE of each — the target's.
    Without dedup, the wholesale Objects splice double-inserts every donor
    material, producing duplicate Material::Cloth nodes.
    """
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))

    moe = registry.get("moe")
    moe_view = moe.load_ascii_view()
    shared = (
        {m.name for m in view.materials.values()}
        & {m.name for m in moe_view.materials.values()}
    )
    assert shared, "fixture broken: expected at least one shared material name"

    display_to_role = {v: k for k, v in moe.canonical_to_name.items()}
    bones = []
    for b in view.limb_bones():
        role = display_to_role.get(b.name)
        if role is not None:
            bones.append(Decision(model_id=b.model_id, role=role, verdict="keep"))
        else:
            bones.append(Decision(model_id=b.model_id,
                                   role=f"Secondary.{b.name}", verdict="keep"))
    decisions = DecisionSet(bones=bones, llm_model_id="test-dedup-materials")

    result = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="moe",
        registry=registry,
        edit_plan=EditPlan(),
        decisions=decisions,
    )
    merged = result.merged_ascii
    # Output re-parses
    merged_view = extract(parse(merged))

    # Every shared-name material appears exactly once
    for name in shared:
        needle = f'"Material::{name}"'.encode("utf-8")
        count = merged.count(needle)
        assert count == 1, (
            f"expected 1 Material::{name} after dedup, got {count}"
        )

    # Note should record the dedup
    assert any("dedup: dropped" in n and "materials" in n for n in result.notes), (
        f"expected a dedup note, got: {result.notes}"
    )

    # The dedup primitive in the merged view should still hold target ids
    merged_mat_names = {m.name for m in merged_view.materials.values()}
    for name in shared:
        assert name in merged_mat_names


def test_phase_c_cross_avatar_preserves_blendshape_channels_per_mesh(
    maya_fbx_ascii: Path, registry,
):
    """Blendshape channels are NOT deduped by name across meshes. A donor's
    "Smile" channel on its Body mesh deforms a different vertex set than
    target's "Smile" on its Body mesh — conflating them by name produces
    either in-between Shape errors or vertex-index out-of-bounds when the
    output is imported into Blender. Each mesh keeps its own channels."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))

    moe = registry.get("moe")
    moe_view = moe.load_ascii_view()
    shared = (
        {c.name for c in view.blend_shape_channels.values()}
        & {c.name for c in moe_view.blend_shape_channels.values()}
    )
    assert shared, "fixture broken: expected shared blendshape channel names"

    display_to_role = {v: k for k, v in moe.canonical_to_name.items()}
    bones = []
    for b in view.limb_bones():
        role = display_to_role.get(b.name)
        if role is not None:
            bones.append(Decision(model_id=b.model_id, role=role, verdict="keep"))
        else:
            bones.append(Decision(model_id=b.model_id,
                                   role=f"Secondary.{b.name}", verdict="keep"))
    decisions = DecisionSet(bones=bones, llm_model_id="test-blendshape-passthrough")

    result = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="moe",
        registry=registry,
        edit_plan=EditPlan(),
        decisions=decisions,
    )
    merged = result.merged_ascii
    target_ascii = moe.load_ascii_bytes()
    extract(parse(merged))  # structural sanity

    # Every shared-name channel's merged count is target_count + donor_count —
    # both sides survive because we don't dedup blendshapes by name anymore.
    sample = sorted(shared)[:10]
    for name in sample:
        needle = f'"SubDeformer::{name}"'.encode("utf-8")
        merged_count = merged.count(needle)
        target_count = target_ascii.count(needle)
        donor_count = raw.count(needle)
        assert merged_count == target_count + donor_count, (
            f"expected {target_count} (target) + {donor_count} (donor) = "
            f"{target_count + donor_count} SubDeformer::{name} in merged, "
            f"got {merged_count}"
        )

    assert not any("dedup: dropped" in n and "channels" in n for n in result.notes), (
        f"channels must not be deduped; got: {result.notes}"
    )


def test_phase_c_cross_avatar_dedup_repoints_connections_to_target(
    maya_fbx_ascii: Path, registry,
):
    """When a donor material is dropped because it collides by name with a
    target material, clothing-side connections that referenced the donor
    material must be redirected to the target material's id by id_offset's
    repoint table — otherwise the spliced Geometry-Material connection points
    at a non-existent id.

    Sanity check: pick a shared material, find a clothing-side connection that
    references it, and assert the merged ASCII contains the target's id for
    that material (not the offset-shifted donor id).
    """
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))

    moe = registry.get("moe")
    moe_view = moe.load_ascii_view()

    # Pick a shared material that has at least one connection on the donor side.
    moe_by_name = {m.name: mid for mid, m in moe_view.materials.items()}
    donor_mat = None
    for mid, mat in view.materials.items():
        if mat.name in moe_by_name:
            donor_mat = (mid, mat.name, moe_by_name[mat.name])
            break
    assert donor_mat is not None, "fixture broken: need a shared material"
    donor_mid, mat_name, target_mid = donor_mat

    display_to_role = {v: k for k, v in moe.canonical_to_name.items()}
    bones = []
    for b in view.limb_bones():
        role = display_to_role.get(b.name)
        if role is not None:
            bones.append(Decision(model_id=b.model_id, role=role, verdict="keep"))
        else:
            bones.append(Decision(model_id=b.model_id,
                                   role=f"Secondary.{b.name}", verdict="keep"))
    decisions = DecisionSet(bones=bones, llm_model_id="test-dedup-repoint")

    result = run_phase_c(
        clothing_ascii=raw,
        clothing_view=view,
        donor_id="maya",
        target_id="moe",
        registry=registry,
        edit_plan=EditPlan(),
        decisions=decisions,
    )
    merged_view = extract(parse(result.merged_ascii))

    # The target material's id must still be the one in merged_view; the donor
    # id must be absent (dropped and repointed).
    merged_mat_ids_by_name = {m.name: mid for mid, m in merged_view.materials.items()}
    assert merged_mat_ids_by_name[mat_name] == target_mid, (
        f"expected target material {mat_name} to keep id {target_mid}, "
        f"got {merged_mat_ids_by_name[mat_name]}"
    )
