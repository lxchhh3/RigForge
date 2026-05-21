"""Tests for the 7 validation rules.

Each rule has positive (clean) and negative (violation) fixtures. The
SectionView is constructed as a hand-built dict rather than parsed FBX so we
can isolate the rule under test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from rigforge.canonical.decisions import Decision, DecisionSet
from rigforge.canonical.schema import CanonicalSchema
from rigforge.canonical.validators import (
    errors,
    rule_drop_safety,
    rule_finger_count_sanity,
    rule_hierarchy_consistency,
    rule_l_r_pairing,
    rule_monotonic_spine_y,
    rule_required_roles_present,
    rule_unique_role,
    validate_all,
    warnings,
)


@pytest.fixture(scope="module")
def schema() -> CanonicalSchema:
    return CanonicalSchema.load_default()


# Build a minimal SectionView-like object. The validators only read:
#   view.bones[model_id].translation_xyz
#   view.bones[model_id].cluster_weight_count
#   view.bones[model_id].parent_id
#   view.bones[model_id].children_ids   (added in v1.1 for drop_safety subtree walk)
#   view.bones[model_id].name
# so a stub with these attrs is enough.

@dataclass
class StubBone:
    model_id: int
    name: str
    parent_id: Optional[int]
    translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cluster_weight_count: int = 100
    children_ids: list[int] = field(default_factory=list)


@dataclass
class StubView:
    bones: dict[int, StubBone]


def _ds(*decisions: Decision) -> DecisionSet:
    return DecisionSet(bones=list(decisions))


def _kept(model_id: int, role: str) -> Decision:
    return Decision(model_id=model_id, role=role, verdict="keep")


def _dropped(model_id: int, role: str = "aux") -> Decision:
    return Decision(model_id=model_id, role=role, verdict="drop", drop_category="aux")


# --- rule_unique_role -------------------------------------------------------


def test_unique_role_passes_when_unique(schema):
    decisions = _ds(_kept(1, "Hips"), _kept(2, "Spine"))
    view = StubView(bones={
        1: StubBone(1, "h", None),
        2: StubBone(2, "s", 1),
    })
    assert rule_unique_role(decisions, view, schema) == []


def test_unique_role_flags_duplicate(schema):
    decisions = _ds(_kept(1, "Hips"), _kept(2, "Hips"))
    view = StubView(bones={1: StubBone(1, "a", None), 2: StubBone(2, "b", None)})
    violations = rule_unique_role(decisions, view, schema)
    assert len(violations) == 1
    assert violations[0].rule == "unique_role"
    assert violations[0].severity == "error"


def test_unique_role_unknown_token_errors(schema):
    decisions = _ds(_kept(1, "Hips"), _kept(2, "unknown"))
    view = StubView(bones={1: StubBone(1, "a", None), 2: StubBone(2, "b", None)})
    violations = rule_unique_role(decisions, view, schema)
    assert any(v.rule == "unique_role" and "unknown" in v.message for v in violations)


def test_unique_role_allows_secondary_repeats(schema):
    """Secondary roles like Breast.L.01, Breast.L.02 can both be kept."""
    decisions = _ds(_kept(1, "Breast.L.01"), _kept(2, "Breast.L.02"))
    view = StubView(bones={1: StubBone(1, "a", None), 2: StubBone(2, "b", None)})
    assert rule_unique_role(decisions, view, schema) == []


# --- rule_required_roles_present --------------------------------------------


def test_required_roles_pass(schema):
    decisions = _ds(*[_kept(i, r) for i, r in enumerate(schema.required_roles)])
    view = StubView(bones={i: StubBone(i, r, None) for i, r in enumerate(schema.required_roles)})
    assert rule_required_roles_present(decisions, view, schema) == []


def test_required_roles_missing_hips(schema):
    """Drop Hips from the assignment → must flag missing required role."""
    needed = [r for r in schema.required_roles if r != "Hips"]
    decisions = _ds(*[_kept(i, r) for i, r in enumerate(needed)])
    view = StubView(bones={})
    violations = rule_required_roles_present(decisions, view, schema)
    assert any("Hips" in v.message for v in violations)


# --- rule_l_r_pairing -------------------------------------------------------


def test_l_r_pairing_passes_on_mirrored_translations(schema):
    decisions = _ds(_kept(1, "UpperArm.L"), _kept(2, "UpperArm.R"))
    view = StubView(bones={
        1: StubBone(1, "ul", None, translation_xyz=(0.12, 0.0, 0.0)),
        2: StubBone(2, "ur", None, translation_xyz=(-0.12, 0.0, 0.0)),
    })
    assert rule_l_r_pairing(decisions, view, schema) == []


def test_l_r_pairing_flags_missing_peer(schema):
    decisions = _ds(_kept(1, "UpperArm.L"))
    view = StubView(bones={1: StubBone(1, "ul", None, translation_xyz=(0.12, 0.0, 0.0))})
    violations = rule_l_r_pairing(decisions, view, schema)
    assert any("UpperArm.R" in v.message for v in violations)


def test_l_r_pairing_flags_bad_mirror(schema):
    """L and R kept but X is not mirrored."""
    decisions = _ds(_kept(1, "UpperArm.L"), _kept(2, "UpperArm.R"))
    view = StubView(bones={
        1: StubBone(1, "ul", None, translation_xyz=(0.12, 0.0, 0.0)),
        2: StubBone(2, "ur", None, translation_xyz=(0.12, 0.0, 0.0)),   # same X — wrong
    })
    violations = rule_l_r_pairing(decisions, view, schema)
    assert any("mirror" in v.message for v in violations)


# --- rule_monotonic_spine_y -------------------------------------------------


def test_spine_y_monotonic(schema):
    """Each spine bone's Lcl Y > 0 (offset upward from parent)."""
    decisions = _ds(
        _kept(1, "Hips"), _kept(2, "Spine"), _kept(3, "Chest"),
        _kept(4, "Neck"), _kept(5, "Head"),
    )
    view = StubView(bones={
        1: StubBone(1, "h", None,  translation_xyz=(0.0, 0.9, 0.0)),
        2: StubBone(2, "s", 1,     translation_xyz=(0.0, 0.1, 0.0)),
        3: StubBone(3, "c", 2,     translation_xyz=(0.0, 0.15, 0.0)),
        4: StubBone(4, "n", 3,     translation_xyz=(0.0, 0.2, 0.0)),
        5: StubBone(5, "hd", 4,    translation_xyz=(0.0, 0.1, 0.0)),
    })
    assert rule_monotonic_spine_y(decisions, view, schema) == []


def test_spine_y_negative_child_flagged(schema):
    decisions = _ds(_kept(1, "Hips"), _kept(2, "Spine"))
    view = StubView(bones={
        1: StubBone(1, "h", None,  translation_xyz=(0.0, 0.9, 0.0)),
        2: StubBone(2, "s", 1,     translation_xyz=(0.0, -0.5, 0.0)),  # bad
    })
    violations = rule_monotonic_spine_y(decisions, view, schema)
    assert any("Spine" in v.message for v in violations)


# --- rule_hierarchy_consistency ---------------------------------------------


def test_hierarchy_consistency_pass(schema):
    """Hand.L → LowerArm.L → UpperArm.L → Shoulder.L → Chest → Spine → Hips."""
    decisions = _ds(
        _kept(1, "Hips"), _kept(2, "Spine"), _kept(3, "Chest"),
        _kept(10, "Shoulder.L"), _kept(11, "UpperArm.L"),
        _kept(12, "LowerArm.L"), _kept(13, "Hand.L"),
    )
    view = StubView(bones={
        1: StubBone(1, "h", None),
        2: StubBone(2, "s", 1),
        3: StubBone(3, "c", 2),
        10: StubBone(10, "shL", 3),
        11: StubBone(11, "uaL", 10),
        12: StubBone(12, "laL", 11),
        13: StubBone(13, "haL", 12),
    })
    assert rule_hierarchy_consistency(decisions, view, schema) == []


def test_hierarchy_consistency_fails_when_parent_misassigned(schema):
    """Hand.L parented under Spine instead of LowerArm.L → error."""
    decisions = _ds(
        _kept(1, "Hips"), _kept(2, "Spine"),
        _kept(13, "Hand.L"),
    )
    view = StubView(bones={
        1: StubBone(1, "h", None),
        2: StubBone(2, "s", 1),
        13: StubBone(13, "haL", 2),   # parented to Spine — wrong
    })
    violations = rule_hierarchy_consistency(decisions, view, schema)
    assert violations, "expected hierarchy_consistency violation"


# --- rule_drop_safety -------------------------------------------------------


def test_drop_safety_pass_for_clean_drop(schema):
    """Bone marked drop with 0 cluster weights anywhere in subtree — safe."""
    decisions = _ds(_dropped(99, role="aux"))
    view = StubView(bones={99: StubBone(99, "ik_target", None, cluster_weight_count=0)})
    assert rule_drop_safety(decisions, view, schema) == []


def test_drop_safety_flags_high_weight_drop(schema):
    """Bone marked drop but has hundreds of skin bindings on itself — likely misclassified."""
    decisions = _ds(_dropped(99, role="aux"))
    view = StubView(bones={99: StubBone(99, "real_bone", None, cluster_weight_count=500)})
    violations = rule_drop_safety(decisions, view, schema)
    assert violations
    assert "drop" in violations[0].message


def test_drop_safety_blocks_orphaning_root_drop(schema):
    """A weight=0 root bone whose descendants carry weights MUST NOT be dropped —
    dropping orphans the descendants. The Head Phones_Root / Chain_1.006 case."""
    decisions = _ds(_dropped(100, role="aux"))
    view = StubView(bones={
        100: StubBone(100, "Head Phones_Root", None, cluster_weight_count=0,
                      children_ids=[101, 102]),
        101: StubBone(101, "Head Phones_L", 100, cluster_weight_count=278),
        102: StubBone(102, "Code_L", 100, cluster_weight_count=42,
                      children_ids=[103]),
        103: StubBone(103, "Code_L.001", 102, cluster_weight_count=48),
    })
    violations = rule_drop_safety(decisions, view, schema)
    assert violations, "drop_safety should block dropping a root with weighted descendants"
    msg = violations[0].message
    assert "subtree" in msg
    assert "self=0" in msg


def test_drop_safety_allows_drop_of_pure_aux_subtree(schema):
    """A bone whose entire subtree is unweighted (true IK chain) is droppable."""
    decisions = _ds(_dropped(100, role="aux"))
    view = StubView(bones={
        100: StubBone(100, "IK_root", None, cluster_weight_count=0,
                      children_ids=[101]),
        101: StubBone(101, "IK_target", 100, cluster_weight_count=0),
    })
    assert rule_drop_safety(decisions, view, schema) == []


def test_drop_safety_subtree_threshold_allows_export_noise(schema):
    """A subtree with <= threshold accidental binds (export noise) is still droppable."""
    decisions = _ds(_dropped(100, role="aux"))
    view = StubView(bones={
        100: StubBone(100, "noisy_aux", None, cluster_weight_count=0,
                      children_ids=[101]),
        101: StubBone(101, "stray_bind", 100, cluster_weight_count=3),
    })
    # 3 ≤ threshold (10) → allowed
    assert rule_drop_safety(decisions, view, schema) == []


# --- rule_drop_safety honors user_dropped category --------------------------


def test_drop_safety_skips_user_dropped_even_with_weighted_subtree(schema):
    """A drop_category='user_dropped' decision represents an explicit user
    choice — drop_safety must trust it and not error on weighted descendants."""
    decisions = _ds(
        Decision(model_id=1, role="aux", verdict="drop",
                 drop_category="user_dropped"),
    )
    view = StubView(bones={
        1: StubBone(1, "root", None,
                    cluster_weight_count=500, children_ids=[2]),
        2: StubBone(2, "child", 1, cluster_weight_count=300),
    })
    # Without the user_dropped exemption this would fire (subtree=800 > 10).
    assert rule_drop_safety(decisions, view, schema) == []


def test_drop_safety_still_fires_on_llm_drop(schema):
    """Regression guard: LLM-issued drops (non-user) still get gated."""
    decisions = _ds(
        Decision(model_id=1, role="aux", verdict="drop", drop_category="aux"),
    )
    view = StubView(bones={
        1: StubBone(1, "root", None,
                    cluster_weight_count=500, children_ids=[]),
    })
    violations = rule_drop_safety(decisions, view, schema)
    assert violations  # the 500-weight drop is still unsafe


# --- rule_finger_count_sanity ----------------------------------------------


def test_finger_count_sanity_passes_when_balanced(schema):
    """Both hands carry the same finger phalanges — no warning."""
    decisions = _ds(
        _kept(50, "Index1.L"), _kept(51, "Index2.L"),
        _kept(60, "Index1.R"), _kept(61, "Index2.R"),
    )
    view = StubView(bones={
        50: StubBone(50, "i1L", None), 51: StubBone(51, "i2L", 50),
        60: StubBone(60, "i1R", None), 61: StubBone(61, "i2R", 60),
    })
    assert rule_finger_count_sanity(decisions, view, schema) == []


def test_finger_count_sanity_warns_on_lateral_mismatch(schema):
    """`.L` finger kept but its `.R` peer missing → warning (not error)."""
    decisions = _ds(_kept(50, "Index1.L"))
    view = StubView(bones={50: StubBone(50, "i1L", None)})
    violations = rule_finger_count_sanity(decisions, view, schema)
    assert violations
    assert all(v.severity == "warning" for v in violations)


def test_finger_count_sanity_ignores_non_canonical(schema):
    """A role name that isn't canonical (typo, junk) shouldn't trigger this rule —
    `unique_role` already flags those paths."""
    decisions = _ds(_kept(50, "IndexFinger.L"))  # not in v2 schema
    view = StubView(bones={50: StubBone(50, "if", None)})
    assert rule_finger_count_sanity(decisions, view, schema) == []


# --- driver -----------------------------------------------------------------


def test_validate_all_clean_run_no_errors(schema):
    """Build a fully consistent decision set; validate_all should find zero errors."""
    decisions = _ds(
        _kept(1, "Hips"), _kept(2, "Spine"), _kept(3, "Chest"),
        _kept(4, "Head"),
        _kept(10, "UpperArm.L"), _kept(11, "UpperArm.R"),
        _kept(20, "UpperLeg.L"), _kept(21, "UpperLeg.R"),
    )
    view = StubView(bones={
        1:  StubBone(1, "h", None,  translation_xyz=(0.0, 0.9, 0.0)),
        2:  StubBone(2, "s", 1,     translation_xyz=(0.0, 0.1, 0.0)),
        3:  StubBone(3, "c", 2,     translation_xyz=(0.0, 0.15, 0.0)),
        4:  StubBone(4, "hd", 3,    translation_xyz=(0.0, 0.5, 0.0)),
        10: StubBone(10, "uaL", 3,  translation_xyz=( 0.15, 0.0, 0.0)),
        11: StubBone(11, "uaR", 3,  translation_xyz=(-0.15, 0.0, 0.0)),
        20: StubBone(20, "ulL", 1,  translation_xyz=( 0.10, 0.0, 0.0)),
        21: StubBone(21, "ulR", 1,  translation_xyz=(-0.10, 0.0, 0.0)),
    })
    # We're skipping Neck (Head's parent in schema) so hierarchy_consistency
    # will balk. That's a real (correct) violation — but let's also confirm
    # the path with Neck included works.
    decisions = _ds(
        _kept(1, "Hips"), _kept(2, "Spine"), _kept(3, "Chest"),
        _kept(5, "Neck"), _kept(4, "Head"),
        _kept(10, "UpperArm.L"), _kept(11, "UpperArm.R"),
        _kept(20, "UpperLeg.L"), _kept(21, "UpperLeg.R"),
    )
    view.bones[5] = StubBone(5, "n", 3, translation_xyz=(0.0, 0.2, 0.0))
    view.bones[4] = StubBone(4, "hd", 5, translation_xyz=(0.0, 0.1, 0.0))
    violations = validate_all(decisions, view, schema)
    assert errors(violations) == [], f"unexpected errors: {[v.message for v in errors(violations)]}"


def test_validate_all_returns_warnings_and_errors_split(schema):
    # Unbalanced finger triggers a warning; "unknown" role triggers an error.
    decisions = _ds(_kept(50, "Index1.L"), _kept(1, "unknown"))
    view = StubView(bones={50: StubBone(50, "i1L", None), 1: StubBone(1, "?", None)})
    violations = validate_all(decisions, view, schema)
    assert any(v.severity == "warning" for v in violations)
    assert any(v.severity == "error" for v in violations)
    assert len(warnings(violations)) >= 1
    assert len(errors(violations)) >= 1
