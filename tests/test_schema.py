"""Tests for the canonical schema loader."""
from __future__ import annotations

import pytest

from rigforge.canonical.schema import CanonicalSchema


@pytest.fixture(scope="module")
def schema() -> CanonicalSchema:
    return CanonicalSchema.load_default()


def test_loads_default(schema: CanonicalSchema):
    assert schema.version == "2.2"
    assert "Hips" in schema.roles
    assert "Head" in schema.roles


def test_required_roles_are_canonical(schema: CanonicalSchema):
    for r in schema.required_roles:
        assert r in schema.roles, f"required role {r} not in roles dict"


def test_spine_chain_is_canonical(schema: CanonicalSchema):
    for r in schema.spine_chain:
        assert r in schema.roles


def test_canonical_hierarchy_resolves(schema: CanonicalSchema):
    """Every role's parent (if any) is itself canonical."""
    for name, rdef in schema.roles.items():
        if rdef.parent is not None:
            assert rdef.parent in schema.roles, f"{name} -> {rdef.parent} not canonical"


def test_fingers_are_canonical(schema: CanonicalSchema):
    """v2 promotes fingers to canonical roles. 5 fingers × 3 phalanges × 2 sides."""
    for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
        for phalanx in (1, 2, 3):
            for side in (".L", ".R"):
                role = f"{finger}{phalanx}{side}"
                assert role in schema.roles, f"v2 must include canonical {role}"
                assert schema.roles[role].category == "finger"
                assert schema.roles[role].optional is True


def test_twist_bones_are_canonical(schema: CanonicalSchema):
    """v2 promotes twist bones to canonical roles. Four limbs × two sides."""
    for stem in ("UpperArm.Twist", "LowerArm.Twist", "UpperLeg.Twist", "LowerLeg.Twist"):
        for side in (".L", ".R"):
            role = f"{stem}{side}"
            assert role in schema.roles, f"v2 must include canonical {role}"
            assert schema.roles[role].category == "twist"
            assert schema.roles[role].optional is True


def test_finger_phalanx_parent_chain(schema: CanonicalSchema):
    """Each phalanx parents into the previous one; phalanx 1 parents the hand."""
    assert schema.parent_of("Index1.L") == "Hand.L"
    assert schema.parent_of("Index2.L") == "Index1.L"
    assert schema.parent_of("Index3.L") == "Index2.L"
    assert schema.parent_of("Thumb1.R") == "Hand.R"


def test_twist_parents_limb(schema: CanonicalSchema):
    assert schema.parent_of("UpperArm.Twist.L") == "UpperArm.L"
    assert schema.parent_of("LowerLeg.Twist.R") == "LowerLeg.R"


def test_ancestor_chain_finger(schema: CanonicalSchema):
    chain = schema.ancestor_chain("Index3.L")
    assert chain == [
        "Index3.L", "Index2.L", "Index1.L", "Hand.L",
        "LowerArm.L", "UpperArm.L", "Shoulder.L",
        "Chest", "Spine", "Hips",
    ]


def test_lateral_pair_arms(schema: CanonicalSchema):
    assert schema.lateral_pair_of("UpperArm.L") == "UpperArm.R"
    assert schema.lateral_pair_of("UpperArm.R") == "UpperArm.L"
    assert schema.lateral_pair_of("Hips") is None  # no lateral peer


def test_ancestor_chain_arm(schema: CanonicalSchema):
    chain = schema.ancestor_chain("Hand.L")
    assert chain == ["Hand.L", "LowerArm.L", "UpperArm.L", "Shoulder.L", "Chest", "Spine", "Hips"]


def test_ancestor_chain_leg(schema: CanonicalSchema):
    chain = schema.ancestor_chain("Toes.R")
    assert chain == ["Toes.R", "Foot.R", "LowerLeg.R", "UpperLeg.R", "Hips"]


def test_is_canonical_vs_secondary(schema: CanonicalSchema):
    assert schema.is_canonical("Hips")
    assert not schema.is_canonical("Breast.L.01")
    assert schema.is_secondary("Breast.L.01")
    assert schema.is_secondary("HairSecondary.C.5")
    assert not schema.is_secondary("RandomGarbage")


def test_accessory_pattern_is_secondary(schema: CanonicalSchema):
    """v2.1: Accessory.* is the catch-all for kept structural parent bones
    that don't fit any canonical role (ribbon roots, jacket roots, decorative
    chain anchors). LLM puts Ribon_Root → Accessory.C.01 instead of dropping it."""
    assert schema.is_secondary("Accessory.C.01")
    assert schema.is_secondary("Accessory.L.01")
    assert schema.is_secondary("Accessory.R.02")
    assert schema.is_secondary("Accessory.C")  # unindexed first instance allowed
    assert not schema.is_canonical("Accessory.C.01")


def test_special_tokens(schema: CanonicalSchema):
    assert schema.is_special_token("unknown")
    assert schema.is_special_token("aux")
    assert not schema.is_special_token("Hips")


def test_verdict_values(schema: CanonicalSchema):
    assert set(schema.verdict_values) == {"keep", "drop"}


def test_secondary_pattern_unindexed_matches():
    """A pattern can match with or without an index suffix."""
    s = CanonicalSchema.load_default()
    assert s.is_secondary("Breast.L")
    assert s.is_secondary("Breast.L.01")
    assert s.is_secondary("Breast.L.5")
    assert not s.is_secondary("Breast.L.foo")  # non-numeric tail
