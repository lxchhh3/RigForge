"""Tests for the avatar registry + fingerprint + Phase A donor identification."""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.avatars.fingerprint import (
    fingerprint,
    jaccard,
    normalize_bone_name,
    rank_candidates,
    rank_clothing_candidates,
)
from rigforge.avatars.registry import (
    AvatarRegistry,
    CuratedAvatar,
    DonorIdentificationError,
    RegistryError,
    identify_donor,
)


# --- normalize -------------------------------------------------------------


def test_normalize_lowercases():
    assert normalize_bone_name("Hips") == "hips"


def test_normalize_strips_separators():
    assert normalize_bone_name("J_Bip_C_Hips") == "jbipchips"
    assert normalize_bone_name("Left arm") == "leftarm"
    assert normalize_bone_name("Eye.L.001") == "eyel001"


def test_normalize_strips_mixamo_prefix():
    assert normalize_bone_name("mixamorig:Hips") == "hips"
    assert normalize_bone_name("mixamorig:LeftArm") == "leftarm"


def test_fingerprint_returns_set():
    names = ["Hips", "Spine", "hips"]  # case-insensitive collapse
    fp = fingerprint(names)
    assert fp == {"hips", "spine"}


# --- jaccard --------------------------------------------------------------


def test_jaccard_identical_sets():
    assert jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0


def test_jaccard_disjoint_sets():
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_partial_overlap():
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


def test_jaccard_both_empty():
    assert jaccard(set(), set()) == 0.0


def test_rank_clothing_candidates_picks_avatar_with_more_canonical_overlap():
    """Containment + uniqueness scoring: clothing's bones should match
    avatar Moe (which carries Hair_L) over Maya (which doesn't), even when
    clothing has many accessory bones neither avatar shares."""
    clothing = {
        "hips", "spine", "chest", "head",      # canonical (in both)
        "leftarm", "rightarm",                   # canonical (in both)
        "hairl001", "hairl002",                  # Moe-only bones
        "ribbon1", "ribbon2", "chain1", "chain2",  # accessory bones in neither
    }
    avatars = {
        "moe":  {"hips", "spine", "chest", "head", "leftarm", "rightarm",
                 "hairl001", "hairl002", "hairl003", "tail001"},
        "maya": {"hips", "spine", "chest", "head", "leftarm", "rightarm",
                 "ahoge001", "twintaill001"},
    }
    ranked = rank_clothing_candidates(clothing, avatars)
    assert ranked[0][0] == "moe"


def test_rank_clothing_candidates_perfect_subset_scores_one():
    """When clothing is a strict subset of an avatar (all clothing bones
    known to that avatar), containment score is 1.0."""
    clothing = {"hips", "spine"}
    avatars = {"moe": {"hips", "spine", "chest", "head"}}
    ranked = rank_clothing_candidates(clothing, avatars)
    assert ranked[0] == ("moe", 1.0)


def test_rank_candidates_sorted_high_low():
    target = {"a", "b", "c"}
    cands = {
        "perfect": {"a", "b", "c"},
        "partial": {"a", "b", "x"},
        "none":    {"x", "y", "z"},
    }
    ranked = rank_candidates(target, cands)
    assert ranked[0][0] == "perfect"
    assert ranked[0][1] == 1.0
    assert ranked[-1][0] == "none"
    assert ranked[-1][1] == 0.0


# --- registry --------------------------------------------------------------


def test_registry_load_default():
    reg = AvatarRegistry.load_default()
    assert "maya" in reg.avatars
    av = reg.get("maya")
    assert av.bone_names
    assert av.fingerprint  # non-empty
    # Canonical mapping covers the v1 required roles
    assert "Hips" in av.canonical_to_name
    assert "UpperArm.L" in av.canonical_to_name


def test_registry_includes_moe():
    reg = AvatarRegistry.load_default()
    assert "moe" in reg.avatars
    av = reg.get("moe")
    assert len(av.bone_names) > 100
    # Moe shares Maya's descriptive convention for limbs
    assert av.canonical_to_name["UpperArm.L"] == "Left arm"
    assert av.canonical_to_name["UpperLeg.R"] == "Right leg"


def test_avatar_load_ascii_view_returns_sectionview():
    """Phase C needs each curated avatar's ASCII parsed into a SectionView so
    it can look up target-bone-id by canonical name. Cached after first call."""
    reg = AvatarRegistry.load_default()
    av = reg.get("maya")
    view1 = av.load_ascii_view()
    assert view1 is not None
    # Look up a known canonical bone name via the section view
    bones_by_name = {b.name: b for b in view1.limb_bones()}
    assert "Hips" in bones_by_name
    # Second call returns the same cached object
    view2 = av.load_ascii_view()
    assert view2 is view1


def test_avatar_target_bone_id_by_canonical():
    """Direct lookup: canonical role -> target bone model_id, used by
    Phase C cluster repoint."""
    reg = AvatarRegistry.load_default()
    av = reg.get("maya")
    bid = av.target_bone_id("Hips")
    assert isinstance(bid, int)
    # UpperArm.L canonical maps to Maya's "Left arm" bone
    bid_arm = av.target_bone_id("UpperArm.L")
    assert isinstance(bid_arm, int)
    assert bid != bid_arm


def test_registry_get_unknown_raises():
    reg = AvatarRegistry.load_default()
    with pytest.raises(RegistryError, match="unknown avatar"):
        reg.get("does_not_exist")


def test_registry_loads_missing_file_raises(tmp_path):
    with pytest.raises(RegistryError, match="missing"):
        AvatarRegistry.load(tmp_path / "nope.json")


# --- Phase A donor identification ------------------------------------------


def test_identify_donor_picks_curated_match():
    reg = AvatarRegistry.load_default()
    av = reg.get("maya")
    # Clothing fingerprint = exactly Maya's bone names → score 1.0
    donor_id, score, ranked = identify_donor(av.bone_names, reg)
    assert donor_id == "maya"
    assert score == 1.0


def test_identify_donor_below_threshold_raises():
    reg = AvatarRegistry.load_default()
    clothing = ["J_Bip_Hips", "J_Bip_Spine", "J_Bip_Chest"]  # 3 fake bones
    with pytest.raises(DonorIdentificationError) as exc_info:
        identify_donor(clothing, reg, threshold=0.85)
    assert exc_info.value.top_candidates
    top_id, top_score = exc_info.value.top_candidates[0]
    assert top_id in reg.ids()
    assert top_score < 0.85


def test_identify_donor_high_partial_overlap():
    """A clothing that shares many bones with the donor (but missing some)
    should still match if Jaccard >= threshold."""
    reg = AvatarRegistry.load_default()
    av = reg.get("maya")
    # Clothing = 95% of Maya's bones
    take_count = int(len(av.bone_names) * 0.95)
    clothing = av.bone_names[:take_count]
    donor_id, score, ranked = identify_donor(clothing, reg)
    assert donor_id == "maya"
    assert score >= 0.85


def test_identify_donor_threshold_override():
    """Caller can lower threshold for v2-style permissive matching."""
    reg = AvatarRegistry.load_default()
    clothing = ["J_Bip_Hips", "J_Bip_Spine"]
    # Permissive threshold should NOT raise even though Jaccard is tiny.
    donor_id, score, ranked = identify_donor(clothing, reg, threshold=0.0)
    assert donor_id in reg.ids()


def test_identify_donor_picks_moe_over_maya():
    """With two curated avatars, fingerprint must pick the right one."""
    reg = AvatarRegistry.load_default()
    moe = reg.get("moe")
    donor_id, score, ranked = identify_donor(moe.bone_names, reg)
    assert donor_id == "moe"
    assert score == 1.0
    others = {cid: s for cid, s in ranked if cid != "moe"}
    assert all(s < 1.0 for s in others.values())
