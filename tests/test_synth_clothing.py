"""Tests for the synthetic clothing fixture generator."""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.avatars.registry import AvatarRegistry, identify_donor
from training.synth_clothing import (
    DEFAULT_PERTURB_TARGETS,
    DEFAULT_PREFIX,
    build_synth_clothing,
)


def test_synth_clothing_prefixes_spine_bones(maya_fbx_ascii: Path, tmp_path: Path):
    out_ascii = tmp_path / "synth_clothing.fbx"
    out_decisions = tmp_path / "synth_decisions.json"
    build = build_synth_clothing(
        source_ascii=maya_fbx_ascii,
        out_ascii=out_ascii,
        out_decisions=out_decisions,
    )

    raw = out_ascii.read_bytes()
    view = extract(parse(raw))
    names = {b.name for b in view.limb_bones()}

    # Perturbed bones are present with prefix
    for role in DEFAULT_PERTURB_TARGETS:
        assert f"{DEFAULT_PREFIX}{role}" in names, f"missing {role} after perturb"

    # Original names are gone (they were renamed in place)
    for role in DEFAULT_PERTURB_TARGETS:
        # Maya names them by the role string (Hips→Hips, etc.)
        assert role not in names, f"original {role} should have been renamed"


def test_synth_clothing_passes_phase_a_fingerprint(maya_fbx_ascii: Path,
                                                    tmp_path: Path):
    """Perturbation must be light enough to keep Jaccard >= 0.85."""
    out_ascii = tmp_path / "synth_clothing.fbx"
    out_decisions = tmp_path / "synth_decisions.json"
    build_synth_clothing(
        source_ascii=maya_fbx_ascii,
        out_ascii=out_ascii,
        out_decisions=out_decisions,
    )

    raw = out_ascii.read_bytes()
    view = extract(parse(raw))
    bone_names = [b.name for b in view.limb_bones()]

    reg = AvatarRegistry.load_default()
    donor_id, score, _ = identify_donor(bone_names, reg)
    assert donor_id == "maya"
    assert score >= 0.85, f"synth fingerprint score too low: {score:.3f}"


def test_synth_clothing_decisions_cover_all_limb_bones(maya_fbx_ascii: Path,
                                                       tmp_path: Path):
    out_ascii = tmp_path / "synth_clothing.fbx"
    out_decisions = tmp_path / "synth_decisions.json"
    build = build_synth_clothing(
        source_ascii=maya_fbx_ascii,
        out_ascii=out_ascii,
        out_decisions=out_decisions,
    )
    raw = out_ascii.read_bytes()
    view = extract(parse(raw))
    decision_ids = {d.model_id for d in build.decisions.bones}
    limb_ids = {b.model_id for b in view.limb_bones()}
    assert limb_ids.issubset(decision_ids), \
        f"missing decisions for {limb_ids - decision_ids}"


def test_synth_decisions_match_expected_canonical_targets(maya_fbx_ascii: Path,
                                                           tmp_path: Path):
    """The mock decisions for the perturbed bones must use the canonical roles
    so Phase B's EditPlan undoes the perturbation."""
    out_ascii = tmp_path / "synth_clothing.fbx"
    out_decisions = tmp_path / "synth_decisions.json"
    build = build_synth_clothing(
        source_ascii=maya_fbx_ascii,
        out_ascii=out_ascii,
        out_decisions=out_decisions,
    )

    canonical_roles_assigned = {d.role for d in build.decisions.bones
                                if d.role in DEFAULT_PERTURB_TARGETS}
    assert canonical_roles_assigned == set(DEFAULT_PERTURB_TARGETS)
