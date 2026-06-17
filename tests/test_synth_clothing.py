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


def test_synth_clothing_keep_meshes_carves_a_partial_distinct_clothing(
    maya_fbx_ascii: Path, tmp_path: Path,
):
    """`keep_meshes` + `mesh_rename_prefix` should leave ONLY the kept meshes,
    renamed distinct, and drop every other mesh — so an assemble onto the donor
    is a real merge (target + garment), not a self-double. Bones are untouched
    (mesh drops don't cascade into the shared skeleton)."""
    out_ascii = tmp_path / "synth_clothing.fbx"
    out_decisions = tmp_path / "synth_decisions.json"

    # Full build first, to learn the source's mesh names without hardcoding.
    full = build_synth_clothing(
        source_ascii=maya_fbx_ascii,
        out_ascii=tmp_path / "full.fbx",
        out_decisions=tmp_path / "full_dec.json",
    )
    full_view = extract(parse((tmp_path / "full.fbx").read_bytes()))
    all_mesh_names = {b.name for b in full_view.bones.values()
                      if b.type_class == "Mesh"}
    assert "Cloth" in all_mesh_names, "fixture assumption: Maya has a Cloth mesh"

    build = build_synth_clothing(
        source_ascii=maya_fbx_ascii,
        out_ascii=out_ascii,
        out_decisions=out_decisions,
        keep_meshes=("Cloth",),
        mesh_rename_prefix="Synth",
    )
    assert build.clothing_mesh_names == ("SynthCloth",)

    view = extract(parse(out_ascii.read_bytes()))
    mesh_names = {b.name for b in view.bones.values() if b.type_class == "Mesh"}
    # Only the kept-and-renamed mesh survives.
    assert mesh_names == {"SynthCloth"}, f"unexpected meshes: {mesh_names}"
    # The original name is gone (renamed, not duplicated).
    assert "Cloth" not in mesh_names
    # The skeleton is untouched: spine bones still perturbed, all bones present.
    limb_names = {b.name for b in view.limb_bones()}
    for role in DEFAULT_PERTURB_TARGETS:
        assert f"{DEFAULT_PREFIX}{role}" in limb_names
    full_limb = {b.name for b in full_view.limb_bones()}
    assert limb_names == full_limb, "mesh subsetting must not change the skeleton"
