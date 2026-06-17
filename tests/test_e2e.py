"""End-to-end test: a PARTIAL synthetic clothing → assemble → binary FBX.

Slow (requires fbx_env subprocesses for ASCII↔binary). Skipped automatically
if the fbx_env toolchain isn't present.

The synth clothing is a real *subset* of Maya — one garment mesh (`Cloth`,
renamed `SynthCloth` so it's distinct), with the rest of Maya's meshes dropped.
Its skeleton is left unperturbed, i.e. already rigged for the `maya` target (the
realistic donor==target case). Assembling it onto `maya` is therefore a genuine
merge — target body + garment — not a self-double.

What this proves:
  1. Phase A picks `maya` as donor (Jaccard score >= 0.85)
  2. Phase B classifies the full skeleton and validates clean; since the
     clothing is already maya-rigged, no renames/drops are needed
  3. Phase C MERGES: the garment is added distinctly, the clothing skeleton is
     repointed onto the target (stripped, not duplicated), the target's own
     meshes survive
  4. The converter round-trips ASCII → binary AND binary → ASCII cleanly, and
     the real binary deliverable re-parses into a structurally-valid section view

(The old assertion — byte-identity to the bare Maya.fbx — was obsolete since the
always-merge pivot `324cd02`: the output is now target+clothing, and a leaf-bone
zero-weight sweep prunes the skeleton, so identity no longer holds by design.)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from rigforge.ascii_fbx.convert import ConverterConfig, bin_to_ascii
from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.avatars.registry import AvatarRegistry
from rigforge.canonical.schema import CanonicalSchema
from rigforge.llm.mock import MockLLMClient
from rigforge.manifest import build_manifest
from rigforge.pipeline.orchestrator import assemble
from training.synth_clothing import build_synth_clothing


@pytest.fixture(scope="module")
def converter_config() -> ConverterConfig:
    cfg = ConverterConfig.from_env()
    if not cfg.fbx_env_python.exists():
        pytest.skip(f"fbx_env Python not present at {cfg.fbx_env_python}")
    if not cfg.toolchain_dir.exists():
        pytest.skip(f"fbx toolchain dir not present at {cfg.toolchain_dir}")
    return cfg


def test_e2e_assemble_partial_clothing_merges_onto_target(
    maya_fbx_ascii: Path,
    converter_config: ConverterConfig,
    tmp_path: Path,
):
    # --- Build a PARTIAL synth clothing: keep only the Cloth garment, renamed
    #     distinct so the merge is a real add (target body + this garment). The
    #     skeleton is left unperturbed (already maya-rigged).
    synth_ascii = tmp_path / "synth_clothing.fbx"
    synth_decisions = tmp_path / "synth_decisions.json"
    build = build_synth_clothing(
        source_ascii=maya_fbx_ascii,
        out_ascii=synth_ascii,
        out_decisions=synth_decisions,
        perturb_targets=(),               # already rigged for maya — no renames
        keep_meshes=("Cloth",),
        mesh_rename_prefix="Synth",
    )
    assert build.clothing_mesh_names == ("SynthCloth",)

    # --- Run the pipeline
    out_fbx = tmp_path / "assembled.fbx"
    run = assemble(
        clothing_fbx=synth_ascii,
        target_id="maya",
        out_fbx=out_fbx,
        registry=AvatarRegistry.load_default(),
        schema=CanonicalSchema.load_default(),
        llm_client=MockLLMClient(build.decisions, model_id="synth-fixture"),
        cache=None,    # disable cache so the LLM call is exercised
        work_dir=tmp_path / "work",
    )

    # --- Pipeline run shape: maya donor, clean (no renames/drops needed)
    assert run.donor_id == "maya"
    assert run.score >= 0.85
    assert run.target_id == "maya"
    assert run.edit_plan.renames == {}
    assert run.edit_plan.drops == []
    assert out_fbx.exists()
    assert out_fbx.read_bytes()[:23].startswith(b"Kaydara FBX Binary"), \
        "output must be binary FBX"

    # --- Manifest is well-formed
    manifest = build_manifest(run)
    assert manifest["edit_plan"]["n_renames"] == 0

    # --- Validate the real binary deliverable: round-trip it back to ASCII
    #     (proves the converter produced a valid FBX) and inspect the MERGE.
    roundtrip_ascii = tmp_path / "assembled_roundtrip.fbx"
    bin_to_ascii(out_fbx, roundtrip_ascii, config=converter_config)
    view = extract(parse(roundtrip_ascii.read_bytes()))

    mesh_names = [b.name for b in view.bones.values() if b.type_class == "Mesh"]
    limb_names = [b.name for b in view.bones.values() if b.type_class == "LimbNode"]

    # The garment was ADDED, distinctly — its renamed copy AND the target's own
    # Cloth both present, no collision.
    assert "SynthCloth" in mesh_names
    assert mesh_names.count("SynthCloth") == 1
    assert "Cloth" in mesh_names, "target's own Cloth mesh must survive the merge"
    # The target avatar's other body meshes survived too.
    assert "Body" in mesh_names
    assert "Hair" in mesh_names

    # The clothing's skeleton was repointed onto the target, NOT duplicated:
    # one shared skeleton, no leftover perturbation, no doubled bones.
    assert not any(n.startswith("JBip_") for n in limb_names)
    assert limb_names.count("Hips") == 1, "clothing skeleton must repoint, not double"
    dups = {n: c for n, c in Counter(limb_names).items() if c > 1}
    assert not dups, f"merge duplicated bones (skeleton not shared): {dups}"
