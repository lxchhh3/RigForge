"""End-to-end smoke test: synthetic clothing → assemble → binary FBX → compare.

Slow (requires fbx_env subprocesses for ASCII↔binary + structural compare).
Skipped automatically if the fbx_env toolchain isn't present.

What this proves:
  1. Phase A picks `maya` as donor (Jaccard score >= 0.85)
  2. Phase B classifies the perturbed spine bones, validates clean, builds
     an EditPlan that undoes the JBip_ prefix
  3. Phase C applies the renames and produces a structurally-valid FBX
  4. The converter round-trips ASCII → binary cleanly
  5. fbx_compare reports zero structural drift vs. original Maya.fbx
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigforge.ascii_fbx.convert import ConverterConfig, compare
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


def test_e2e_assemble_synth_clothing_to_binary_fbx(
    maya_fbx_ascii: Path,
    maya_fbx_binary: Path,
    converter_config: ConverterConfig,
    tmp_path: Path,
):
    # --- Build synth clothing fixture
    synth_ascii = tmp_path / "synth_clothing.fbx"
    synth_decisions = tmp_path / "synth_decisions.json"
    build = build_synth_clothing(
        source_ascii=maya_fbx_ascii,
        out_ascii=synth_ascii,
        out_decisions=synth_decisions,
    )

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

    # --- Sanity: pipeline run shape
    assert run.donor_id == "maya"
    assert run.score >= 0.85
    assert run.target_id == "maya"
    assert run.edit_plan.renames, "Phase B should have produced bone renames"
    # 5 perturbed canonical bones → 5 renames in the plan
    assert len(run.edit_plan.renames) == 5
    assert run.edit_plan.drops == []
    assert out_fbx.exists()
    assert out_fbx.read_bytes()[:23].startswith(b"Kaydara FBX Binary"), \
        "output must be binary FBX"

    # --- Manifest is well-formed
    manifest = build_manifest(run)
    assert manifest["edit_plan"]["n_renames"] == 5

    # --- Structural compare: assembled vs original Maya.fbx
    result = compare(out_fbx, maya_fbx_binary, config=converter_config)
    assert result.identical, (
        f"e2e output drifted from Maya.fbx (drift={result.drift_count})\n"
        f"{result.raw_output}"
    )
