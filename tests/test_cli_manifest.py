"""Tests for the CLI and manifest emitter.

Most of the CLI surface is integration-tested implicitly in test_e2e.py.
Here we cover:
  - manifest schema/shape from a constructed PipelineRun
  - CLI argument parsing
  - CLI error paths (no --llm-* flag)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.cli import build_parser, main
from rigforge.manifest import build_manifest


def test_parser_assemble_required_args():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["assemble"])  # missing required args


def test_parser_assemble_accepts_full_args():
    parser = build_parser()
    args = parser.parse_args([
        "assemble", "--clothing", "in.fbx",
        "--target", "maya", "--out", "out.fbx",
        "--llm-mock", "decisions.json",
    ])
    assert args.cmd == "assemble"
    assert args.clothing == "in.fbx"
    assert args.target == "maya"
    assert args.llm_mock == "decisions.json"


def test_parser_assemble_llm_flags_mutually_exclusive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "assemble", "--clothing", "in.fbx",
            "--target", "maya", "--out", "out.fbx",
            "--llm-mock", "x.json", "--llm-ollama", "tag",
        ])


def test_cli_assemble_requires_llm_flag(tmp_path):
    """Running assemble without --llm-mock or --llm-ollama should hard-exit
    via SystemExit with a clear message."""
    with pytest.raises(SystemExit) as exc:
        main([
            "assemble", "--clothing", str(tmp_path / "in.fbx"),
            "--target", "maya", "--out", str(tmp_path / "out.fbx"),
        ])
    assert "llm-mock" in str(exc.value)


# --- manifest ---------------------------------------------------------------


class _FakeViolation:
    rule = "x"
    message = "y"
    bone_ids = [1]


def test_build_manifest_shape():
    from rigforge.canonical.decisions import Decision, DecisionSet
    from rigforge.pipeline.edit_plan import EditPlan
    from rigforge.pipeline.orchestrator import PipelineRun
    from rigforge.pipeline.phase_a import PhaseAResult
    from rigforge.pipeline.phase_b import PhaseBResult
    from rigforge.pipeline.phase_c import PhaseCResult

    ds = DecisionSet(
        bones=[Decision(model_id=1, role="Hips", verdict="keep"),
               Decision(model_id=2, role="aux", verdict="drop")],
        llm_model_id="mock@test",
    )
    plan = EditPlan(drops=[2], renames={1: "Pelvis"})
    fake_run = PipelineRun(
        output_fbx=Path("/tmp/out.fbx"),
        donor_id="maya", target_id="maya",
        score=1.0,
        candidates=[("maya", 1.0)],
        cache_hit=True,
        cache_key="a" * 64,
        edit_plan=plan,
        warnings=[_FakeViolation()],
        phase_a=PhaseAResult(donor_id="maya", score=1.0, candidates=[],
                              ascii_path=Path("/tmp/x.fbx"), view=None),  # type: ignore
        phase_b=PhaseBResult(decisions=ds, edit_plan=plan, warnings=[],
                              cache_hit=True, cache_key="a"*64),
        phase_c=PhaseCResult(merged_ascii=b"", notes=["pass-through"]),
        notes=["pass-through"],
    )
    m = build_manifest(fake_run)
    assert m["schema_version"] == 1
    assert m["input"]["donor_id"] == "maya"
    assert m["cache"]["hit"] is True
    assert m["edit_plan"]["n_renames"] == 1
    assert m["edit_plan"]["n_drops"] == 1
    assert m["decisions_summary"]["kept"] == 1
    assert m["decisions_summary"]["dropped"] == 1
    assert m["warnings"][0]["rule"] == "x"
