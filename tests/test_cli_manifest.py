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
    # --clothing repeats (several garments -> ONE merged output), so it
    # always parses to a list.
    assert args.clothing == ["in.fbx"]
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


def test_parser_assemble_accepts_bone_transform_flags():
    parser = build_parser()
    args = parser.parse_args([
        "assemble", "--clothing", "in.fbx", "--target", "maya", "--out", "out.fbx",
        "--llm-mock", "decisions.json",
        "--bone-transform", "E#5_Hat.Root;pos=0,0.02,0",
        "--bone-transform", "#123;rot=0,0,10",
        "--target-bone-transform", "Hat Root;scale=1.1,1.1,1.1",
        "--bone-transform-frame", "fbx",
    ])
    assert args.bone_transform == ["E#5_Hat.Root;pos=0,0.02,0", "#123;rot=0,0,10"]
    assert args.target_bone_transform == ["Hat Root;scale=1.1,1.1,1.1"]
    assert args.bone_transform_frame == "fbx"


def test_parser_bone_transform_frame_defaults_to_unity():
    parser = build_parser()
    args = parser.parse_args([
        "assemble", "--clothing", "in.fbx", "--target", "maya", "--out", "out.fbx",
        "--llm-mock", "decisions.json",
    ])
    assert args.bone_transform_frame == "unity"
    assert args.bone_transform in (None, [])


def test_cli_assemble_forwards_bone_transforms(tmp_path, monkeypatch):
    """The specs reach the orchestrator as BoneTransformEdit objects with the
    chosen frame; a malformed spec is a clean error before any work."""
    from rigforge.ascii_fbx.transform import LocalTransform
    from rigforge.pipeline.bone_transforms import BoneTransformEdit
    import rigforge.cli as cli

    captured: dict = {}

    def fake_assemble(**kwargs):
        captured.update(kwargs)
        raise cli.PipelineError("stop here")   # short-circuit after capture

    monkeypatch.setattr(cli, "assemble", fake_assemble)
    monkeypatch.setattr(cli, "_build_llm_client", lambda args: object())
    rc = main([
        "assemble", "--clothing", str(tmp_path / "in.fbx"),
        "--target", "maya", "--out", str(tmp_path / "out.fbx"),
        "--llm-mock", "decisions.json",
        "--bone-transform", "E#5_Hat.Root;pos=0,0.02,0",
        "--target-bone-transform", "#5;scale=2,2,2",
    ])
    assert rc == 1   # our fake raised PipelineError after capturing
    assert captured["bone_transforms"] == [
        BoneTransformEdit(bone="E#5_Hat.Root", transform=LocalTransform(position=(0, 0.02, 0)), frame="unity"),
    ]
    assert captured["target_bone_transforms"] == [
        BoneTransformEdit(bone=5, transform=LocalTransform(scale=(2, 2, 2)), frame="unity"),
    ]


def test_cli_assemble_rejects_malformed_bone_transform(tmp_path, monkeypatch, capsys):
    import rigforge.cli as cli
    called = []
    monkeypatch.setattr(cli, "assemble", lambda **kw: called.append(kw))
    monkeypatch.setattr(cli, "_build_llm_client", lambda args: object())
    rc = main([
        "assemble", "--clothing", str(tmp_path / "in.fbx"),
        "--target", "maya", "--out", str(tmp_path / "out.fbx"),
        "--llm-mock", "decisions.json",
        "--bone-transform", "E#5_Hat.Root;pos=0,0.02",
    ])
    assert rc == 2
    assert called == []
    assert "bone-transform" in capsys.readouterr().err


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
    from rigforge.ascii_fbx.transform import LocalTransform
    from rigforge.canonical.decisions import Decision, DecisionSet
    from rigforge.pipeline.bone_transforms import BoneTransformEdit
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
        bone_transforms=[BoneTransformEdit(
            bone="Hat Root", transform=LocalTransform(position=(0, 0.05, 0)), frame="unity",
        )],
    )
    m = build_manifest(fake_run)
    assert m["schema_version"] == 1
    assert m["input"]["donor_id"] == "maya"
    # Bone moves are written down in the manifest (unlike the FE's other
    # selections, which live only in browser storage).
    assert m["input"]["bone_transforms"] == [{
        "bone": "Hat Root", "frame": "unity",
        "position": [0.0, 0.05, 0.0], "rotation": None, "scale": None,
    }]
    assert m["input"]["target_bone_transforms"] == []
    assert m["cache"]["hit"] is True
    assert m["edit_plan"]["n_renames"] == 1
    assert m["edit_plan"]["n_drops"] == 1
    assert m["decisions_summary"]["kept"] == 1
    assert m["decisions_summary"]["dropped"] == 1
    assert m["warnings"][0]["rule"] == "x"
