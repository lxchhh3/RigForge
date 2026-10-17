"""Tests for the Blender subprocess wrapper config + command building.

Blender-free: these cover config resolution and argv construction (the parts
with logic). The real headless round-trip is exercised by the manual smoke
`training/smoke_blender_roundtrip.py`, the same way the fbx_env subprocess is
smoked rather than unit-run.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.geometry.blender_edit import (
    BlenderConfig,
    BlenderError,
    ENV_BLENDER_EXE,
    ENV_SCRIPT_DIR,
    _blender_cmd,
    _resolve_config,
    bone_report,
)


def test_bone_report_builds_args_and_returns_report(tmp_path, monkeypatch):
    """`bone_report` shells out to blender_env/bone_report.py with the bone
    names + preview dir, and hands back the JSON report the script wrote."""
    import json
    import subprocess
    from rigforge.geometry import blender_edit

    exe = tmp_path / "blender.exe"
    exe.write_text("")
    scripts = tmp_path / "blender_env"
    scripts.mkdir()
    (scripts / "bone_report.py").write_text("# stub")
    src = tmp_path / "in.fbx"
    src.write_bytes(b"Kaydara FBX Binary  \x00")
    cfg = BlenderConfig(blender_exe=exe, script_dir=scripts)

    seen: dict = {}

    def fake_run(config, script_name, script_args, *, timeout):
        seen["script"] = script_name
        seen["args"] = script_args
        # Emulate the script: write the report where it was told to.
        report_path = Path(script_args[script_args.index("--report") + 1])
        report_path.write_text(json.dumps({
            "bones": {"Hat Root": {"head_world": [0, 1.2, 0], "pose_head_world": [0, 1.25, 0]}},
            "previews": {},
        }))
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(blender_edit, "_run_blender", fake_run)
    report = bone_report(
        src, bones=["Hat Root", "Head"], preview_dir=tmp_path / "prev",
        frame_mesh="Body all", config=cfg,
    )
    assert seen["script"] == "bone_report.py"
    args = seen["args"]
    assert args[args.index("--in") + 1] == str(src.resolve())
    assert args[args.index("--bones") + 1] == "Hat Root,Head"
    assert args[args.index("--frame-mesh") + 1] == "Body all"
    assert args[args.index("--preview-dir") + 1] == str((tmp_path / "prev").resolve())
    assert report["bones"]["Hat Root"]["pose_head_world"] == [0, 1.25, 0]


def test_from_env_defaults(monkeypatch):
    monkeypatch.delenv(ENV_BLENDER_EXE, raising=False)
    monkeypatch.delenv(ENV_SCRIPT_DIR, raising=False)
    cfg = BlenderConfig.from_env()
    # The default exe is the Mochifitter-bundled Blender 4.0.2.
    assert cfg.blender_exe.name.lower() == "blender.exe"
    # The default script dir is the in-repo blender_env/.
    assert cfg.script_dir.name == "blender_env"


def test_env_var_override(monkeypatch, tmp_path):
    fake_exe = tmp_path / "blender.exe"
    fake_dir = tmp_path / "scripts"
    monkeypatch.setenv(ENV_BLENDER_EXE, str(fake_exe))
    monkeypatch.setenv(ENV_SCRIPT_DIR, str(fake_dir))
    cfg = BlenderConfig.from_env()
    assert cfg.blender_exe == fake_exe
    assert cfg.script_dir == fake_dir


def test_blender_cmd_structure(tmp_path):
    exe = tmp_path / "blender.exe"
    scripts = tmp_path / "blender_env"
    scripts.mkdir()
    (scripts / "geometry_edit.py").write_text("# stub")
    cfg = BlenderConfig(blender_exe=exe, script_dir=scripts)

    cmd = _blender_cmd(cfg, "geometry_edit.py", ["--in", "a.fbx", "--op", "noop"])

    assert cmd[0] == str(exe)
    # Headless, reproducible startup, then the script after the bpy flags.
    assert "--background" in cmd
    assert "--python" in cmd
    assert str(scripts / "geometry_edit.py") in cmd
    # Script args must come after the `--` separator so Blender ignores them.
    sep = cmd.index("--")
    assert cmd[sep + 1:] == ["--in", "a.fbx", "--op", "noop"]
    # The python flag and script path precede the separator.
    assert cmd.index("--python") < sep


def test_resolve_config_missing_exe_raises(tmp_path):
    scripts = tmp_path / "blender_env"
    scripts.mkdir()
    cfg = BlenderConfig(blender_exe=tmp_path / "nope.exe", script_dir=scripts)
    with pytest.raises(BlenderError, match="Blender executable not found"):
        _resolve_config(cfg)


def test_resolve_config_missing_script_dir_raises(tmp_path):
    exe = tmp_path / "blender.exe"
    exe.write_bytes(b"")
    cfg = BlenderConfig(blender_exe=exe, script_dir=tmp_path / "no_such_dir")
    with pytest.raises(BlenderError, match="Blender script dir not found"):
        _resolve_config(cfg)
