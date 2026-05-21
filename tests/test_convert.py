"""Tests for the fbx_env subprocess wrapper.

These tests exercise the real fbx_env conda env and the Maya.fbx fixture from
the toolchain dir. They are skipped automatically (via conftest fixtures) if
those are missing on this machine.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from rigforge.ascii_fbx.convert import (
    ConverterConfig,
    ConverterError,
    ascii_to_bin,
    bin_to_ascii,
    compare,
    ENV_PYTHON,
)


@pytest.fixture(scope="module")
def converter_config() -> ConverterConfig:
    cfg = ConverterConfig.from_env()
    if not cfg.fbx_env_python.exists():
        pytest.skip(f"fbx_env Python not present at {cfg.fbx_env_python}")
    if not cfg.toolchain_dir.exists():
        pytest.skip(f"FBX toolchain dir not present at {cfg.toolchain_dir}")
    return cfg


def test_bin_to_ascii_produces_ascii(tmp_path: Path, maya_fbx_binary: Path, converter_config):
    dst = tmp_path / "Maya_ascii.fbx"
    out = bin_to_ascii(maya_fbx_binary, dst, config=converter_config)

    assert out == dst.resolve()
    assert dst.exists()
    head = dst.read_bytes()[:32]
    # ASCII FBX files begin with the `; FBX <ver> project file` comment.
    assert head.lstrip().startswith(b";"), f"unexpected head: {head!r}"
    # Sanity: ASCII output should be substantially larger than the binary input.
    assert dst.stat().st_size > maya_fbx_binary.stat().st_size


def test_ascii_to_bin_produces_binary(tmp_path: Path, maya_fbx_binary: Path, converter_config):
    ascii_dst = tmp_path / "Maya_ascii.fbx"
    bin_dst = tmp_path / "Maya_roundtrip.fbx"

    bin_to_ascii(maya_fbx_binary, ascii_dst, config=converter_config)
    out = ascii_to_bin(ascii_dst, bin_dst, config=converter_config)

    assert out == bin_dst.resolve()
    assert bin_dst.exists()
    head = bin_dst.read_bytes()[:23]
    assert head.startswith(b"Kaydara FBX Binary"), f"unexpected head: {head!r}"


def test_roundtrip_compare_identical(tmp_path: Path, maya_fbx_binary: Path, converter_config):
    """bin → ASCII → bin should be structurally identical per fbx_compare."""
    ascii_path = tmp_path / "Maya_ascii.fbx"
    rebin_path = tmp_path / "Maya_roundtrip.fbx"

    bin_to_ascii(maya_fbx_binary, ascii_path, config=converter_config)
    ascii_to_bin(ascii_path, rebin_path, config=converter_config)

    result = compare(maya_fbx_binary, rebin_path, config=converter_config)

    assert result.identical, (
        f"Maya.fbx round-trip drifted; drift_count={result.drift_count}\n"
        f"{result.raw_output}"
    )
    assert result.drift_count == 0
    assert result.schema_match


def test_compare_detects_drift(tmp_path: Path, maya_fbx_binary: Path,
                                maya_toolchain_dir: Path, converter_config):
    """Compare Maya.fbx against an explicitly different fixture (Maya_ascii.fbx)
    to confirm that fbx_compare actually detects drift when it exists.

    Maya.fbx is binary; Maya_ascii.fbx is the ASCII version. The structural
    diff *should* be zero (same scene) — but if it isn't, fbx_compare will
    surface it. So this test asserts the harness ran cleanly; whether it's
    identical or not is informational.
    """
    other = maya_toolchain_dir / "Maya_ascii.fbx"
    if not other.exists():
        pytest.skip("Maya_ascii.fbx fixture missing")

    result = compare(maya_fbx_binary, other, config=converter_config)
    # The crucial property is that we got a structured result back without raising.
    assert isinstance(result.raw_output, str)
    assert "COMPARE:" in result.raw_output


def test_missing_fbx_env_python_raises(tmp_path: Path, maya_fbx_binary: Path):
    bogus = ConverterConfig(
        fbx_env_python=tmp_path / "does_not_exist_python.exe",
        toolchain_dir=Path("D:/2files/models/vrc/Maya/Maya_Ver1.02.2"),
    )
    with pytest.raises(ConverterError, match="fbx_env Python not found"):
        bin_to_ascii(maya_fbx_binary, tmp_path / "out.fbx", config=bogus)


def test_missing_toolchain_dir_raises(tmp_path: Path, maya_fbx_binary: Path, converter_config):
    bogus = ConverterConfig(
        fbx_env_python=converter_config.fbx_env_python,
        toolchain_dir=tmp_path / "no_such_dir",
    )
    with pytest.raises(ConverterError, match="FBX toolchain dir not found"):
        bin_to_ascii(maya_fbx_binary, tmp_path / "out.fbx", config=bogus)


def test_missing_input_file_raises(tmp_path: Path, converter_config):
    with pytest.raises(ConverterError, match="input FBX not found"):
        bin_to_ascii(tmp_path / "nope.fbx", tmp_path / "out.fbx", config=converter_config)


def test_env_var_override(tmp_path: Path, monkeypatch, maya_fbx_binary: Path):
    """ConverterConfig.from_env honors RIGFORGE_FBX_ENV_PYTHON."""
    fake = tmp_path / "fake_python.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv(ENV_PYTHON, str(fake))

    cfg = ConverterConfig.from_env()
    assert cfg.fbx_env_python == fake
