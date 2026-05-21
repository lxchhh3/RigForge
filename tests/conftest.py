"""Shared pytest fixtures and config."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MAYA_TOOLCHAIN = Path("D:/2files/models/vrc/Maya/Maya_Ver1.02.2")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def maya_toolchain_dir() -> Path:
    """Where the fbx_bin2ascii / fbx_ascii2bin / fbx_compare scripts and Maya.fbx live."""
    if not MAYA_TOOLCHAIN.exists():
        pytest.skip(f"Maya toolchain dir not present: {MAYA_TOOLCHAIN}")
    return MAYA_TOOLCHAIN


@pytest.fixture(scope="session")
def maya_fbx_binary(maya_toolchain_dir: Path) -> Path:
    p = maya_toolchain_dir / "Maya.fbx"
    if not p.exists():
        pytest.skip(f"Maya.fbx not present: {p}")
    return p


@pytest.fixture(scope="session")
def maya_fbx_ascii(maya_toolchain_dir: Path) -> Path:
    p = maya_toolchain_dir / "Maya_ascii.fbx"
    if not p.exists():
        pytest.skip(f"Maya_ascii.fbx not present: {p}")
    return p
