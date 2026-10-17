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


# A garment with NO bones — one rigid mesh with a blendshape, the shape of
# kevin's GG_headset.fbx (2026-10-14). Model ids are this file's own; the
# merge offsets them clear of the target's.
RIGID_GARMENT = b"""\
; FBX 7.4.0 project file
Objects:  {
\tGeometry: 9001, "Geometry::Visor", "Mesh" {
\t\tVertices: *9 {
\t\t\ta: 0,0,0,0.1,0,0,0,0.1,0
\t\t}
\t\tPolygonVertexIndex: *3 {
\t\t\ta: 0,1,-3
\t\t}
\t}
\tGeometry: 9002, "Geometry::Open", "Shape" {
\t\tVersion: 100
\t\tIndexes: *1 {
\t\t\ta: 2
\t\t}
\t\tVertices: *3 {
\t\t\ta: 0,0.05,0
\t\t}
\t}
\tModel: 9003, "Model::Visor", "Mesh" {
\t\tVersion: 232
\t\tProperties70:  {
\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",-90,0,0
\t\t}
\t}
\tDeformer: 9004, "Deformer::Visor", "BlendShape" {
\t\tVersion: 100
\t}
\tDeformer: 9005, "SubDeformer::Open", "BlendShapeChannel" {
\t\tVersion: 100
\t\tDeformPercent: 0
\t\tFullWeights: *1 {
\t\t\ta: 100
\t\t}
\t}
}
Connections:  {
\tC: "OO",9003,0
\tC: "OO",9001,9003
\tC: "OO",9004,9001
\tC: "OO",9005,9004
\tC: "OO",9002,9005
}
"""


@pytest.fixture
def rigid_garment_fbx(tmp_path: Path) -> Path:
    p = tmp_path / "visor.fbx"
    p.write_bytes(RIGID_GARMENT)
    return p
