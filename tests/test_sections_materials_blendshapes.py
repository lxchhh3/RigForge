"""Tests for v2 section extraction: Materials + BlendShapes.

Same fixture style as test_sections.py — synthetic minimal ASCII FBX inlined,
plus a real-Maya smoke check that the extractor copes with the actual file.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import (
    BlendShapeChannelRecord,
    BlendShapeRecord,
    MaterialRecord,
    extract,
)


# Two materials, one BlendShape + two channels. Geometry 300 owns BlendShape 800.
# BlendShape 800 owns channels 900, 901. Channels point at their shape geometries
# 910, 911 (we don't model shapes — just verify channels round-trip).
SECTIONS_FBX = b"""\
; FBX 7.4.0 project file
Objects:  {
\tModel: 100, "Model::Hips", "LimbNode" {
\t}
\tGeometry: 300, "Geometry::ShirtMesh", "Mesh" {
\t}
\tMaterial: 700, "Material::Cloth", "" {
\t\tVersion: 102
\t\tShadingModel: "phong"
\t\tProperties70:  {
\t\t\tP: "DiffuseColor", "Color", "", "A",0.8,0.8,0.8
\t\t}
\t}
\tMaterial: 701, "Material::Skin", "" {
\t\tVersion: 102
\t\tShadingModel: "phong"
\t\tProperties70:  {
\t\t}
\t}
\tDeformer: 800, "Deformer::ShirtBlend", "BlendShape" {
\t\tVersion: 100
\t}
\tDeformer: 900, "SubDeformer::Big", "BlendShapeChannel" {
\t\tVersion: 100
\t\tDeformPercent: 0
\t\tFullWeights: *1 {
\t\t\ta: 100
\t\t}
\t}
\tDeformer: 901, "SubDeformer::Small", "BlendShapeChannel" {
\t\tVersion: 100
\t\tDeformPercent: 0
\t\tFullWeights: *1 {
\t\t\ta: 100
\t\t}
\t}
}
Connections:  {
\tC: "OO",100,0
\tC: "OO",700,300
\tC: "OO",701,300
\tC: "OO",800,300
\tC: "OO",900,800
\tC: "OO",901,800
}
"""


# --- material extraction -----------------------------------------------------


def test_extract_materials_basic_two():
    doc = parse(SECTIONS_FBX)
    view = extract(doc)
    assert set(view.materials) == {700, 701}


def test_material_record_fields():
    doc = parse(SECTIONS_FBX)
    view = extract(doc)
    cloth = view.materials[700]
    assert isinstance(cloth, MaterialRecord)
    assert cloth.material_id == 700
    assert cloth.name == "Cloth"
    assert cloth.full_name == "Material::Cloth"
    # node_ref must be the same FBXNode the lexer produced (so byte offsets
    # survive — same identity used by drop/rename ops elsewhere).
    objects = doc.root("Objects")
    material_nodes = [n for n in objects.children if n.name == "Material"]
    assert cloth.node_ref is material_nodes[0]


def test_material_byte_offsets_preserved():
    """The Material record's node_ref byte span must slice back to the source text."""
    doc = parse(SECTIONS_FBX)
    view = extract(doc)
    cloth = view.materials[700]
    start, end = cloth.node_ref.extent_span
    snippet = SECTIONS_FBX[start:end]
    assert b"Material::Cloth" in snippet
    assert snippet.startswith(b"Material:")


# --- blendshape extraction ---------------------------------------------------


def test_extract_blendshapes_and_channels():
    doc = parse(SECTIONS_FBX)
    view = extract(doc)
    assert set(view.blend_shapes) == {800}
    assert set(view.blend_shape_channels) == {900, 901}


def test_blendshape_record_fields():
    doc = parse(SECTIONS_FBX)
    view = extract(doc)
    bs = view.blend_shapes[800]
    assert isinstance(bs, BlendShapeRecord)
    assert bs.deformer_id == 800
    assert bs.name == "ShirtBlend"
    assert bs.full_name == "Deformer::ShirtBlend"


def test_blendshape_channel_record_fields():
    doc = parse(SECTIONS_FBX)
    view = extract(doc)
    big = view.blend_shape_channels[900]
    assert isinstance(big, BlendShapeChannelRecord)
    assert big.channel_id == 900
    assert big.name == "Big"
    assert big.full_name == "SubDeformer::Big"


def test_blendshape_channel_byte_offsets_preserved():
    doc = parse(SECTIONS_FBX)
    view = extract(doc)
    big = view.blend_shape_channels[900]
    start, end = big.node_ref.extent_span
    snippet = SECTIONS_FBX[start:end]
    assert b"SubDeformer::Big" in snippet
    assert snippet.startswith(b"Deformer:")


# --- channel ↔ blendshape linkage --------------------------------------------


def test_blendshape_channel_owner_link():
    """Each channel records the BlendShape Deformer id that owns it."""
    doc = parse(SECTIONS_FBX)
    view = extract(doc)
    assert view.blend_shape_channels[900].blend_shape_id == 800
    assert view.blend_shape_channels[901].blend_shape_id == 800


# --- bones still work --------------------------------------------------------


def test_existing_bone_extraction_unchanged():
    """v2 additions must not break v1 bone extraction."""
    doc = parse(SECTIONS_FBX)
    view = extract(doc)
    assert 100 in view.bones
    assert view.bones[100].name == "Hips"


# --- absence is fine ---------------------------------------------------------


_NO_MATS_FBX = b"""\
Objects:  {
\tModel: 1, "Model::Hips", "LimbNode" {
\t}
}
Connections:  {
\tC: "OO",1,0
}
"""


def test_extract_empty_materials_and_blendshapes_when_absent():
    doc = parse(_NO_MATS_FBX)
    view = extract(doc)
    assert view.materials == {}
    assert view.blend_shapes == {}
    assert view.blend_shape_channels == {}


# --- real Maya smoke ---------------------------------------------------------


def test_real_maya_has_materials(maya_fbx_ascii: Path):
    doc = parse(maya_fbx_ascii.read_bytes())
    view = extract(doc)
    # Maya.fbx readme references multiple Material:: entries (Hair, Cloth, Body...)
    names = {m.name for m in view.materials.values()}
    assert {"Hair", "Cloth", "Body"} <= names, (
        f"missing expected materials; got {sorted(names)[:10]}"
    )


def test_real_maya_has_blendshape_channels(maya_fbx_ascii: Path):
    doc = parse(maya_fbx_ascii.read_bytes())
    view = extract(doc)
    # Maya rig carries face blend shapes; at minimum a few channels exist.
    assert len(view.blend_shape_channels) > 0
    # Channels should be linked back to a BlendShape parent.
    for ch in view.blend_shape_channels.values():
        # blend_shape_id may legitimately be None if Connections is malformed,
        # but Maya.fbx is well-formed — every channel must trace to a BlendShape.
        assert ch.blend_shape_id is not None
        assert ch.blend_shape_id in view.blend_shapes
