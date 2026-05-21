"""Tests for the section extractor.

Two layers of coverage:
- Synthetic minimal ASCII FBX fixtures embedded inline (fast, edge-case focused)
- Real Maya_ascii.fbx (slower; sanity checks on a full character)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import (
    BoneRecord,
    SectionError,
    extract,
    parse_args,
    bone_to_mesh_names,
)


# --- args parser -------------------------------------------------------------


def test_parse_args_mixed_types():
    args = parse_args(b'123, "Model::Foo", "LimbNode"')
    assert args == [("num", 123), ("str", "Model::Foo"), ("str", "LimbNode")]


def test_parse_args_floats_and_scientific():
    args = parse_args(b"1.5, 2.0e-3, -0.5, 7")
    kinds = [(k, type(v).__name__) for k, v in args]
    assert kinds == [
        ("num", "float"), ("num", "float"), ("num", "float"), ("num", "int")
    ]
    assert args[2] == ("num", -0.5)


def test_parse_args_lcl_translation_line():
    text = b'"Lcl Translation", "Lcl Translation", "", "A",-1.5e-5,0.807,-0.04'
    args = parse_args(text)
    nums = [v for k, v in args if k == "num"]
    assert nums == [-1.5e-5, 0.807, -0.04]


def test_parse_args_asterisk_count_is_str():
    args = parse_args(b" *4782 ")
    assert args == [("str", "*4782")]


# --- minimal synthetic FBX ---------------------------------------------------


# Two-bone rig: Hips (root) -> Spine. One cluster attached to Hips.
MINI_FBX = b"""\
; FBX 7.4.0 project file
Objects:  {
\tModel: 100, "Model::Hips", "LimbNode" {
\t\tProperties70:  {
\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0,0.9,0
\t\t}
\t}
\tModel: 101, "Model::Spine", "LimbNode" {
\t\tProperties70:  {
\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0,0.1,0
\t\t}
\t}
\tDeformer: 200, "SubDeformer::Hips", "Cluster" {
\t\tVersion: 100
\t\tIndexes: *3 {
\t\t\ta: 0,1,2
\t\t}
\t}
}
Connections:  {
\t;Model::Spine, Model::Hips
\tC: "OO",101,100
\t;Model::Hips, Model::RootNode
\tC: "OO",100,0
\t;Cluster Hips -> Model Hips
\tC: "OO",200,100
}
"""


# Rig with cluster TransformLink + child without cluster + L/R/back bones.
# Used to drive the world_xyz / bucketing tests.
#
# TransformLink layout (row-major flat 16 floats); FBX stores translation at
# flat indices 12,13,14 (column-major last column from the engine's POV).
WORLD_FBX = b"""\
; FBX 7.4.0 project file
Objects:  {
\tModel: 100, "Model::Hips", "LimbNode" {
\t\tProperties70:  {
\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0,0,0
\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,0
\t\t\tP: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1
\t\t}
\t}
\tModel: 101, "Model::TipNoCluster", "LimbNode" {
\t\tProperties70:  {
\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0,0.1,0
\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,0
\t\t\tP: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1
\t\t}
\t}
\tModel: 102, "Model::LeftPanel", "LimbNode" {
\t\tProperties70:  {
\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0,0,0
\t\t}
\t}
\tModel: 103, "Model::BackPanel", "LimbNode" {
\t\tProperties70:  {
\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0,0,0
\t\t}
\t}
\tDeformer: 200, "SubDeformer::Hips", "Cluster" {
\t\tVersion: 100
\t\tIndexes: *3 { a: 0,1,2 }
\t\tTransformLink: *16 {
\t\t\ta: 1,0,0,0,0,1,0,0,0,0,1,0,0,0.8,0,1
\t\t}
\t}
\tDeformer: 201, "SubDeformer::LeftPanel", "Cluster" {
\t\tVersion: 100
\t\tIndexes: *2 { a: 0,1 }
\t\tTransformLink: *16 {
\t\t\ta: 1,0,0,0,0,1,0,0,0,0,1,0,0.15,0.85,0.05,1
\t\t}
\t}
\tDeformer: 202, "SubDeformer::BackPanel", "Cluster" {
\t\tVersion: 100
\t\tIndexes: *2 { a: 0,1 }
\t\tTransformLink: *16 {
\t\t\ta: 1,0,0,0,0,1,0,0,0,0,1,0,0,1.5,-0.1,1
\t\t}
\t}
}
Connections:  {
\tC: "OO",101,100
\tC: "OO",102,100
\tC: "OO",103,100
\tC: "OO",100,0
\tC: "OO",200,100
\tC: "OO",201,102
\tC: "OO",202,103
}
"""


def test_world_xyz_uses_cluster_transform_link():
    """When a bone has a cluster, its world_xyz comes from TransformLink[12,13,14]."""
    doc = parse(WORLD_FBX)
    view = extract(doc)
    hips = view.bones[100]
    # TransformLink row 3 was (0, 0.8, 0, 1)
    assert hips.world_xyz == pytest.approx((0.0, 0.8, 0.0))


def test_world_xyz_walks_parents_when_no_cluster():
    """TipNoCluster has no cluster — should fall back to parent's world + local."""
    doc = parse(WORLD_FBX)
    view = extract(doc)
    tip = view.bones[101]
    # Hips world Y = 0.8, tip Lcl Translation = (0, 0.1, 0), no rotation
    # → tip world Y ≈ 0.9
    assert tip.world_xyz[1] == pytest.approx(0.9, abs=1e-6)
    assert tip.world_xyz[0] == pytest.approx(0.0, abs=1e-6)


def test_bucketing_lateral_and_front_back():
    """LeftPanel sits at X=+0.15 (L), BackPanel at Z=-0.1 (B), Hips at center (C/C)."""
    doc = parse(WORLD_FBX)
    view = extract(doc)
    assert view.bones[100].lateral == "C"
    assert view.bones[100].front_back == "C"
    assert view.bones[102].lateral == "L"
    assert view.bones[103].front_back == "B"


def test_height_pct_normalizes_to_scene_y_bounds():
    """Y range here is [0.8, 1.5] (Hips bottom, BackPanel top)."""
    doc = parse(WORLD_FBX)
    view = extract(doc)
    # Hips at y=0.8 should map to 0.0; BackPanel at y=1.5 should map to 1.0
    assert view.bones[100].height_pct == pytest.approx(0.0, abs=1e-3)
    assert view.bones[103].height_pct == pytest.approx(1.0, abs=1e-3)


def test_to_json_record_includes_world_fields():
    doc = parse(WORLD_FBX)
    view = extract(doc)
    rec = view.bones[102].to_json_record()
    assert rec["world_xyz"] == pytest.approx([0.15, 0.85, 0.05])
    assert rec["lateral"] == "L"
    assert rec["front_back"] == "F"


# --- mesh affinity (bone -> deforms-which-meshes) -------------------------
#
# Two cases the inspect UI needs to surface:
#   (1) A LimbNode bone whose clusters bind to a mesh (skinned chain).
#       Chain: bone -> Cluster -> Skin -> Geometry -> Mesh-Model.
#   (2) A Mesh-type Model that directly holds a Geometry (no skinning).
#       Chain: Mesh-Model has child Geometry connected via OO.


AFFINITY_FBX = b"""\
; FBX 7.4.0 project file
Objects:  {
\tModel: 100, "Model::Hips", "LimbNode" {
\t\tProperties70:  {
\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0,0.9,0
\t\t}
\t}
\tModel: 200, "Model::SkirtMesh", "Mesh" {
\t\tProperties70:  {
\t\t}
\t}
\tModel: 250, "Model::GlovesMesh", "Mesh" {
\t\tProperties70:  {
\t\t}
\t}
\tGeometry: 300, "Geometry::Skirt", "Mesh" {
\t}
\tGeometry: 350, "Geometry::Gloves", "Mesh" {
\t}
\tDeformer: 400, "Deformer::Skirt_Skin", "Skin" {
\t\tVersion: 101
\t}
\tDeformer: 500, "SubDeformer::Hips", "Cluster" {
\t\tVersion: 100
\t\tIndexes: *3 {
\t\t\ta: 0,1,2
\t\t}
\t}
}
Connections:  {
\t;Geometry Skirt -> Model SkirtMesh
\tC: "OO",300,200
\t;Geometry Gloves -> Model GlovesMesh
\tC: "OO",350,250
\t;Skin -> Geometry
\tC: "OO",400,300
\t;Cluster -> Skin
\tC: "OO",500,400
\t;Cluster -> Hips (bone)
\tC: "OO",500,100
\t;Hips -> RootNode
\tC: "OO",100,0
\t;SkirtMesh, GlovesMesh -> RootNode (scene parents)
\tC: "OO",200,0
\tC: "OO",250,0
}
"""


def test_extract_parses_geometries():
    view = extract(parse(AFFINITY_FBX))
    assert 300 in view.geometries
    assert 350 in view.geometries
    assert view.geometries[300].name == "Skirt"
    assert view.geometries[350].name == "Gloves"


def test_extract_parses_skins():
    view = extract(parse(AFFINITY_FBX))
    assert 400 in view.skins
    # The skin knows which geometry it attaches to.
    assert view.skins[400].geometry_id == 300


def test_bone_to_mesh_names_for_skinned_bone():
    """A LimbNode's mesh affinity comes from cluster -> skin -> geometry ->
    owning Mesh-Model. Hips's only cluster (weighted) binds via the Skirt
    skin, so the affinity surface should list 'SkirtMesh'."""
    view = extract(parse(AFFINITY_FBX))
    names = bone_to_mesh_names(view, 100)
    assert names == ["SkirtMesh"]


def test_bone_to_mesh_names_for_mesh_type_model():
    """A Mesh-type Model directly holds its own Geometry — it 'deforms' that
    one mesh (itself). The Gloves Mesh-Model has no skinning chain; it just
    owns Geometry::Gloves."""
    view = extract(parse(AFFINITY_FBX))
    # GlovesMesh has no clusters/skin — its affinity is the geometry it owns.
    assert bone_to_mesh_names(view, 250) == ["GlovesMesh"]


def test_bone_to_mesh_names_empty_for_non_deforming():
    """A LimbNode bone with no clusters and no skinning chain → empty list."""
    view = extract(parse(MINI_FBX))
    # Spine (101) has no cluster at all in MINI_FBX
    assert bone_to_mesh_names(view, 101) == []


# Chain-root pattern: a bone with zero-weight clusters whose CHILD has the
# real weighted cluster. Maya rigs do this — every bone is added to every
# mesh's skin even with zero weights, so we must ignore zero-weight clusters
# AND walk descendants to surface the meaningful affinity at chain roots.
CHAIN_ROOT_FBX = b"""\
; FBX 7.4.0 project file
Objects:  {
\tModel: 100, "Model::PB_skirt_root", "LimbNode" {
\t\tProperties70:  {
\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0,0,0
\t\t}
\t}
\tModel: 110, "Model::PB_skirt_001", "LimbNode" {
\t\tProperties70:  {
\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",0,-0.1,0
\t\t}
\t}
\tModel: 200, "Model::SkirtMesh", "Mesh" {
\t\tProperties70:  {
\t\t}
\t}
\tGeometry: 300, "Geometry::Skirt", "Mesh" {
\t}
\tDeformer: 400, "Deformer::Skirt_Skin", "Skin" {
\t\tVersion: 101
\t}
\tDeformer: 500, "SubDeformer::PB_skirt_root", "Cluster" {
\t\tVersion: 100
\t}
\tDeformer: 510, "SubDeformer::PB_skirt_001", "Cluster" {
\t\tVersion: 100
\t\tIndexes: *4 {
\t\t\ta: 0,1,2,3
\t\t}
\t}
}
Connections:  {
\tC: "OO",300,200
\tC: "OO",400,300
\tC: "OO",500,400
\tC: "OO",510,400
\tC: "OO",500,100
\tC: "OO",510,110
\tC: "OO",110,100
\tC: "OO",100,0
\tC: "OO",200,0
}
"""


def test_bone_to_mesh_names_skips_zero_weight_clusters():
    """PB_skirt_root's own cluster has weight_count=0 (Maya adds the bone to
    every skin even with zero weights). That zero-weight cluster must NOT
    count as real affinity — only weighted clusters do."""
    view = extract(parse(CHAIN_ROOT_FBX))
    # Cluster 510 (PB_skirt_001 → Skirt) has weights; cluster 500 doesn't.
    # The leaf bone 110 should report SkirtMesh.
    assert bone_to_mesh_names(view, 110) == ["SkirtMesh"]


def test_bone_to_mesh_names_walks_subtree_for_chain_root():
    """A chain root (PB_skirt_root) with no direct weights but a child that
    DOES deform Skirt should still surface 'SkirtMesh' — the user thinks of
    the chain by its root, and dropping the root drops the whole chain."""
    view = extract(parse(CHAIN_ROOT_FBX))
    assert bone_to_mesh_names(view, 100) == ["SkirtMesh"]


def test_extract_minimal_two_bone():
    doc = parse(MINI_FBX)
    view = extract(doc)
    assert set(view.bones) == {100, 101}
    hips = view.bones[100]
    spine = view.bones[101]

    assert hips.name == "Hips"
    assert hips.full_name == "Model::Hips"
    assert hips.type_class == "LimbNode"
    assert hips.parent_id == 0           # RootNode
    assert hips.parent_name is None
    assert hips.children_ids == [101]
    assert hips.child_names == ["Spine"]
    assert hips.translation_xyz == (0.0, 0.9, 0.0)
    assert hips.has_skin_cluster is True
    assert hips.cluster_weight_count == 3

    assert spine.parent_id == 100
    assert spine.parent_name == "Hips"
    assert spine.children_ids == []
    assert spine.has_skin_cluster is False
    assert spine.cluster_weight_count == 0


def test_to_json_record_shape():
    doc = parse(MINI_FBX)
    view = extract(doc)
    rec = view.bones[100].to_json_record()
    assert set(rec) == {
        "model_id", "name", "parent_name",
        "translation_xyz", "child_names",
        "has_skin_cluster", "cluster_weight_count",
        "world_xyz", "height_pct", "lateral", "front_back",
    }
    assert rec["model_id"] == 100
    assert rec["translation_xyz"] == [0.0, 0.9, 0.0]
    assert isinstance(rec["world_xyz"], list) and len(rec["world_xyz"]) == 3
    assert isinstance(rec["height_pct"], (int, float))
    assert rec["lateral"] in {"L", "R", "C"}
    assert rec["front_back"] in {"F", "B", "C"}


def test_json_records_filters_by_type_class():
    doc = parse(MINI_FBX)
    view = extract(doc)
    recs = view.json_records(type_classes=("LimbNode",))
    assert len(recs) == 2
    recs_mesh = view.json_records(type_classes=("Mesh",))
    assert recs_mesh == []


def test_section_error_when_no_objects():
    """Documents lacking Objects must raise (we can't extract bones without it)."""
    doc = parse(b"Documents:  {\n  Count: 1\n}\n")
    with pytest.raises(SectionError):
        extract(doc)


# --- cluster handling --------------------------------------------------------


CLUSTER_NO_INDEXES = b"""\
Objects:  {
\tModel: 1, "Model::Hips", "LimbNode" {
\t}
\tDeformer: 50, "SubDeformer::Hips", "Cluster" {
\t\tVersion: 100
\t}
}
Connections:  {
\tC: "OO",1,0
\tC: "OO",50,1
}
"""


def test_cluster_without_indexes_yields_zero_weight():
    """Some Clusters have no Indexes child (no skinned vertices). Should not crash."""
    doc = parse(CLUSTER_NO_INDEXES)
    view = extract(doc)
    hips = view.bones[1]
    assert hips.has_skin_cluster is True   # Cluster is still connected
    assert hips.cluster_weight_count == 0


# --- real fixture ------------------------------------------------------------


def test_real_maya_extracts_limb_bones(maya_fbx_ascii: Path):
    doc = parse(maya_fbx_ascii.read_bytes())
    view = extract(doc)
    limbs = view.limb_bones()
    # PLAN README says 1368 clusters; bone count is in the low hundreds.
    assert 50 < len(limbs) < 500, f"unexpected limb count: {len(limbs)}"
    names = {b.name for b in limbs}
    # Spot-check: Maya.fbx has these canonical roots
    for required in ("Hips", "Spine", "Chest", "Head"):
        assert required in names, f"missing {required} in {sorted(names)[:20]}..."


def test_real_maya_hierarchy(maya_fbx_ascii: Path):
    """Spine should descend from Hips; Chest from Spine."""
    doc = parse(maya_fbx_ascii.read_bytes())
    view = extract(doc)
    by_name = {b.name: b for b in view.limb_bones()}
    assert "Hips" in by_name and "Spine" in by_name and "Chest" in by_name
    assert by_name["Spine"].parent_name == "Hips"
    assert by_name["Chest"].parent_name == "Spine"


def test_real_maya_cluster_counts(maya_fbx_ascii: Path):
    """At least some bones should have cluster attachments (otherwise it's not a
    skinned rig)."""
    doc = parse(maya_fbx_ascii.read_bytes())
    view = extract(doc)
    skinned = [b for b in view.limb_bones() if b.has_skin_cluster]
    assert len(skinned) > 10, "expected many bones with skin clusters"
    total_weights = sum(b.cluster_weight_count for b in view.limb_bones())
    assert total_weights > 1000, f"total weight count seems too small: {total_weights}"
