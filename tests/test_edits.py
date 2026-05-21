"""Tests for atomic ASCII FBX edits."""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.ascii_fbx.edits import (
    EditError,
    TextEdit,
    apply_edits,
    drop_bone_edits,
    drop_cluster_edits,
    drop_mesh_edits,
    drop_node_edit,
    rename_bone_edits,
)
from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract


# Two-bone rig: Hips (root) -> Spine. Cluster on Hips.
# Note: uses FBX SDK convention `C: "OO", bone_id, cluster_id` (model is src).
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
\t;Model Hips -> Cluster Hips
\tC: "OO",100,200
}
"""


# --- apply_edits primitives --------------------------------------------------


def test_apply_edits_empty_returns_source():
    src = b"hello world"
    assert apply_edits(src, []) == src


def test_apply_edits_single_replacement():
    src = b"hello world"
    edits = [TextEdit(6, 11, b"there")]
    assert apply_edits(src, edits) == b"hello there"


def test_apply_edits_multiple_non_overlapping():
    src = b"AAA BBB CCC"
    edits = [TextEdit(0, 3, b"XXX"), TextEdit(8, 11, b"YYY")]
    assert apply_edits(src, edits) == b"XXX BBB YYY"


def test_apply_edits_rejects_overlap():
    src = b"AAA BBB"
    with pytest.raises(EditError, match="overlapping"):
        apply_edits(src, [TextEdit(0, 5, b""), TextEdit(3, 7, b"")])


def test_apply_edits_rejects_out_of_bounds():
    with pytest.raises(EditError, match="out-of-bounds"):
        apply_edits(b"abc", [TextEdit(0, 100, b"")])


# --- drop edits --------------------------------------------------------------


def test_drop_leaf_removes_whole_line():
    src = b"\tA: 1\n\tB: 2\n\tC: 3\n"
    doc = parse(src)
    # parse() returns roots only — wrap inside a body for a sibling scenario.
    src2 = b"Outer:  {\n\tA: 1\n\tB: 2\n\tC: 3\n}\n"
    doc = parse(src2)
    outer = doc.roots[0]
    b_node = outer.child("B")
    edit = drop_node_edit(b_node, src2)
    out = apply_edits(src2, [edit])
    assert out == b"Outer:  {\n\tA: 1\n\tC: 3\n}\n"


def test_drop_body_node_removes_whole_block():
    src = b"Outer:  {\n\tFoo:  {\n\t\tA: 1\n\t}\n\tBar: 2\n}\n"
    doc = parse(src)
    foo = doc.roots[0].child("Foo")
    out = apply_edits(src, [drop_node_edit(foo, src)])
    assert out == b"Outer:  {\n\tBar: 2\n}\n"


def test_drop_bone_removes_model_cluster_and_connections():
    doc = parse(MINI_FBX)
    view = extract(doc)
    hips = view.bones[100]

    edits = drop_bone_edits(view, hips)
    out = apply_edits(MINI_FBX, edits)

    # Hips Model should be gone
    assert b'"Model::Hips"' not in out
    # Hips Cluster should be gone
    assert b'"SubDeformer::Hips"' not in out
    # Spine should still be present
    assert b'"Model::Spine"' in out

    # Connections section: should still parse cleanly, with no entries
    # referencing 100 or 200.
    out_doc = parse(out)
    conn = out_doc.root("Connections")
    assert conn is not None
    # Verify no C: leaf references 100 or 200
    from rigforge.ascii_fbx.sections import parse_args
    for c in conn.children:
        if c.name != "C":
            continue
        args = parse_args(c.args_bytes(out))
        if len(args) < 3:
            continue
        if args[1][0] == "num" and args[2][0] == "num":
            src_id = int(args[1][1])
            dst_id = int(args[2][1])
            assert src_id not in (100, 200)
            assert dst_id not in (100, 200)


def test_drop_cluster_does_not_drop_bone():
    doc = parse(MINI_FBX)
    view = extract(doc)
    cluster = view.clusters[200]
    out = apply_edits(MINI_FBX, drop_cluster_edits(view, cluster))
    assert b'"Model::Hips"' in out          # Hips bone retained
    assert b'"SubDeformer::Hips"' not in out  # cluster gone


# Mesh + skinning chain: Hips bone, Body mesh-Model, Geometry, Skin, Cluster.
# This is the minimum needed to verify drop_mesh_edits cleans the whole chain.
MINI_MESH_FBX = b"""\
; FBX 7.4.0 project file
Objects:  {
\tModel: 100, "Model::Hips", "LimbNode" {
\t}
\tModel: 200, "Model::Body", "Mesh" {
\t}
\tModel: 201, "Model::Hair", "Mesh" {
\t}
\tGeometry: 300, "Geometry::Body", "Mesh" {
\t\tVertices: *3 {
\t\t\ta: 0,0,0
\t\t}
\t}
\tGeometry: 301, "Geometry::Hair", "Mesh" {
\t}
\tDeformer: 400, "Deformer::Body", "Skin" {
\t\tVersion: 101
\t}
\tDeformer: 500, "SubDeformer::Hips_to_Body", "Cluster" {
\t\tVersion: 100
\t\tIndexes: *2 {
\t\t\ta: 0,1
\t\t}
\t}
}
Connections:  {
\t;Geometry::Body, Model::Body
\tC: "OO",300,200
\t;Geometry::Hair, Model::Hair
\tC: "OO",301,201
\t;Skin Body, Geometry::Body
\tC: "OO",400,300
\t;Cluster Hips_to_Body, Skin Body
\tC: "OO",500,400
\t;Hips bone, Cluster Hips_to_Body
\tC: "OO",100,500
\t;Hips, RootNode
\tC: "OO",100,0
\t;Body Mesh, RootNode
\tC: "OO",200,0
\t;Hair Mesh, RootNode
\tC: "OO",201,0
}
"""


def test_drop_mesh_removes_model_geometry_skin_clusters_and_connections():
    """Dropping a Mesh-Model must cascade to its Geometry, Skin, and every
    Cluster — otherwise the output FBX has orphan Geometries that crash
    downstream readers."""
    doc = parse(MINI_MESH_FBX)
    view = extract(doc)
    body_id = 200

    edits = drop_mesh_edits(view, body_id)
    out = apply_edits(MINI_MESH_FBX, edits)

    # Body chain gone
    assert b'"Model::Body"' not in out
    assert b'"Geometry::Body"' not in out
    assert b'"Deformer::Body"' not in out
    assert b'"SubDeformer::Hips_to_Body"' not in out

    # Bone + sibling mesh untouched
    assert b'"Model::Hips"' in out
    assert b'"Model::Hair"' in out
    assert b'"Geometry::Hair"' in out

    # Reparse: no connection references any of the dropped ids
    out_doc = parse(out)
    conn = out_doc.root("Connections")
    dropped = {200, 300, 400, 500}
    from rigforge.ascii_fbx.sections import parse_args
    for c in conn.children:
        if c.name != "C":
            continue
        args = parse_args(c.args_bytes(out))
        if len(args) < 3:
            continue
        if args[1][0] == "num" and args[2][0] == "num":
            src_id, dst_id = int(args[1][1]), int(args[2][1])
            assert src_id not in dropped, f"orphan src ref to dropped id {src_id}"
            assert dst_id not in dropped, f"orphan dst ref to dropped id {dst_id}"


def test_drop_mesh_unknown_id_returns_empty():
    """A model_id that isn't a Mesh in the view should no-op (return no edits)
    rather than crash, so the API can pass a stale id without breaking the run."""
    doc = parse(MINI_MESH_FBX)
    view = extract(doc)
    edits = drop_mesh_edits(view, 9999)
    assert edits == []


def test_drop_mesh_on_bone_model_returns_empty():
    """Passing a LimbNode id to drop_mesh_edits is a no-op — use
    drop_bone_edits for bones. Defensive: the API path shouldn't accidentally
    nuke bones just because a mesh-id was mis-routed."""
    doc = parse(MINI_MESH_FBX)
    view = extract(doc)
    assert drop_mesh_edits(view, 100) == []
    # And the resulting source is unchanged
    assert apply_edits(MINI_MESH_FBX, drop_mesh_edits(view, 100)) == MINI_MESH_FBX


# --- rename edits ------------------------------------------------------------


def test_rename_bone_updates_model_and_cluster():
    doc = parse(MINI_FBX)
    view = extract(doc)
    hips = view.bones[100]
    edits = rename_bone_edits(view, hips, "Pelvis")
    out = apply_edits(MINI_FBX, edits)

    assert b'"Model::Hips"' not in out
    assert b'"Model::Pelvis"' in out
    assert b'"SubDeformer::Hips"' not in out
    assert b'"SubDeformer::Pelvis"' in out

    # Other bone untouched
    assert b'"Model::Spine"' in out

    # Reparse and confirm structure
    out_view = extract(parse(out))
    assert out_view.bones[100].name == "Pelvis"
    assert out_view.clusters[200].name == "Pelvis"


def test_rename_does_not_change_unrelated_text():
    """Renaming Hips must NOT affect Spine, or comments about Hips elsewhere."""
    doc = parse(MINI_FBX)
    view = extract(doc)
    hips = view.bones[100]
    edits = rename_bone_edits(view, hips, "ZZZ")
    out = apply_edits(MINI_FBX, edits)
    # Comments contain "Model::Hips" verbatim — those should remain (we only edit
    # inside the Model/Cluster's args span). This is intentional in v1.
    assert out.count(b'"Model::Hips"') == 0
    assert b';Model::Hips, Model::RootNode' in out
    assert b'"Model::ZZZ"' in out


# --- preservation guarantees -------------------------------------------------


def test_unchanged_doc_round_trip():
    """No edits → output == input bytes."""
    out = apply_edits(MINI_FBX, [])
    assert out == MINI_FBX


def test_apply_edits_then_reparse_is_valid(maya_fbx_ascii: Path):
    """Drop a leaf bone from the real Maya rig and confirm the file still
    parses structurally."""
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    # Find a LimbNode with NO children (true leaf in the hierarchy).
    leaf_bones = [b for b in view.limb_bones() if not b.children_ids]
    assert leaf_bones, "no leaf bones in real fixture"
    victim = leaf_bones[0]

    edits = drop_bone_edits(view, victim)
    out = apply_edits(raw, edits)

    # Still parses (structurally well-formed)
    out_doc = parse(out)
    # Bone is actually gone
    out_view = extract(out_doc)
    assert victim.model_id not in out_view.bones


def test_rename_on_real_maya_then_reparse(maya_fbx_ascii: Path):
    raw = maya_fbx_ascii.read_bytes()
    view = extract(parse(raw))
    by_name = {b.name: b for b in view.limb_bones()}
    hips = by_name["Hips"]
    edits = rename_bone_edits(view, hips, "PELVIS")
    out = apply_edits(raw, edits)

    out_view = extract(parse(out))
    out_by_name = {b.name: b for b in out_view.limb_bones()}
    assert "PELVIS" in out_by_name
    assert "Hips" not in out_by_name
    # Original hierarchy preserved: Spine still parents under Hips's id
    assert out_view.bones[hips.model_id].name == "PELVIS"
