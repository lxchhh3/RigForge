"""Tests for atomic ASCII FBX edits."""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.ascii_fbx.edits import (
    EditError,
    TextEdit,
    apply_edits,
    drop_blend_shape_channel_edits,
    drop_bone_edits,
    drop_cluster_edits,
    drop_mesh_edits,
    drop_node_edit,
    rename_bone_edits,
    set_blend_shape_channel_deform_percent_edits,
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


# Mesh + blendshape chain: Body mesh with one BlendShape deformer owning one
# channel that references one Shape geometry. Hair mesh has its own BlendShape
# chain that must stay untouched when only Body is dropped.
MINI_MESH_WITH_BLENDSHAPE_FBX = b"""\
; FBX 7.4.0 project file
Objects:  {
\tModel: 200, "Model::Body", "Mesh" {
\t}
\tModel: 201, "Model::Hair", "Mesh" {
\t}
\tGeometry: 300, "Geometry::Body", "Mesh" {
\t}
\tGeometry: 301, "Geometry::Hair", "Mesh" {
\t}
\tGeometry: 900, "Geometry::Body_Smile_shape", "Shape" {
\t}
\tGeometry: 901, "Geometry::Hair_Wave_shape", "Shape" {
\t}
\tDeformer: 600, "Deformer::Body_BS", "BlendShape" {
\t}
\tDeformer: 601, "Deformer::Hair_BS", "BlendShape" {
\t}
\tDeformer: 700, "SubDeformer::Smile", "BlendShapeChannel" {
\t\tDeformPercent: 0
\t}
\tDeformer: 701, "SubDeformer::Wave", "BlendShapeChannel" {
\t\tDeformPercent: 0
\t}
}
Connections:  {
\tC: "OO",300,200
\tC: "OO",301,201
\tC: "OO",600,300
\tC: "OO",601,301
\tC: "OO",700,600
\tC: "OO",701,601
\tC: "OO",900,700
\tC: "OO",901,701
\tC: "OO",200,0
\tC: "OO",201,0
}
"""


def test_drop_mesh_cascades_to_blendshape_chain():
    """Dropping a Mesh-Model cascades into the BlendShape graph attached to
    its Geometry: the BlendShape deformer, every channel it owns, and every
    Shape geometry those channels reference all get dropped together. Without
    this the morph decls survive but the Geometry→BlendShape connection is
    severed, leaving orphan shape keys that never render in Blender."""
    doc = parse(MINI_MESH_WITH_BLENDSHAPE_FBX)
    view = extract(doc)

    edits = drop_mesh_edits(view, 200)  # drop Body
    out = apply_edits(MINI_MESH_WITH_BLENDSHAPE_FBX, edits)

    # Body chain gone (mesh + geometry + blendshape + channel + shape geom)
    assert b'"Model::Body"' not in out
    assert b'"Geometry::Body"' not in out
    assert b'"Deformer::Body_BS"' not in out
    assert b'"SubDeformer::Smile"' not in out
    assert b'"Geometry::Body_Smile_shape"' not in out

    # Hair chain untouched
    assert b'"Model::Hair"' in out
    assert b'"Geometry::Hair"' in out
    assert b'"Deformer::Hair_BS"' in out
    assert b'"SubDeformer::Wave"' in out
    assert b'"Geometry::Hair_Wave_shape"' in out

    # No surviving connection references any dropped id
    out_doc = parse(out)
    dropped = {200, 300, 600, 700, 900}
    from rigforge.ascii_fbx.sections import parse_args
    for c in out_doc.root("Connections").children:
        if c.name != "C":
            continue
        args = parse_args(c.args_bytes(out))
        if len(args) < 3:
            continue
        if args[1][0] == "num" and args[2][0] == "num":
            sid, did = int(args[1][1]), int(args[2][1])
            assert sid not in dropped, f"orphan src ref to dropped id {sid}"
            assert did not in dropped, f"orphan dst ref to dropped id {did}"


# --- drop blendshape channel edits ------------------------------------------


# Minimal fixture: one Geometry with two BlendShape Deformers, each owning
# one channel; one channel ("Wink") is shared in name but distinct id from
# the other. Tests below drop one channel and assert the second + owners
# are untouched.
MINI_BLENDSHAPE_FBX = b"""\
; FBX 7.4.0 project file
Objects:  {
\tGeometry: 300, "Geometry::Body", "Mesh" {
\t}
\tDeformer: 700, "Deformer::Face", "BlendShape" {
\t\tVersion: 100
\t}
\tDeformer: 701, "Deformer::Brow", "BlendShape" {
\t\tVersion: 100
\t}
\tDeformer: 800, "SubDeformer::Smile", "BlendShapeChannel" {
\t\tVersion: 100
\t\tDeformPercent: 0
\t}
\tDeformer: 801, "SubDeformer::Wink", "BlendShapeChannel" {
\t\tVersion: 100
\t\tDeformPercent: 0
\t}
\tDeformer: 802, "SubDeformer::BrowUp", "BlendShapeChannel" {
\t\tVersion: 100
\t\tDeformPercent: 0
\t}
}
Connections:  {
\t;Face -> Geometry
\tC: "OO",700,300
\t;Brow -> Geometry
\tC: "OO",701,300
\t;Smile -> Face
\tC: "OO",800,700
\t;Wink -> Face
\tC: "OO",801,700
\t;BrowUp -> Brow
\tC: "OO",802,701
}
"""


def test_drop_blend_shape_channel_removes_node_and_connections():
    """Dropping one BlendShapeChannel must remove its node + the connection
    that links it to its owning BlendShape Deformer, while leaving the owner
    and sibling channels intact."""
    doc = parse(MINI_BLENDSHAPE_FBX)
    view = extract(doc)
    smile_id = 800

    edits = drop_blend_shape_channel_edits(view, smile_id)
    out = apply_edits(MINI_BLENDSHAPE_FBX, edits)

    # Smile gone
    assert b'"SubDeformer::Smile"' not in out
    # Owner + siblings intact
    assert b'"Deformer::Face"' in out
    assert b'"SubDeformer::Wink"' in out
    assert b'"SubDeformer::BrowUp"' in out

    # No connection still references the dropped channel id
    out_doc = parse(out)
    conn = out_doc.root("Connections")
    from rigforge.ascii_fbx.sections import parse_args
    for c in conn.children:
        if c.name != "C":
            continue
        args = parse_args(c.args_bytes(out))
        if len(args) < 3:
            continue
        if args[1][0] == "num" and args[2][0] == "num":
            assert int(args[1][1]) != smile_id
            assert int(args[2][1]) != smile_id


def test_drop_blend_shape_channel_does_not_cascade_to_owner():
    """When a channel is dropped, the BlendShape Deformer that owns it stays
    even if it had only one child. Other channels under the same owner (or a
    sibling owner) might need it; an empty BlendShape is benign in FBX."""
    doc = parse(MINI_BLENDSHAPE_FBX)
    view = extract(doc)
    edits = drop_blend_shape_channel_edits(view, 802)  # only child of "Brow"
    out = apply_edits(MINI_BLENDSHAPE_FBX, edits)
    assert b'"SubDeformer::BrowUp"' not in out
    assert b'"Deformer::Brow"' in out, (
        "owning BlendShape must survive even if all its channels are dropped"
    )


def test_drop_blend_shape_channel_unknown_id_returns_empty():
    doc = parse(MINI_BLENDSHAPE_FBX)
    view = extract(doc)
    assert drop_blend_shape_channel_edits(view, 9999) == []


def test_drop_blend_shape_channel_on_non_channel_id_returns_empty():
    """Passing a BlendShape Deformer id (not a channel) is a no-op — the
    primitive is strictly per-channel."""
    doc = parse(MINI_BLENDSHAPE_FBX)
    view = extract(doc)
    assert drop_blend_shape_channel_edits(view, 700) == []  # "Face" owner


# --- DeformPercent override -------------------------------------------------


def test_set_deform_percent_rewrites_numeric_value():
    """Setting the channel's DeformPercent must rewrite ONLY the numeric
    token — surrounding whitespace, the property name, and sibling lines
    stay byte-identical."""
    doc = parse(MINI_BLENDSHAPE_FBX)
    view = extract(doc)
    edits = set_blend_shape_channel_deform_percent_edits(view, 800, 75)
    assert edits, "expected one edit for the Smile channel"
    out = apply_edits(MINI_BLENDSHAPE_FBX, edits)

    # Smile's DeformPercent is now 75; the other channels' lines untouched.
    out_doc = parse(out)
    out_view = extract(out_doc)
    smile = out_view.blend_shape_channels[800]
    # Re-locate the DeformPercent child and read the new value
    for child in smile.node_ref.children:
        if child.name == "DeformPercent":
            args = child.args_bytes(out_doc.source).strip()
            assert args == b"75", f"expected b'75', got {args!r}"
            break
    else:
        raise AssertionError("DeformPercent child missing on Smile after edit")

    # Sibling channels' DeformPercent untouched (still 0)
    for cid in (801, 802):
        ch = out_view.blend_shape_channels[cid]
        for child in ch.node_ref.children:
            if child.name == "DeformPercent":
                assert child.args_bytes(out_doc.source).strip() == b"0"


def test_set_deform_percent_formats_int_when_whole():
    """A whole-number percent should serialize as a bare int ('50'), matching
    Maya's own ASCII export style — keeps diffs clean."""
    doc = parse(MINI_BLENDSHAPE_FBX)
    view = extract(doc)
    edits = set_blend_shape_channel_deform_percent_edits(view, 800, 50.0)
    out = apply_edits(MINI_BLENDSHAPE_FBX, edits)
    # The replacement string itself
    assert edits[0].replacement == b"50"
    assert b"DeformPercent: 50\n" in out


def test_set_deform_percent_formats_fractional_via_g():
    """A fractional percent uses %g (six significant figures), no trailing
    zeros. 33.5 → '33.5', not '33.500000'."""
    doc = parse(MINI_BLENDSHAPE_FBX)
    view = extract(doc)
    edits = set_blend_shape_channel_deform_percent_edits(view, 800, 33.5)
    assert edits[0].replacement == b"33.5"


def test_set_deform_percent_unknown_channel_returns_empty():
    doc = parse(MINI_BLENDSHAPE_FBX)
    view = extract(doc)
    assert set_blend_shape_channel_deform_percent_edits(view, 9999, 50) == []


def test_set_deform_percent_on_non_channel_id_returns_empty():
    """Passing a BlendShape Deformer id is a no-op — strictly per-channel."""
    doc = parse(MINI_BLENDSHAPE_FBX)
    view = extract(doc)
    assert set_blend_shape_channel_deform_percent_edits(view, 700, 50) == []


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
