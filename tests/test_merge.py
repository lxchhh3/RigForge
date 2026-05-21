"""Tests for cross-avatar armature merge primitives (rigforge/ascii_fbx/merge.py).

These exercise the byte-level building blocks Phase C uses when donor != target:
  - positioned_args:        arg tokenizer with byte spans
  - id_offset_edits:        rewrite every clothing object id + every reference
  - cluster_repoint_edits:  swap clothing-bone-id for target-bone-id in
                            bone↔cluster connections only
  - bone_keep_strip_edits:  drop a kept bone's Model + parent connection,
                            but keep its cluster bodies
  - splice_into_section_edit: append bytes before a section's closing brace
"""
from __future__ import annotations

from rigforge.ascii_fbx.edits import apply_edits
from rigforge.ascii_fbx.lexer import parse as parse_fbx
from rigforge.ascii_fbx.merge import (
    bone_keep_strip_edits,
    cluster_repoint_edits,
    id_offset_edits,
    positioned_args,
    splice_into_section_edit,
    zero_weight_leaf_bone_ids,
)
from rigforge.ascii_fbx.sections import extract


# Minimal but structurally valid ASCII FBX fragment.
# Two bones (100 Hips, 101 Spine), a Skin (200), a Cluster on Hips (201),
# plus parent (Spine→Hips), skin attach (Hips→Cluster), Cluster→Skin
SAMPLE = b"""FBXHeaderExtension:  {
\tFBXHeaderVersion: 1003
}
Objects:  {
\tModel: 100, "Model::Hips", "LimbNode" {
\t\tProperties70:  {
\t\t}
\t}
\tModel: 101, "Model::Spine", "LimbNode" {
\t\tProperties70:  {
\t\t}
\t}
\tGeometry: 300, "Geometry::ShirtMesh", "Mesh" {
\t}
\tModel: 400, "Model::ShirtMeshModel", "Mesh" {
\t}
\tDeformer: 200, "Deformer::Skin1", "Skin" {
\t}
\tDeformer: 201, "SubDeformer::Hips", "Cluster" {
\t\tIndexes: *3 {
\t\t\ta: 1,2,3
\t\t}
\t}
}
Connections:  {
\tC: "OO", 101, 100
\tC: "OO", 100, 201
\tC: "OO", 201, 200
\tC: "OO", 200, 400
}
"""


# --- positioned arg tokenizer ---------------------------------------------


def test_positioned_args_picks_numeric_token_span():
    doc = parse_fbx(SAMPLE)
    objects = doc.root("Objects")
    hips_node = objects.children[0]  # Model: 100, ...
    args_start, args_end = hips_node.args_span
    region = SAMPLE[args_start:args_end]
    tokens = positioned_args(region)
    # First token is the numeric id 100
    assert tokens[0].kind == "num"
    assert tokens[0].value == 100
    # Byte span must yield "100" when sliced from the region
    assert region[tokens[0].start : tokens[0].end] == b"100"
    # Second token is the quoted name
    assert tokens[1].kind == "str"
    assert tokens[1].value == "Model::Hips"


def test_positioned_args_handles_negative_and_float():
    tokens = positioned_args(b' -42, 3.14 , "x" ')
    kinds = [t.kind for t in tokens]
    vals = [t.value for t in tokens]
    assert kinds == ["num", "num", "str"]
    assert vals == [-42, 3.14, "x"]


# --- id offset ------------------------------------------------------------


def test_id_offset_edits_shifts_every_object_id():
    doc = parse_fbx(SAMPLE)
    view = extract(doc)
    edits = id_offset_edits(view, offset=1_000_000, repoint={})
    out = apply_edits(SAMPLE, edits)
    # Every object id must have moved
    assert b"Model: 1000100," in out
    assert b"Model: 1000101," in out
    assert b"Geometry: 1000300," in out
    assert b"Deformer: 1000200," in out
    assert b"Deformer: 1000201," in out
    # And the originals must be gone
    assert b"Model: 100," not in out
    assert b"Model: 101," not in out


def test_id_offset_edits_shifts_connection_refs():
    doc = parse_fbx(SAMPLE)
    view = extract(doc)
    edits = id_offset_edits(view, offset=1_000_000, repoint={})
    out = apply_edits(SAMPLE, edits)
    assert b'C: "OO", 1000101, 1000100' in out  # parent-child
    assert b'C: "OO", 1000100, 1000201' in out  # bone→cluster
    assert b'C: "OO", 1000201, 1000200' in out  # cluster→skin
    assert b'C: "OO", 1000200, 1000400' in out  # skin→mesh


def test_id_offset_leaves_rootnode_zero_alone():
    """RootNode (id=0) is the scene root, shared between clothing and target
    after splice. It must NEVER be offset, otherwise clothing top-level objects
    end up parented to a non-existent node."""
    fbx = b"""Objects:  {
\tModel: 100, "Model::Hips", "LimbNode" {
\t}
\tModel: 50, "Model::Mesh1", "Mesh" {
\t}
}
Connections:  {
\tC: "OO", 100, 0
\tC: "OO", 50, 0
}
"""
    doc = parse_fbx(fbx)
    view = extract(doc)
    edits = id_offset_edits(view, offset=1000, repoint={})
    out = apply_edits(fbx, edits)
    # Both objects' parent should still be 0 (RootNode), not 1000
    assert b'C: "OO", 1100, 0' in out
    assert b'C: "OO", 1050, 0' in out
    assert b'C: "OO", 100, 1000' not in out


def test_id_offset_with_repoint_keeps_repointed_ids_unchanged():
    """When repoint maps clothing-bone-id -> target-bone-id, the connection
    arg uses the target id directly (no offset). Other clothing ids still
    get offset."""
    doc = parse_fbx(SAMPLE)
    view = extract(doc)
    # Repoint clothing Hips (100) to target bone id 7777777
    edits = id_offset_edits(view, offset=1_000_000, repoint={100: 7777777})
    out = apply_edits(SAMPLE, edits)
    # The bone Model's own header still keeps its (offset) id — we'll
    # drop the bone Model in bone_keep_strip_edits separately. id_offset
    # only governs the numbers it writes; the dropping is orthogonal.
    # Connection arg 100 (skin attach) becomes 7777777 (target id)
    assert b'C: "OO", 7777777, 1000201' in out
    # Connection arg 100 used as parent of 101 ALSO gets repointed
    # (any reference to id=100 in connections becomes target id).
    assert b'C: "OO", 1000101, 7777777' in out


# --- cluster repoint ------------------------------------------------------


def test_cluster_repoint_edits_only_touches_bone_cluster_connections():
    """cluster_repoint_edits should rewrite ONLY connections of the form
    bone↔cluster. Parent-child (bone↔bone) must be untouched."""
    doc = parse_fbx(SAMPLE)
    view = extract(doc)
    # Hips id=100 attaches to Cluster id=201
    repoint_table = {100: 9999999}
    edits = cluster_repoint_edits(view, repoint_table)
    out = apply_edits(SAMPLE, edits)
    # Bone→Cluster: 100 -> 9999999
    assert b'C: "OO", 9999999, 201' in out
    # Parent-child: 101→100 must NOT be touched (100 is parent here, not cluster)
    assert b'C: "OO", 101, 100' in out


# --- bone keep strip ------------------------------------------------------


def test_bone_keep_strip_drops_model_and_own_parent_ref_only():
    """For a kept bone (cross-avatar merge), we drop the Model node and the
    bone's OWN upward parent reference (src=bone_id, dst=parent_model).
    Connections where this bone is the PARENT (dst=bone_id) stay so the
    id_offset+repoint pass can redirect children to the target bone id.
    Skin-attach (src=bone_id, dst=cluster) is also preserved.
    """
    doc = parse_fbx(SAMPLE)
    view = extract(doc)
    spine = next(b for b in view.bones.values() if b.name == "Spine")
    edits = bone_keep_strip_edits(view, spine)
    out = apply_edits(SAMPLE, edits)
    # Model: 101 Spine header is gone
    assert b'"Model::Spine"' not in out
    # Hips Model is untouched
    assert b'"Model::Hips"' in out
    # Spine's own parent ref (C: "OO", 101, 100) is gone
    assert b'C: "OO", 101, 100' not in out


def test_bone_keep_strip_keeps_cluster_connection_for_repoint():
    """The skin-attach connection (bone→cluster) must survive — the id_offset
    pass with a repoint will redirect its bone-side arg to the target id."""
    doc = parse_fbx(SAMPLE)
    view = extract(doc)
    hips = next(b for b in view.bones.values() if b.name == "Hips")
    edits = bone_keep_strip_edits(view, hips)
    out = apply_edits(SAMPLE, edits)
    assert b'"Model::Hips"' not in out
    assert b'"SubDeformer::Hips"' in out
    # Bone→Cluster (C: "OO", 100, 201) survives so id_offset can repoint it.
    assert b'C: "OO", 100, 201' in out


# --- splice ----------------------------------------------------------------


def test_splice_into_section_inserts_before_closing_brace():
    """splice_into_section_edit appends a bytes blob into the named section
    just before its '}'. Used to merge clothing Objects/Connections into the
    target's."""
    doc = parse_fbx(SAMPLE)
    edit = splice_into_section_edit(doc, "Objects", b"\tFoo: 999, \"Foo::Bar\" {\n\t}\n")
    out = apply_edits(SAMPLE, [edit])
    # The Foo node lives inside Objects (before the closing brace)
    obj_open = out.index(b"Objects:  {")
    obj_close = out.index(b"\n}\n", obj_open)
    assert b"Foo: 999, \"Foo::Bar\"" in out[obj_open:obj_close]
    # Connections section remains intact
    assert b"Connections:  {" in out


# --- zero-weight leaf sweep ------------------------------------------------


# Chain: Spine (weighted) -> Hand (weighted) -> Hand_end (zero weight, leaf).
# Sibling chain off Spine: HelperA (no cluster at all) -> HelperB (cluster
# with weight_count=0). All three weightless nodes should be pruned; the
# weighted core (Spine, Hand) survives.
ZERO_WEIGHT_SAMPLE = b"""FBXHeaderExtension:  {
\tFBXHeaderVersion: 1003
}
Objects:  {
\tModel: 100, "Model::Spine", "LimbNode" {
\t}
\tModel: 200, "Model::Hand", "LimbNode" {
\t}
\tModel: 300, "Model::Hand_end", "LimbNode" {
\t}
\tModel: 400, "Model::HelperA", "LimbNode" {
\t}
\tModel: 500, "Model::HelperB", "LimbNode" {
\t}
\tGeometry: 90, "Geometry::Body", "Mesh" {
\t}
\tDeformer: 80, "Deformer::Body_Skin", "Skin" {
\t}
\tDeformer: 1000, "SubDeformer::Spine_Cluster", "Cluster" {
\t\tIndexes: *3 {
\t\t\ta: 0,1,2
\t\t}
\t}
\tDeformer: 1001, "SubDeformer::Hand_Cluster", "Cluster" {
\t\tIndexes: *2 {
\t\t\ta: 3,4
\t\t}
\t}
\tDeformer: 1002, "SubDeformer::Hand_end_Cluster", "Cluster" {
\t\tIndexes: *0 {
\t\t}
\t}
\tDeformer: 1005, "SubDeformer::HelperB_Cluster", "Cluster" {
\t\tIndexes: *0 {
\t\t}
\t}
}
Connections:  {
\tC: "OO",100,0
\tC: "OO",200,100
\tC: "OO",300,200
\tC: "OO",400,100
\tC: "OO",500,400
\tC: "OO",100,1000
\tC: "OO",200,1001
\tC: "OO",300,1002
\tC: "OO",500,1005
\tC: "OO",1000,80
\tC: "OO",1001,80
\tC: "OO",1002,80
\tC: "OO",1005,80
\tC: "OO",80,90
}
"""


def test_zero_weight_sweep_drops_leaf_with_zero_count_cluster():
    """A leaf bone whose only cluster has Indexes: *0 must be flagged for
    drop. weight_count==0 means no vertex is skinned to this bone — it
    contributes nothing to deformation."""
    view = extract(parse_fbx(ZERO_WEIGHT_SAMPLE))
    ids = zero_weight_leaf_bone_ids(view)
    assert 300 in ids, "Hand_end (zero-count cluster, leaf) must be dropped"


def test_zero_weight_sweep_drops_bone_with_no_cluster():
    """A leaf bone with no cluster at all has no skin influence by definition
    and must be swept."""
    view = extract(parse_fbx(ZERO_WEIGHT_SAMPLE))
    ids = zero_weight_leaf_bone_ids(view)
    assert 400 in ids, "HelperA (no cluster) must be dropped after HelperB goes"
    assert 500 in ids, "HelperB (zero-count cluster, leaf) must be dropped"


def test_zero_weight_sweep_preserves_weighted_chain():
    """Bones with non-zero weight clusters survive, even if a sibling has
    zero weight. The whole point of leaf-pruning is to keep the deforming
    skeleton intact."""
    view = extract(parse_fbx(ZERO_WEIGHT_SAMPLE))
    ids = zero_weight_leaf_bone_ids(view)
    assert 100 not in ids, "Spine has weight, must survive"
    assert 200 not in ids, "Hand has weight, must survive"


def test_zero_weight_sweep_iterates_until_stable():
    """HelperA has no cluster but its CHILD (HelperB) does — except HelperB's
    cluster is zero-count. Iteration must catch HelperB on pass 1, then
    HelperA on pass 2 once HelperB is in the drop set (so HelperA has no
    kept children). Single-pass would miss HelperA."""
    view = extract(parse_fbx(ZERO_WEIGHT_SAMPLE))
    ids = zero_weight_leaf_bone_ids(view)
    assert {300, 400, 500} <= ids


def test_zero_weight_sweep_skips_non_limb_models():
    """Mesh and Null Models are out of scope — only LimbNode bones can be
    swept. A free-floating Mesh-Model with no cluster is still meant to
    appear in the output."""
    view = extract(parse_fbx(ZERO_WEIGHT_SAMPLE))
    ids = zero_weight_leaf_bone_ids(view)
    for bid in ids:
        bone = view.bones[bid]
        assert bone.type_class == "LimbNode"
