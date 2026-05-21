"""Tests for the Materials + BlendShapes section merge (rigforge.sections.merge).

Merge composes byte-level operations on the TARGET document. Each MergePlan is
a list of MergeEdit triples (start, end, replacement) — identical shape to
TextEdit, so the integration pass can convert them by type-cast.

Rules:
  - Materials dedup by `name` (e.g. "Cloth"). Target wins.
  - BlendShape *channels* dedup by `name` (e.g. "Big"). Target wins.
  - When a donor entity has no name collision, splice it into the target's
    Objects { ... } section before its closing brace.
  - When donor and target are identical (no-op merge), the plan should be
    empty (round-trip identity).
"""
from __future__ import annotations

from dataclasses import asdict

from rigforge.ascii_fbx.edits import TextEdit, apply_edits
from rigforge.ascii_fbx.lexer import parse as parse_fbx
from rigforge.ascii_fbx.sections import extract
from rigforge.sections.merge import (
    MergeEdit,
    MergePlan,
    merge_blendshapes,
    merge_materials,
)


def _to_text_edits(plan: MergePlan) -> list[TextEdit]:
    """Materialize MergePlan into TextEdits by structural shape match.
    Mirrors what the integration pass will do — the merge module deliberately
    does not import edits.py."""
    return [TextEdit(e.start, e.end, e.replacement) for e in plan.edits]


# --- minimal fixtures --------------------------------------------------------


_TARGET = b"""\
Objects:  {
\tModel: 1, "Model::Hips", "LimbNode" {
\t}
\tMaterial: 700, "Material::Cloth", "" {
\t\tVersion: 102
\t\tShadingModel: "phong"
\t}
\tMaterial: 701, "Material::Skin", "" {
\t\tVersion: 102
\t\tShadingModel: "phong"
\t}
\tDeformer: 800, "Deformer::TargetBlend", "BlendShape" {
\t\tVersion: 100
\t}
\tDeformer: 900, "SubDeformer::Smile", "BlendShapeChannel" {
\t\tVersion: 100
\t\tDeformPercent: 0
\t}
}
Connections:  {
\tC: "OO",1,0
\tC: "OO",900,800
}
"""


# Donor has:
#  - "Cloth" (collision — target wins, donor dropped)
#  - "Hat"   (new — donor copied in)
#  - channel "Smile" (collision — target wins, donor dropped)
#  - channel "Wink"  (new — donor copied in)
_DONOR = b"""\
Objects:  {
\tMaterial: 770, "Material::Cloth", "" {
\t\tVersion: 102
\t\tShadingModel: "lambert"
\t}
\tMaterial: 771, "Material::Hat", "" {
\t\tVersion: 102
\t\tShadingModel: "phong"
\t}
\tDeformer: 880, "Deformer::DonorBlend", "BlendShape" {
\t\tVersion: 100
\t}
\tDeformer: 990, "SubDeformer::Smile", "BlendShapeChannel" {
\t\tVersion: 100
\t\tDeformPercent: 0
\t}
\tDeformer: 991, "SubDeformer::Wink", "BlendShapeChannel" {
\t\tVersion: 100
\t\tDeformPercent: 0
\t}
}
Connections:  {
\tC: "OO",990,880
\tC: "OO",991,880
}
"""


# --- shape of MergePlan ------------------------------------------------------


def test_merge_plan_has_edits_attribute():
    target = extract(parse_fbx(_TARGET))
    donor = extract(parse_fbx(_DONOR))
    plan = merge_materials(donor=donor, target=target)
    assert isinstance(plan, MergePlan)
    assert isinstance(plan.edits, list)


def test_merge_edit_shape_matches_text_edit():
    """A MergeEdit must have start/end/replacement and convert cleanly to TextEdit."""
    e = MergeEdit(start=0, end=0, replacement=b"hello")
    fields = set(asdict(e).keys())
    assert fields == {"start", "end", "replacement"}
    te = TextEdit(e.start, e.end, e.replacement)
    assert te.start == 0
    assert te.end == 0
    assert te.replacement == b"hello"


# --- no-op (identical donor and target) --------------------------------------


def test_merge_materials_noop_returns_empty_plan():
    """Merging a doc with itself produces no edits (everything is a collision)."""
    a = extract(parse_fbx(_TARGET))
    b = extract(parse_fbx(_TARGET))
    plan = merge_materials(donor=a, target=b)
    assert plan.edits == []


def test_merge_blendshapes_noop_returns_empty_plan():
    a = extract(parse_fbx(_TARGET))
    b = extract(parse_fbx(_TARGET))
    plan = merge_blendshapes(donor=a, target=b)
    assert plan.edits == []


# --- round-trip on no-op ------------------------------------------------------


def test_merge_materials_noop_round_trip_byte_identical():
    """Apply an empty plan → source bytes unchanged."""
    target_doc = parse_fbx(_TARGET)
    a = extract(target_doc)
    b = extract(parse_fbx(_TARGET))
    plan = merge_materials(donor=b, target=a)
    out = apply_edits(_TARGET, _to_text_edits(plan))
    assert out == _TARGET


def test_merge_blendshapes_noop_round_trip_byte_identical():
    target_doc = parse_fbx(_TARGET)
    a = extract(target_doc)
    b = extract(parse_fbx(_TARGET))
    plan = merge_blendshapes(donor=b, target=a)
    out = apply_edits(_TARGET, _to_text_edits(plan))
    assert out == _TARGET


# --- material merge ----------------------------------------------------------


def test_merge_materials_target_wins_on_name_collision():
    """`Cloth` exists in both — donor's must NOT be appended; target's stays."""
    target = extract(parse_fbx(_TARGET))
    donor = extract(parse_fbx(_DONOR))
    plan = merge_materials(donor=donor, target=target)
    out = apply_edits(_TARGET, _to_text_edits(plan))
    # Only one "Material::Cloth" in the result, the target's (lambert shading
    # from donor must NOT appear).
    assert out.count(b'"Material::Cloth"') == 1
    assert b"lambert" not in out


def test_merge_materials_appends_novel_donor_material():
    target = extract(parse_fbx(_TARGET))
    donor = extract(parse_fbx(_DONOR))
    plan = merge_materials(donor=donor, target=target)
    out = apply_edits(_TARGET, _to_text_edits(plan))
    assert b'"Material::Hat"' in out


def test_merge_materials_keeps_existing_target_materials():
    target = extract(parse_fbx(_TARGET))
    donor = extract(parse_fbx(_DONOR))
    plan = merge_materials(donor=donor, target=target)
    out = apply_edits(_TARGET, _to_text_edits(plan))
    assert b'"Material::Skin"' in out
    assert b'"Material::Cloth"' in out


def test_merge_materials_inserts_inside_objects_section():
    """Novel donor material is spliced into target's Objects { ... }, not
    appended to the file."""
    target_doc = parse_fbx(_TARGET)
    target = extract(target_doc)
    donor = extract(parse_fbx(_DONOR))
    plan = merge_materials(donor=donor, target=target)
    out = apply_edits(_TARGET, _to_text_edits(plan))
    # The Hat material must appear before the Objects section's closing brace
    # and before the Connections section opens.
    hat_idx = out.find(b'"Material::Hat"')
    conn_idx = out.find(b"Connections:")
    assert hat_idx != -1 and conn_idx != -1
    assert hat_idx < conn_idx


# --- blendshape channel merge -----------------------------------------------


def test_merge_blendshapes_target_wins_on_channel_name_collision():
    target = extract(parse_fbx(_TARGET))
    donor = extract(parse_fbx(_DONOR))
    plan = merge_blendshapes(donor=donor, target=target)
    out = apply_edits(_TARGET, _to_text_edits(plan))
    # `Smile` was in both; donor's must NOT have been spliced.
    assert out.count(b'"SubDeformer::Smile"') == 1


def test_merge_blendshapes_appends_novel_channel():
    target = extract(parse_fbx(_TARGET))
    donor = extract(parse_fbx(_DONOR))
    plan = merge_blendshapes(donor=donor, target=target)
    out = apply_edits(_TARGET, _to_text_edits(plan))
    assert b'"SubDeformer::Wink"' in out


def test_merge_blendshapes_brings_owning_blendshape_for_novel_channels():
    """If donor has a novel channel, the BlendShape Deformer that owns it must
    come along too — otherwise the channel has no parent. (Names dedup against
    target's existing BlendShape deformers if the same name exists.)"""
    target = extract(parse_fbx(_TARGET))
    donor = extract(parse_fbx(_DONOR))
    plan = merge_blendshapes(donor=donor, target=target)
    out = apply_edits(_TARGET, _to_text_edits(plan))
    # Donor's BlendShape owner ("DonorBlend") must appear because it owns Wink.
    assert b'"Deformer::DonorBlend"' in out


def test_merge_blendshapes_inserts_inside_objects_section():
    target_doc = parse_fbx(_TARGET)
    target = extract(target_doc)
    donor = extract(parse_fbx(_DONOR))
    plan = merge_blendshapes(donor=donor, target=target)
    out = apply_edits(_TARGET, _to_text_edits(plan))
    wink_idx = out.find(b'"SubDeformer::Wink"')
    conn_idx = out.find(b"Connections:")
    assert wink_idx != -1 and conn_idx != -1
    assert wink_idx < conn_idx


# --- composability: material + blendshape plans together ---------------------


def test_material_and_blendshape_plans_compose_without_overlap():
    """Both plans target the same Objects-closing-brace splice point, but the
    edit splice points are insertions (start == end) so they must be allowed
    together. (apply_edits sorts by start; two insertions at the same offset
    are non-overlapping.)"""
    target_doc = parse_fbx(_TARGET)
    target = extract(target_doc)
    donor = extract(parse_fbx(_DONOR))
    mat_plan = merge_materials(donor=donor, target=target)
    bs_plan = merge_blendshapes(donor=donor, target=target)
    combined = _to_text_edits(mat_plan) + _to_text_edits(bs_plan)
    out = apply_edits(_TARGET, combined)
    assert b'"Material::Hat"' in out
    assert b'"SubDeformer::Wink"' in out


# --- no donor content --------------------------------------------------------


_EMPTY_DONOR = b"""\
Objects:  {
\tModel: 1, "Model::Hips", "LimbNode" {
\t}
}
Connections:  {
\tC: "OO",1,0
}
"""


def test_merge_materials_with_empty_donor_is_noop():
    target = extract(parse_fbx(_TARGET))
    donor = extract(parse_fbx(_EMPTY_DONOR))
    plan = merge_materials(donor=donor, target=target)
    assert plan.edits == []


def test_merge_blendshapes_with_empty_donor_is_noop():
    target = extract(parse_fbx(_TARGET))
    donor = extract(parse_fbx(_EMPTY_DONOR))
    plan = merge_blendshapes(donor=donor, target=target)
    assert plan.edits == []


# --- novel donor BlendShape with channels (no collisions at all) -------------


_TARGET_NO_BS = b"""\
Objects:  {
\tModel: 1, "Model::Hips", "LimbNode" {
\t}
}
Connections:  {
\tC: "OO",1,0
}
"""


def test_merge_blendshapes_into_target_with_none():
    """Target has zero BlendShapes; donor's must come along whole."""
    target = extract(parse_fbx(_TARGET_NO_BS))
    donor = extract(parse_fbx(_DONOR))
    plan = merge_blendshapes(donor=donor, target=target)
    out = apply_edits(_TARGET_NO_BS, _to_text_edits(plan))
    assert b'"Deformer::DonorBlend"' in out
    assert b'"SubDeformer::Smile"' in out
    assert b'"SubDeformer::Wink"' in out
