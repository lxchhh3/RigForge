"""Phase C dictionary-based morph + mesh name translation.

Exercises the `_translate_morph_and_mesh_edits` helper directly (the same code
path runs for the clothing side, always, and the target side, behind a flag).
Uses the shipped dictionary so the JP entries (笑い->Smile, 髪->Hair) are real.
"""
from __future__ import annotations

from rigforge.ascii_fbx.edits import apply_edits
from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.naming import load_translation_table
from rigforge.pipeline.phase_c import _translate_morph_and_mesh_edits


# Mesh 100 (name 髪 -> Hair) owns Geometry 300 (also 髪) and a BlendShape with
# two channels: 笑い (-> Smile) and an already-English "Big breasts" (no-op).
# Bone 101 is a LimbNode — this helper must NOT touch bones.
FBX = (
    "; FBX 7.4.0 project file\n"
    "Objects:  {\n"
    '\tModel: 100, "Model::髪", "Mesh" {\n'
    "\t}\n"
    '\tModel: 101, "Model::Hips", "LimbNode" {\n'
    "\t}\n"
    '\tGeometry: 300, "Geometry::髪", "Mesh" {\n'
    "\t}\n"
    '\tDeformer: 800, "Deformer::Blend", "BlendShape" {\n'
    "\t\tVersion: 100\n"
    "\t}\n"
    '\tDeformer: 900, "SubDeformer::笑い", "BlendShapeChannel" {\n'
    "\t\tVersion: 100\n"
    "\t\tDeformPercent: 0\n"
    "\t}\n"
    '\tDeformer: 901, "SubDeformer::Big breasts", "BlendShapeChannel" {\n'
    "\t\tVersion: 100\n"
    "\t\tDeformPercent: 0\n"
    "\t}\n"
    "}\n"
    "Connections:  {\n"
    '\tC: "OO",100,0\n'
    '\tC: "OO",101,0\n'
    '\tC: "OO",300,100\n'
    '\tC: "OO",800,300\n'
    '\tC: "OO",900,800\n'
    '\tC: "OO",901,800\n'
    "}\n"
).encode("utf-8")


def _apply(view, edits):
    return extract(parse(apply_edits(FBX, edits)))


def test_translates_jp_morph_and_mesh():
    view = extract(parse(FBX))
    notes: list[str] = []
    edits = _translate_morph_and_mesh_edits(
        view, load_translation_table(),
        skip_channel_ids=set(), skip_mesh_ids=set(), notes=notes, side="clothing",
    )
    out = _apply(view, edits)
    assert out.blend_shape_channels[900].name == "Smile"
    assert out.bones[100].name == "Hair"
    assert out.geometries[300].name == "Hair"


def test_already_english_morph_is_untouched():
    view = extract(parse(FBX))
    notes: list[str] = []
    edits = _translate_morph_and_mesh_edits(
        view, load_translation_table(),
        skip_channel_ids=set(), skip_mesh_ids=set(), notes=notes, side="clothing",
    )
    out = _apply(view, edits)
    assert out.blend_shape_channels[901].name == "Big breasts"


def test_does_not_touch_bones():
    view = extract(parse(FBX))
    notes: list[str] = []
    edits = _translate_morph_and_mesh_edits(
        view, load_translation_table(),
        skip_channel_ids=set(), skip_mesh_ids=set(), notes=notes, side="clothing",
    )
    out = _apply(view, edits)
    assert out.bones[101].name == "Hips"  # LimbNode untouched


def test_skips_dropped_channel():
    view = extract(parse(FBX))
    notes: list[str] = []
    edits = _translate_morph_and_mesh_edits(
        view, load_translation_table(),
        skip_channel_ids={900}, skip_mesh_ids=set(), notes=notes, side="clothing",
    )
    out = _apply(view, edits)
    # 900 is being dropped elsewhere -> not renamed (no overlapping edit).
    assert out.blend_shape_channels[900].name == "笑い"
    # The mesh still translates.
    assert out.bones[100].name == "Hair"


def test_skips_dropped_mesh():
    view = extract(parse(FBX))
    notes: list[str] = []
    edits = _translate_morph_and_mesh_edits(
        view, load_translation_table(),
        skip_channel_ids=set(), skip_mesh_ids={100}, notes=notes, side="clothing",
    )
    out = _apply(view, edits)
    assert out.bones[100].name == "髪"
    assert out.blend_shape_channels[900].name == "Smile"


def test_notes_record_counts():
    view = extract(parse(FBX))
    notes: list[str] = []
    _translate_morph_and_mesh_edits(
        view, load_translation_table(),
        skip_channel_ids=set(), skip_mesh_ids=set(), notes=notes, side="target",
    )
    assert any("translate (target)" in n for n in notes)


# Mesh with an ASCII Model name ("Body") but a Korean mesh-DATA name
# ("평면.050" = Blender's localized "Plane"). The real-world case: the object
# was renamed but the mesh data kept Blender's default. The Model must NOT
# translate (it's already English); the Geometry must.
MESH_DATA_FBX = (
    "; FBX 7.4.0 project file\n"
    "Objects:  {\n"
    '\tModel: 100, "Model::Body", "Mesh" {\n'
    "\t}\n"
    '\tGeometry: 300, "Geometry::평면.050", "Mesh" {\n'
    "\t}\n"
    "}\n"
    "Connections:  {\n"
    '\tC: "OO",100,0\n'
    '\tC: "OO",300,100\n'
    "}\n"
).encode("utf-8")


def test_translates_korean_mesh_data_name_keeps_ascii_model():
    view = extract(parse(MESH_DATA_FBX))
    notes: list[str] = []
    edits = _translate_morph_and_mesh_edits(
        view, load_translation_table(),
        skip_channel_ids=set(), skip_mesh_ids=set(), notes=notes, side="clothing",
    )
    out = extract(parse(apply_edits(MESH_DATA_FBX, edits)))
    assert out.bones[100].name == "Body"          # ASCII model untouched
    assert out.geometries[300].name == "Plane.050"  # mesh-data translated


def test_skipped_mesh_also_skips_its_geometry_data_name():
    view = extract(parse(MESH_DATA_FBX))
    notes: list[str] = []
    edits = _translate_morph_and_mesh_edits(
        view, load_translation_table(),
        skip_channel_ids=set(), skip_mesh_ids={100}, notes=notes, side="clothing",
    )
    out = extract(parse(apply_edits(MESH_DATA_FBX, edits)))
    # Mesh 100 is being dropped -> its geometry name is left alone too.
    assert out.geometries[300].name == "평면.050"
