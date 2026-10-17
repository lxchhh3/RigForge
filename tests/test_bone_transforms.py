"""Tests for rigforge.pipeline.bone_transforms — the request-side model of a
bone move (id-or-name + partial transform + frame), its CLI spec parser, and
the resolver that turns a list of them into `{model_id: LocalTransform}` in
the FBX frame against a SectionView."""
from __future__ import annotations

import pytest

from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.ascii_fbx.transform import LocalTransform
from rigforge.pipeline.bone_transforms import (
    BoneTransformEdit,
    BoneTransformError,
    parse_bone_transform_spec,
    resolve_bone_transform_edits,
)
from tests.test_edits import MINI_FBX


# --- CLI spec parsing ---------------------------------------------------------


def test_parse_spec_name_with_all_components():
    e = parse_bone_transform_spec("Hat Root;pos=0,0.05,0;rot=0,0,10;scale=1,1,1.5")
    assert e.bone == "Hat Root"
    assert e.transform == LocalTransform(
        position=(0.0, 0.05, 0.0), rotation=(0.0, 0.0, 10.0), scale=(1.0, 1.0, 1.5),
    )
    assert e.frame == "unity"   # CLI default matches the FE: Unity Inspector numbers


def test_parse_spec_partial_components_and_frame():
    e = parse_bone_transform_spec("E#5_Hat.Root;pos=0,0.02,0", frame="fbx")
    assert e.bone == "E#5_Hat.Root"
    assert e.transform == LocalTransform(position=(0.0, 0.02, 0.0))
    assert e.frame == "fbx"


def test_parse_spec_hash_prefix_means_model_id():
    e = parse_bone_transform_spec("#2360967979536;scale=2,2,2")
    assert e.bone == 2360967979536
    assert e.transform.scale == (2.0, 2.0, 2.0)


def test_parse_spec_tolerates_spaces_around_separators():
    e = parse_bone_transform_spec(" Hips ; pos = 1, 2, 3 ; rot=4,5,6 ")
    assert e.bone == "Hips"
    assert e.transform.position == (1.0, 2.0, 3.0)
    assert e.transform.rotation == (4.0, 5.0, 6.0)


@pytest.mark.parametrize("bad", [
    "Hips",                      # no components at all
    "Hips;pos=1,2",              # wrong arity
    "Hips;pos=a,b,c",            # not numbers
    "Hips;colour=1,2,3",         # unknown key
    ";pos=1,2,3",                # empty bone
    "Hips;pos=1,2,3;pos=4,5,6",  # duplicate key
])
def test_parse_spec_rejects_malformed(bad):
    with pytest.raises(BoneTransformError):
        parse_bone_transform_spec(bad)


def test_parse_spec_rejects_unknown_frame():
    with pytest.raises(BoneTransformError):
        parse_bone_transform_spec("Hips;pos=1,2,3", frame="blender")


def test_edit_to_dict_round_trips():
    e = BoneTransformEdit(
        bone="Hat Root", transform=LocalTransform(position=(0, 0.05, 0)), frame="unity",
    )
    d = e.to_dict()
    assert d == {
        "bone": "Hat Root", "frame": "unity",
        "position": [0.0, 0.05, 0.0], "rotation": None, "scale": None,
    }
    assert BoneTransformEdit.from_dict(d) == e


# --- resolution against a view -------------------------------------------------


def _view():
    return extract(parse(MINI_FBX))   # Hips=100, Spine=101


def test_resolve_by_id_and_by_name():
    notes: list[str] = []
    out = resolve_bone_transform_edits(_view(), [
        BoneTransformEdit(bone="Spine", transform=LocalTransform(position=(0, 1, 0)), frame="fbx"),
        BoneTransformEdit(bone=100, transform=LocalTransform(scale=(2, 2, 2)), frame="fbx"),
    ], notes=notes, side="clothing")
    assert out == {
        101: LocalTransform(position=(0.0, 1.0, 0.0)),
        100: LocalTransform(scale=(2.0, 2.0, 2.0)),
    }
    assert notes == []


def test_resolve_converts_unity_frame_to_fbx():
    """The default frame is what Unity's Inspector shows; the resolver hands
    Phase C FBX-frame values (X negated, Y/Z rotations flipped)."""
    out = resolve_bone_transform_edits(_view(), [
        BoneTransformEdit(
            bone="Hips",
            transform=LocalTransform(position=(0.1, 0.2, 0.3), rotation=(0, 90, 0)),
            frame="unity",
        ),
    ], notes=[], side="clothing")
    t = out[100]
    assert t.position == pytest.approx((-0.1, 0.2, 0.3))
    assert t.rotation == pytest.approx((0.0, -90.0, 0.0))


def test_resolve_unknown_name_and_id_are_noted_not_fatal():
    notes: list[str] = []
    out = resolve_bone_transform_edits(_view(), [
        BoneTransformEdit(bone="NoSuchBone", transform=LocalTransform(position=(0, 0, 0))),
        BoneTransformEdit(bone=99999, transform=LocalTransform(position=(0, 0, 0))),
    ], notes=notes, side="target")
    assert out == {}
    assert len(notes) == 2 and all(n.startswith("skip: bone_transform") for n in notes)
    assert "NoSuchBone" in notes[0] and "99999" in notes[1]


def test_resolve_ambiguous_name_raises():
    dup = MINI_FBX.replace(b'"Model::Spine"', b'"Model::Hips"')   # two Models named Hips
    view = extract(parse(dup))
    with pytest.raises(BoneTransformError) as exc:
        resolve_bone_transform_edits(view, [
            BoneTransformEdit(bone="Hips", transform=LocalTransform(position=(0, 0, 0))),
        ], notes=[], side="clothing")
    assert "ambiguous" in str(exc.value)


def test_resolve_merges_components_for_the_same_bone():
    """Two entries for one bone (e.g. position from one FE row, scale from a
    CLI flag) merge component-wise; a later entry wins on conflicts."""
    out = resolve_bone_transform_edits(_view(), [
        BoneTransformEdit(bone=100, transform=LocalTransform(position=(1, 1, 1)), frame="fbx"),
        BoneTransformEdit(bone="Hips", transform=LocalTransform(scale=(2, 2, 2)), frame="fbx"),
        BoneTransformEdit(bone=100, transform=LocalTransform(position=(3, 3, 3)), frame="fbx"),
    ], notes=[], side="clothing")
    assert out == {100: LocalTransform(position=(3.0, 3.0, 3.0), scale=(2.0, 2.0, 2.0))}


def test_resolve_skips_empty_transforms():
    notes: list[str] = []
    out = resolve_bone_transform_edits(_view(), [
        BoneTransformEdit(bone=100, transform=LocalTransform()),
    ], notes=notes, side="clothing")
    assert out == {}
