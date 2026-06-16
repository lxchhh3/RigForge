"""Tests for the orientation monitor (rigforge.diagnostics.orientation).

Models the 6/16 "clothing rotated 90 degrees" bug: clothing mesh Models carry
Lcl Rotation (-90,0,0) while the avatar's are identity. The monitor must flag
that mismatch both within one merged file and across a clothing/target pair.
"""
from __future__ import annotations

from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.diagnostics.orientation import (
    compare_conventions,
    dominant_rotation,
    find_mixed_orientation,
    find_root_rotation_split,
)


def _fbx(*mesh_rots: tuple[str, int, str]) -> bytes:
    """Build a minimal FBX with Mesh Models carrying the given Lcl Rotations.
    Each arg is (name, model_id, 'rx,ry,rz')."""
    models = []
    conns = []
    for name, mid, rot in mesh_rots:
        models.append(
            f'\tModel: {mid}, "Model::{name}", "Mesh" {{\n'
            f'\t\tProperties70:  {{\n'
            f'\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",{rot}\n'
            f'\t\t}}\n'
            f'\t}}'
        )
        conns.append(f'\tC: "OO",{mid},0')
    body = (
        "; FBX 7.4.0 project file\n"
        "Objects:  {\n" + "\n".join(models) + "\n}\n"
        "Connections:  {\n" + "\n".join(conns) + "\n}\n"
    )
    return body.encode("utf-8")


# Noisy -90 like real exporters emit — must snap to -90.
_M90 = "-90.0000093346673,0,0"
_ID = "0,0,0"


def test_consistent_orientation_is_not_flagged():
    view = extract(parse(_fbx(("Body", 200, _ID), ("Hair", 250, _ID))))
    assert find_mixed_orientation(view) is None


def test_mixed_orientation_is_flagged():
    view = extract(parse(_fbx(("Body", 200, _ID), ("Cape", 250, _M90))))
    mixed = find_mixed_orientation(view)
    assert mixed is not None
    # Two distinct conventions: identity and -90 X.
    assert (0.0, 0.0, 0.0) in mixed.groups
    assert (-90.0, 0.0, 0.0) in mixed.groups
    assert mixed.groups[(-90.0, 0.0, 0.0)] == ["Cape"]
    assert "rotation convention" in mixed.message()


def test_dominant_rotation_picks_the_majority():
    view = extract(parse(_fbx(
        ("A", 1, _M90), ("B", 2, _M90), ("C", 3, _ID),
    )))
    assert dominant_rotation(view) == (-90.0, 0.0, 0.0)


def test_compare_conventions_flags_clothing_vs_target_mismatch():
    clothing = extract(parse(_fbx(("Tops", 1, _M90), ("Skirt", 2, _M90))))
    target = extract(parse(_fbx(("Body", 10, _ID), ("Hair", 11, _ID))))
    warn = compare_conventions(clothing, target)
    assert warn is not None
    assert "clothing=(-90.0, 0.0, 0.0)" in warn
    assert "target=(0.0, 0.0, 0.0)" in warn


def test_compare_conventions_clean_when_aligned():
    clothing = extract(parse(_fbx(("Tops", 1, _ID))))
    target = extract(parse(_fbx(("Body", 10, _ID))))
    assert compare_conventions(clothing, target) is None


# --- armature-split check (the actual 6/16 cause) --------------------------

def _rig_fbx(armature_rot: str, cloth_root_under_armature: bool) -> bytes:
    """A merged-style rig: a rotated Armature Null with maya's root under it,
    plus a clothing root bone either under the armature (clean) or bypassing it
    to the scene root (the bug)."""
    cloth_parent = 1 if cloth_root_under_armature else 0
    body = (
        "; FBX 7.4.0 project file\n"
        "Objects:  {\n"
        f'\tModel: 1, "Model::Armature", "Null" {{\n'
        f'\t\tProperties70:  {{\n'
        f'\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",{armature_rot}\n'
        f'\t\t}}\n\t}}\n'
        '\tModel: 2, "Model::MayaHips", "LimbNode" {\n\t\tProperties70:  {\n\t\t}\n\t}\n'
        '\tModel: 3, "Model::ClothHips", "LimbNode" {\n\t\tProperties70:  {\n\t\t}\n\t}\n'
        "}\n"
        "Connections:  {\n"
        '\tC: "OO",1,0\n'      # Armature -> scene root
        '\tC: "OO",2,1\n'      # MayaHips -> Armature
        f'\tC: "OO",3,{cloth_parent}\n'  # ClothHips -> Armature or scene root
        "}\n"
    )
    return body.encode("utf-8")


def test_armature_split_is_flagged_when_root_bypasses_rotated_armature():
    view = extract(parse(_rig_fbx(_M90, cloth_root_under_armature=False)))
    split = find_root_rotation_split(view)
    assert split is not None
    names = [n for n, _ in split.bypass_roots]
    assert "ClothHips" in names
    assert "armature" in split.message().lower()


def test_no_split_when_all_roots_share_the_armature():
    view = extract(parse(_rig_fbx(_M90, cloth_root_under_armature=True)))
    assert find_root_rotation_split(view) is None


def test_no_split_when_armature_is_identity():
    # An identity armature carries no rotation to lose, so a bypassing root is fine.
    view = extract(parse(_rig_fbx(_ID, cloth_root_under_armature=False)))
    assert find_root_rotation_split(view) is None
