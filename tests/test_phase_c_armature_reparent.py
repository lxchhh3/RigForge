"""Phase C armature-reparent fix (the 90-degree rotation).

When the clothing's `Armature` Null is stripped, its surviving children must be
reparented under the TARGET's armature Null (which carries the same -90 Y-up/Z-up
rotation) — NOT the identity scene root, which drops the rotation and tilts the
clothing 90 degrees relative to the avatar.
"""
from __future__ import annotations

from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.pipeline.phase_c import _find_target_armature_null


def _fbx(nulls: list[tuple[int, str, str]], limbs: list[tuple[int, str, int]]) -> bytes:
    """nulls: (id, name, 'rx,ry,rz'); limbs: (id, name, parent_id)."""
    models = []
    conns = []
    for nid, name, rot in nulls:
        models.append(
            f'\tModel: {nid}, "Model::{name}", "Null" {{\n'
            f'\t\tProperties70:  {{\n'
            f'\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",{rot}\n'
            f'\t\t}}\n\t}}'
        )
        conns.append(f'\tC: "OO",{nid},0')
    for lid, name, pid in limbs:
        models.append(f'\tModel: {lid}, "Model::{name}", "LimbNode" {{\n\t\tProperties70:  {{\n\t\t}}\n\t}}')
        conns.append(f'\tC: "OO",{lid},{pid}')
    body = (
        "; FBX 7.4.0 project file\n"
        "Objects:  {\n" + "\n".join(models) + "\n}\n"
        "Connections:  {\n" + "\n".join(conns) + "\n}\n"
    )
    return body.encode("utf-8")


def test_finds_rotated_armature_null():
    view = extract(parse(_fbx(
        nulls=[(1, "Armature", "-90.0000093346673,0,0")],
        limbs=[(2, "Hips", 1)],
    )))
    assert _find_target_armature_null(view) == 1


def test_prefers_rotated_null_over_identity():
    view = extract(parse(_fbx(
        nulls=[(1, "Origin", "0,0,0"), (2, "Armature", "-90,0,0")],
        limbs=[(3, "Hips", 2)],
    )))
    assert _find_target_armature_null(view) == 2


def test_none_when_no_top_level_null():
    # Only a skeleton at scene root, no armature Null.
    view = extract(parse(_fbx(nulls=[], limbs=[(1, "Hips", 0)])))
    assert _find_target_armature_null(view) is None
