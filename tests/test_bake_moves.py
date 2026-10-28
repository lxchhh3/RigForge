"""Baking a move on a bone the merge replaces (kept bone) into the clothing's
vertices — kevin's pick 2026-10-28 after the bind-rewrite test moved Maya's own
Head rest in Blender. Each vertex moves by its weight on the moved bone (and on
kept bones riding it), as if moved in edit mode; ride-along child bones without
their own move follow the bone."""
from __future__ import annotations

import pytest

from rigforge.ascii_fbx.edits import apply_edits
from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract, model_local_transform
from rigforge.ascii_fbx.transform import LocalTransform
from rigforge.pipeline.bake_moves import bake_kept_bone_moves, geometry_array
from tests.fbx_fixtures import skinned_fbx

HIPS, HEAD, EYE, RIBBON, ACC = 1, 2, 3, 4, 10
BONES = [
    (HIPS, "Hips", 0, (0, 1.0, 0), (0, 0, 0)),
    (HEAD, "Head", HIPS, (0, 0.5, 0), (0, 0, 0)),
    (EYE, "Eye", HEAD, (0.03, 0.05, 0), (0, 0, 0)),
    (RIBBON, "Ribbon", HEAD, (0, 0.1, 0), (0, 0, 0)),
]
WORLDS = {HIPS: (0, 1.0, 0), HEAD: (0, 1.5, 0), EYE: (0.03, 1.55, 0), RIBBON: (0, 1.6, 0)}
KEPT = {HIPS, HEAD, EYE}           # name-matched: the merge uses the avatar's copies
UP7 = {HEAD: LocalTransform(position=(0.0, 0.57, 0.0))}


def _fixture(*, mesh_rot=(0, 0, 0), local=None, **kw):
    verts = local or [(0, 1.6, 0), (0, 1.6, 0.1), (0, 1.0, 0), (0.03, 1.55, 0)]
    return skinned_fbx(
        BONES, (ACC, "Acc", (0, 0, 0), mesh_rot), verts, [(0, 1, 2), (0, 2, 3)],
        {HEAD: [(0, 1.0), (1, 0.5)], RIBBON: [(1, 0.5)], HIPS: [(2, 1.0)], EYE: [(3, 1.0)]},
        normals=[(1, 0, 0)] * 6, shape=("Smile", [0], [(0.01, 0, 0)]),
        bone_worlds=WORLDS, **kw,
    )


def _bake(raw: bytes, moves, **kw):
    view = extract(parse(raw))
    notes: list[str] = []
    res = bake_kept_bone_moves(view, moves, KEPT, notes=notes, **kw)
    out = apply_edits(raw, res.edits)
    return extract(parse(out)), res, notes


def _verts(view, name="Acc"):
    flat = geometry_array(view, name, "Vertices")
    return [tuple(flat[i:i + 3]) for i in range(0, len(flat), 3)]


def test_vertices_follow_the_moved_bone_by_their_weight():
    view, res, _ = _bake(_fixture(), UP7)
    v = _verts(view)
    assert v[0] == pytest.approx((0, 1.67, 0))           # all on Head
    assert v[1] == pytest.approx((0, 1.635, 0.1))        # half Head, half Ribbon
    assert v[2] == pytest.approx((0, 1.0, 0))            # on Hips: untouched
    assert res.baked_bone_ids == {HEAD}


def test_kept_bones_riding_the_moved_one_carry_it_too():
    """Eye is the avatar's own bone after the merge, but in Unity the outfit's
    Eye rode on the moved Head: its vertices move with it."""
    view, _, _ = _bake(_fixture(), UP7)
    assert _verts(view)[3] == pytest.approx((0.03, 1.62, 0))


def test_a_ride_along_child_without_its_own_move_follows():
    view, _, notes = _bake(_fixture(), UP7)
    rib = next(b for b in view.bones.values() if b.name == "Ribbon")
    assert model_local_transform(view, rib.model_id).position == pytest.approx((0, 0.17, 0))
    assert any("Head" in n and "bake" in n for n in notes)


def test_a_ride_along_child_with_its_own_move_is_left_to_that_move():
    view, _, _ = _bake(_fixture(), UP7, explicit_move_ids={RIBBON})
    rib = next(b for b in view.bones.values() if b.name == "Ribbon")
    assert model_local_transform(view, rib.model_id).position == pytest.approx((0, 0.1, 0))


def test_a_rotated_mesh_node_is_baked_in_its_own_frame():
    """Blender exports put meshes at -90 X: a world +7 cm up is +7 cm local Z."""
    local = [(0, 0, 1.6), (0, -0.1, 1.6), (0, 0, 1.0), (0.03, 0, 1.55)]
    view, _, _ = _bake(_fixture(mesh_rot=(-90, 0, 0), local=local), UP7)
    v = _verts(view)
    assert v[0] == pytest.approx((0, 0, 1.67), abs=1e-9)
    assert v[2] == pytest.approx((0, 0, 1.0), abs=1e-9)


def test_a_rotation_turns_positions_normals_and_shape_deltas():
    raw = skinned_fbx(
        BONES, (ACC, "Acc", (0, 0, 0), (0, 0, 0)),
        [(0.1, 1.5, 0), (0, 1.0, 0), (0, 1.0, 0.1)], [(0, 1, 2)],
        {HEAD: [(0, 1.0)], HIPS: [(1, 1.0), (2, 1.0)]},
        normals=[(1, 0, 0)] * 3, shape=("Smile", [0], [(0.01, 0, 0)]), bone_worlds=WORLDS,
    )
    turn = {HEAD: LocalTransform(rotation=(0.0, 90.0, 0.0))}
    view, _, _ = _bake(raw, turn)
    assert _verts(view)[0] == pytest.approx((0, 1.5, -0.1), abs=1e-9)
    n = geometry_array(view, "Acc", "Normals")
    assert n[0:3] == pytest.approx([0, 0, -1], abs=1e-9)      # corner on v0 turns
    assert n[3:6] == pytest.approx([1, 0, 0], abs=1e-9)       # corner on Hips does not
    assert geometry_array(view, "Smile", "Vertices") == pytest.approx([0, 0, -0.01], abs=1e-9)


def test_a_translation_leaves_normals_and_shapes_alone():
    view, res, _ = _bake(_fixture(), UP7)
    assert geometry_array(view, "Acc", "Normals") == pytest.approx([1, 0, 0] * 6)
    assert geometry_array(view, "Smile", "Vertices") == pytest.approx([0.01, 0, 0])


def test_a_dropped_mesh_is_not_touched():
    raw = _fixture()
    view, res, _ = _bake(raw, UP7, skip_mesh_ids={ACC})
    assert _verts(view)[0] == pytest.approx((0, 1.6, 0))


def test_moves_on_bones_that_are_not_kept_bake_nothing():
    view, res, _ = _bake(_fixture(), {RIBBON: LocalTransform(position=(0, 0.2, 0))})
    assert res.edits == [] and res.baked_bone_ids == set()


def test_unweighted_meshes_get_no_edit():
    raw = _fixture(extra_meshes=[(50, "Plain")])
    view, res, _ = _bake(raw, UP7)
    assert _verts(view, "Plain") == [(0, 0, 0), (1, 0, 0), (0, 1, 0)]


def test_float_noise_in_a_rotation_does_not_turn_normals_or_shapes():
    """Unity's 7-digit numbers leave ~1e-7 of rotation in a translation-only
    move (Eclat_Coffee's Head): that must not rewrite normals or shapes."""
    noisy = {HEAD: LocalTransform(position=(0.0, 0.57, 0.0), rotation=(2e-6, -1e-6, 0.0))}
    view, res, notes = _bake(_fixture(), noisy)
    assert geometry_array(view, "Acc", "Normals") == pytest.approx([1, 0, 0] * 6)
    assert geometry_array(view, "Smile", "Vertices") == [0.01, 0.0, 0.0]
    assert not any("normals" in n for n in notes)
