"""Expanding a Unity scene's prefab instances into one object graph."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from rigforge.unity.hierarchy import UnityProject, compose_id, expand
from tests.unity_fixtures import (
    HEADER,
    MODEL_ROOT_GO,
    MODEL_ROOT_TRANSFORM,
    game_object,
    make_project,
    mono,
    prefab_instance,
    skinned_mesh,
    stripped,
    transform,
    write_asset,
)

OUTFIT_GUID = "0" * 31 + "1"
NESTED_GUID = "0" * 31 + "2"
FBX_GUID = "0" * 31 + "3"
SCRIPT_GUID = "a162bb8ec7e24a5abcf457887f1df3fa"


def _outfit_prefab() -> str:
    # Outfit (root) -> Skirt (mesh, 3 blendshapes), Hips (bone)
    return HEADER + "".join([
        game_object(10, "Outfit", [11]),
        transform(11, 10, 0, children=[21, 31]),
        game_object(20, "Skirt", [21, 22]),
        transform(21, 20, 11),
        skinned_mesh(22, 20, [0, 0, 0]),
        game_object(30, "Hips", [31]),
        transform(31, 30, 11, pos=(0, 1, 0)),
    ])


@pytest.fixture
def project(tmp_path: Path) -> Path:
    proj = make_project(tmp_path)
    write_asset(proj, "Assets/Outfit.prefab", _outfit_prefab(), OUTFIT_GUID)
    (proj / "Assets" / "Scripts").mkdir()
    write_asset(proj, "Assets/Scripts/ModularAvatarObjectToggle.cs", "// cs", SCRIPT_GUID)
    return proj


def _scene(*parts: str) -> str:
    return HEADER + "".join(parts)


def test_compose_id_matches_unitys_nested_rule():
    # From kevin's Eclat_Coffee: the Toast prefab's instance 1822428105269825218
    # of its FBX puts the FBX root Transform at 2215816963118508329.
    assert compose_id(1822428105269825218, MODEL_ROOT_TRANSFORM) == 2215816963118508329


def test_prefab_instance_applies_overrides_and_parents_under_the_scene(project: Path):
    scene = write_asset(project, "Assets/S.unity", _scene(
        game_object(1, "Avatar", [2]),
        transform(2, 1),
        prefab_instance(500, OUTFIT_GUID, parent=2, mods=[
            (22, "m_BlendShapeWeights.Array.data[1]", "84"),
            (31, "m_LocalPosition.y", "1.07"),
            (10, "m_Name", "My Outfit"),
        ]),
    ), "f" * 32)
    g = expand(UnityProject(project), scene)

    skirt = g.find_go("Skirt")
    smr = g.component(skirt, "SkinnedMeshRenderer")
    assert smr.data["m_BlendShapeWeights"] == ["0", "84", "0"]
    hips = g.transform_of(g.find_go("Hips"))
    assert g.local_position(hips) == (0.0, 1.07, 0.0)
    outfit = g.find_go("My Outfit")
    assert g.parent(g.transform_of(outfit)) == 2
    assert g.path(skirt) == "Avatar/My Outfit/Skirt"
    # ids are composed into the scene's space
    assert skirt == compose_id(500, 20)


def test_added_component_on_a_stripped_game_object_lands_on_it(project: Path):
    scene = write_asset(project, "Assets/S.unity", _scene(
        prefab_instance(500, OUTFIT_GUID, added_components=[(10, 700)]),
        stripped(1, "GameObject", 600, 10, OUTFIT_GUID, 500),
        mono(700, 600, SCRIPT_GUID, "  m_inverted: 0\n  m_objects: []\n"),
    ), "f" * 32)
    proj = UnityProject(project)
    g = expand(proj, scene)
    outfit = g.find_go("Outfit")
    toggles = [c for c in g.components(outfit) if proj.script_name(c) == "ModularAvatarObjectToggle"]
    assert len(toggles) == 1


def test_removed_game_object_is_gone(project: Path):
    scene = write_asset(project, "Assets/S.unity", _scene(
        prefab_instance(500, OUTFIT_GUID, removed_gos=[30]),
    ), "f" * 32)
    g = expand(UnityProject(project), scene)
    assert g.find_go("Hips", missing_ok=True) is None


def test_nested_prefab_override_from_the_scene_reaches_the_inner_object(project: Path):
    # Nested.prefab = Wrapper root + an instance (id 900) of Outfit.prefab.
    write_asset(project, "Assets/Nested.prefab", HEADER + "".join([
        game_object(40, "Wrapper", [41]),
        transform(41, 40, 0, children=[compose_id(900, 11)]),
        prefab_instance(900, OUTFIT_GUID, parent=41, mods=[
            (22, "m_BlendShapeWeights.Array.data[0]", "10"),
        ]),
    ]), NESTED_GUID)
    inner_smr = compose_id(900, 22)
    scene = write_asset(project, "Assets/S.unity", _scene(
        prefab_instance(501, NESTED_GUID, mods=[
            (inner_smr, "m_BlendShapeWeights.Array.data[2]", "55"),
        ]),
    ), "f" * 32)
    g = expand(UnityProject(project), scene)
    smr = g.component(g.find_go("Skirt"), "SkinnedMeshRenderer")
    assert smr.data["m_BlendShapeWeights"] == ["10", "0", "55"]
    assert g.path(g.find_go("Skirt")) == "Wrapper/Outfit/Skirt"


def test_model_instance_root_is_known_and_inner_overrides_are_reported(project: Path):
    fbx = project / "Assets" / "Pin.fbx"
    fbx.write_bytes(b"; FBX 7.4.0 project file\n")
    Path(str(fbx) + ".meta").write_text(
        f"fileFormatVersion: 2\nguid: {FBX_GUID}\nModelImporter:\n", encoding="utf-8")
    scene = write_asset(project, "Assets/S.unity", _scene(
        game_object(1, "Avatar", [2]),
        transform(2, 1),
        prefab_instance(502, FBX_GUID, parent=2, mods=[
            (MODEL_ROOT_TRANSFORM, "m_LocalPosition.x", "0.03"),
            (MODEL_ROOT_GO, "m_Name", "HairPin"),
            (1234567, "m_LocalPosition.z", "0.004"),
        ]),
    ), "f" * 32)
    g = expand(UnityProject(project), scene)
    root = g.find_go("HairPin")
    assert g.model_roots[root] == fbx
    assert g.local_position(g.transform_of(root)) == (0.03, 0.0, 0.0)
    (io,) = g.inner_overrides
    assert io.candidates == [(root, fbx, 1234567)] and io.prop == "m_LocalPosition.z"


def test_world_matrix_composes_parent_rotation(project: Path):
    s = math.sqrt(0.5)
    scene = write_asset(project, "Assets/S.unity", _scene(
        game_object(1, "Arm", [2]),
        transform(2, 1, rot=(s, 0, 0, s), children=[4]),      # +90 deg about X
        game_object(3, "Hand", [4]),
        transform(4, 3, 2, pos=(0, 1, 0)),
    ), "f" * 32)
    g = expand(UnityProject(project), scene)
    w = g.world_matrix(4)
    assert [round(w[i][3], 6) for i in range(3)] == [0.0, 0.0, 1.0]


def test_project_root_is_found_from_a_scene_path(project: Path):
    scene = write_asset(project, "Assets/Sub/S.unity", _scene(game_object(1, "A", [])), "f" * 32)
    assert UnityProject.for_asset(scene).root == project


def test_project_lookup_outside_a_unity_project_fails(tmp_path: Path):
    p = tmp_path / "loose.unity"
    p.write_text(HEADER, encoding="utf-8")
    with pytest.raises(ValueError):
        UnityProject.for_asset(p)


def test_overrides_on_a_models_inner_objects_are_kept_for_the_reader(project: Path):
    """kevin's Toast: the Toast prefab instances its FBX (id 900); the prefab
    sets a weight on an inner renderer, the scene moves an inner transform.
    Both are kept, in Unity's order (prefab first), with the inner fileID
    and the model instance's root in the scene's space."""
    from rigforge.unity.hierarchy import MASK
    fbx = project / "Assets" / "Toast.fbx"
    fbx.write_bytes(b"; FBX 7.4.0 project file\n")
    Path(str(fbx) + ".meta").write_text(
        f"fileFormatVersion: 2\nguid: {FBX_GUID}\nModelImporter:\n", encoding="utf-8")
    inner_smr, inner_t = 5120338455441729675, -7185857084041579234
    write_asset(project, "Assets/Toast.prefab", HEADER + prefab_instance(
        900, FBX_GUID, mods=[(inner_smr, "m_BlendShapeWeights.Array.data[0]", "30")],
    ), NESTED_GUID)
    scene = write_asset(project, "Assets/S.unity", _scene(
        prefab_instance(501, NESTED_GUID, mods=[
            (compose_id(900, inner_smr), "m_BlendShapeWeights.Array.data[0]", "100"),
            (compose_id(900, inner_t), "m_LocalPosition.y", "-0.0526"),
        ]),
    ), "f" * 32)
    g = expand(UnityProject(project), scene)
    root = compose_id(501, compose_id(900, MODEL_ROOT_GO))
    assert g.model_roots[root] == fbx
    got = [(o.candidates, o.prop, o.value) for o in g.inner_overrides]
    assert got == [
        ([(root, fbx, inner_smr & MASK)], "m_BlendShapeWeights.Array.data[0]", "30"),
        ([(root, fbx, inner_smr & MASK)], "m_BlendShapeWeights.Array.data[0]", "100"),
        ([(root, fbx, inner_t & MASK)], "m_LocalPosition.y", "-0.0526"),
    ]
