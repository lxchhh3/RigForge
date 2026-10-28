"""Scene shapes found by running the reader over kevin's historical scenes
(2026-10-28): GG.unity (a prop prefab whose root IS the renderer, named after
the prefab), Chrome.unity (the vendor's Maya+outfit prefab: two FBXs' meshes
on one armature), Marette.unity (duplicate object names; FBX scales a hair
off 1.0), and MochiFitter_VCC scenes whose outfit prefabs were deleted."""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.avatars.fingerprint import fingerprint
from rigforge.avatars.registry import AvatarRegistry, CuratedAvatar
from rigforge.unity.scene_read import read_scene
from tests.unity_fixtures import (
    HEADER,
    fbx_ascii,
    game_object,
    make_project,
    prefab_instance,
    skinned_mesh,
    stripped,
    transform,
    write_asset,
    write_fbx,
)

AVA_FBX, AVA_PREFAB = "1" * 32, "2" * 32
OUT_FBX, PROP_FBX, PROP_PREFAB = "3" * 32, "4" * 32, "5" * 32
AVA_MODELS = [
    (1, "Armature", "Null", 0, (0, 0, 0), (-90, 0, 0)),
    (2, "Hips", "LimbNode", 1, (0, 0, 1.0), (90, 0, 0)),
    (3, "Head", "LimbNode", 2, (0, 0.5, 0), (0, 0, 0)),
    (4, "Body", "Mesh", 0, (0, 0, 0), (-90, 0, 0)),
]
# An outfit the vendor ships pre-merged on the avatar (Chrome_Maya_LilToon).
OUT_MODELS = [
    (10, "Armature", "Null", 0, (0, 0, 0), (-90, 0, 0)),
    (11, "Hips", "LimbNode", 10, (0, 0, 1.0), (90, 0, 0)),
    (12, "Jacket_Root", "LimbNode", 11, (0, 0.2, 0), (0, 0, 0)),
    (13, "Jacket", "Mesh", 0, (0, 0, 0), (-90, 0, 0)),
    (14, "Hood", "Mesh", 0, (0, 0, 0), (-90, 0, 0)),
]
PROP_MODELS = [(20, "Visor", "Mesh", 0, (0, 0, 0), (0, 0, 0))]


def _registry(ava_fbx: Path) -> AvatarRegistry:
    names = [m[1] for m in AVA_MODELS]
    return AvatarRegistry(avatars={"ava": CuratedAvatar(
        id="ava", display_name="Ava", fbx_path=ava_fbx, bone_names=names,
        fingerprint=fingerprint(names), canonical_to_name={"Hips": "Hips", "Head": "Head"},
        ascii_fbx_path=ava_fbx,
    )})


def _combined_prefab(*, hood_active: int = 1, jacket_active: int = 1) -> str:
    """Ava + the outfit's meshes on ONE armature that also has Jacket_Root."""
    return HEADER + "".join([
        game_object(100, "Ava", [101]),
        transform(101, 100, 0, children=[103, 109, 115, 119]),
        game_object(102, "Armature", [103]),
        transform(103, 102, 101, children=[105]),
        game_object(104, "Hips", [105]),
        transform(105, 104, 103, pos=(0, 1, 0), children=[107, 113]),
        game_object(106, "Head", [107]),
        transform(107, 106, 105, pos=(0, 0.5, 0)),
        game_object(112, "Jacket_Root", [113]),
        transform(113, 112, 105, pos=(0, 0.2, 0.03)),     # the vendor nudged it 3 cm
        game_object(108, "Body", [109, 110]),
        transform(109, 108, 101),
        skinned_mesh(110, 108, [], mesh_guid=AVA_FBX),
        game_object(114, "Jacket", [115, 116], active=jacket_active),
        transform(115, 114, 101),
        skinned_mesh(116, 114, [], mesh_guid=OUT_FBX),
        game_object(118, "Hood", [119, 120], active=hood_active),
        transform(119, 118, 101),
        skinned_mesh(120, 118, [], mesh_guid=OUT_FBX),
    ])


def _project(tmp_path: Path, scene_body: str, prefab: str):
    proj = make_project(tmp_path)
    ava = write_fbx(proj, "Assets/Ava.fbx", fbx_ascii(AVA_MODELS), AVA_FBX)
    write_fbx(proj, "Assets/Out.fbx", fbx_ascii(OUT_MODELS), OUT_FBX)
    write_asset(proj, "Assets/Ava.prefab", prefab, AVA_PREFAB)
    scene = write_asset(proj, "Assets/S.unity", HEADER + scene_body, "f" * 32)
    return scene, _registry(ava)


def _read(scene, reg):
    return read_scene(scene, registry=reg, load_view=lambda p: extract(parse(Path(p).read_bytes())))


def test_an_outfit_baked_into_the_avatar_prefab_is_its_own_clothing(tmp_path: Path):
    scene, reg = _project(tmp_path, prefab_instance(1000, AVA_PREFAB), _combined_prefab())
    res = _read(scene, reg)
    (c,) = res.clothings
    assert Path(c.clothing_path).name == "Out.fbx"
    # Its ride-along bone sits where the vendor put it (+3 cm).
    (mv,) = c.bone_transforms
    assert mv["bone_id"] == 12 and mv["position"] == pytest.approx([0.0, 0.2, 0.03], abs=1e-6)
    # The outfit's meshes are not "strays" of the target any more.
    assert not any("not meshes of this FBX" in n for n in res.notes)


def test_hidden_meshes_of_an_embedded_outfit_are_dropped(tmp_path: Path):
    scene, reg = _project(tmp_path, prefab_instance(1000, AVA_PREFAB), _combined_prefab(hood_active=0))
    (c,) = _read(scene, reg).clothings
    assert c.drop_mesh_ids == [14]


def test_a_garment_whose_meshes_are_all_hidden_is_skipped(tmp_path: Path):
    """Maya's broom (Houki) ships inactive: nothing of it would merge."""
    scene, reg = _project(tmp_path, prefab_instance(1000, AVA_PREFAB),
                          _combined_prefab(hood_active=0, jacket_active=0))
    res = _read(scene, reg)
    assert res.clothings == []
    (sk,) = res.skipped
    assert sk["name"] == "Out" and "hidden" in sk["reason"]


def test_an_instance_of_a_deleted_prefab_is_reported_by_its_name(tmp_path: Path):
    scene, reg = _project(tmp_path, prefab_instance(1000, AVA_PREFAB) + prefab_instance(
        2000, "9" * 32, parent=0, mods=[(77, "m_Name", "finalized_Meira_Manuka 2")],
    ), _combined_prefab())
    res = _read(scene, reg)
    sk = {s["name"]: s["reason"] for s in res.skipped}
    assert "missing from the Unity project" in sk["finalized_Meira_Manuka 2"]


def _prop_scene(tmp_path: Path):
    """GG.unity: the prop prefab's root object IS its renderer and is named
    after the prefab (GG_headset_Maya_3), not after the FBX's mesh (Visor)."""
    proj = make_project(tmp_path)
    ava = write_fbx(proj, "Assets/Ava.fbx", fbx_ascii(AVA_MODELS), AVA_FBX)
    write_fbx(proj, "Assets/Prop.fbx", fbx_ascii(PROP_MODELS), PROP_FBX)
    write_asset(proj, "Assets/Ava.prefab", HEADER + "".join([
        game_object(100, "Ava", [101]), transform(101, 100, 0, children=[103, 109]),
        game_object(102, "Armature", [103]), transform(103, 102, 101, children=[105]),
        game_object(104, "Hips", [105]), transform(105, 104, 103, pos=(0, 1, 0), children=[107]),
        game_object(106, "Head", [107]), transform(107, 106, 105, pos=(0, 0.5, 0)),
        game_object(108, "Body", [109, 110]), transform(109, 108, 101),
        skinned_mesh(110, 108, [], mesh_guid=AVA_FBX),
    ]), AVA_PREFAB)
    write_asset(proj, "Assets/Prop.prefab", HEADER + "".join([
        game_object(300, "GG_headset_Maya_3", [301, 302]),
        transform(301, 300, 0, pos=(0, -0.06, 1.08)),
        skinned_mesh(302, 300, [], mesh_guid=PROP_FBX),
    ]), PROP_PREFAB)
    scene = write_asset(proj, "Assets/S.unity", HEADER + "".join([
        prefab_instance(1000, AVA_PREFAB),
        stripped(4, "Transform", 1002, 103, AVA_PREFAB, 1000),
        prefab_instance(3000, PROP_PREFAB, parent=1002),
    ]), "f" * 32)
    return scene, _registry(ava)


def test_a_prop_renderer_named_after_its_prefab_binds_to_the_fbxs_only_mesh(tmp_path: Path):
    scene, reg = _prop_scene(tmp_path)
    (c,) = _read(scene, reg).clothings
    assert c.rigid is True
    (mv,) = c.bone_transforms
    assert mv["bone_id"] == 20
    # Under the avatar's Armature (identity in Unity) at (0, -0.06, 1.08):
    # world (0, -0.06, 1.08) -> in the FBX's -90 X Armature frame.
    assert mv["position"] == pytest.approx([0.0, -1.08, -0.06], abs=1e-6)


def test_near_unit_fbx_scales_do_not_read_as_shear(tmp_path: Path):
    """Marette's FBX carries scales like (0.999998, 1, 0.999995) above
    rotated skirt bones; Unity's copy has 1.0. Not a move, not an error."""
    models = [
        (10, "Armature", "Null", 0, (0, 0, 0), (-90, 0, 0)),
        (11, "Hips", "LimbNode", 10, (0, 0, 1.0), (90, 0, 0)),
        (12, "Skirt_L01.000", "LimbNode", 11, (0, -0.1, 0), (10.5, -33.2, 179.9)),
        (13, "Skirt_L01.001", "LimbNode", 12, (0, 0.1, 0), (-1.0, 12.4, -2.1)),
        (14, "Skirt", "Mesh", 0, (0, 0, 0), (-90, 0, 0)),
    ]
    text = fbx_ascii(models).decode().replace(
        'P: "Lcl Rotation", "Lcl Rotation", "", "A",10.5,-33.2,179.9\n',
        'P: "Lcl Rotation", "Lcl Rotation", "", "A",10.5,-33.2,179.9\n'
        '\t\t\tP: "Lcl Scaling", "Lcl Scaling", "", "A",0.999998,1,0.999995\n')
    proj = make_project(tmp_path)
    ava = write_fbx(proj, "Assets/Ava.fbx", fbx_ascii(AVA_MODELS), AVA_FBX)
    skirt = write_fbx(proj, "Assets/Skirt.fbx", text.encode(), OUT_FBX)

    from rigforge.ascii_fbx.sections import model_local_transform
    from rigforge.ascii_fbx.transform import fbx_to_unity, unity_euler_to_matrix
    from rigforge.unity.scene_read import _quat_from_matrix
    view = extract(parse(skirt.read_bytes()))

    def unity_local(mid):
        u = fbx_to_unity(model_local_transform(view, mid))
        return u.position, _quat_from_matrix(unity_euler_to_matrix(*u.rotation))

    p12, q12 = unity_local(12)
    p13, q13 = unity_local(13)
    write_asset(proj, "Assets/Ava.prefab", HEADER + "".join([
        game_object(100, "Ava", [101]), transform(101, 100, 0, children=[103, 109]),
        game_object(102, "Armature", [103]), transform(103, 102, 101, children=[105]),
        game_object(104, "Hips", [105]), transform(105, 104, 103, pos=(0, 1, 0), children=[107]),
        game_object(106, "Head", [107]), transform(107, 106, 105, pos=(0, 0.5, 0)),
        game_object(108, "Body", [109, 110]), transform(109, 108, 101),
        skinned_mesh(110, 108, [], mesh_guid=AVA_FBX),
    ]), AVA_PREFAB)
    write_asset(proj, "Assets/Skirt.prefab", HEADER + "".join([
        game_object(200, "SkirtOutfit", [201]), transform(201, 200, 0, children=[203, 211]),
        game_object(202, "Armature", [203]), transform(203, 202, 201, children=[205]),
        game_object(204, "Hips", [205]), transform(205, 204, 203, pos=(0, 1, 0), children=[207]),
        game_object(206, "Skirt_L01.000", [207]),
        transform(207, 206, 205, pos=p12, rot=q12, children=[209]),     # Unity: scale 1
        game_object(208, "Skirt_L01.001", [209]), transform(209, 208, 207, pos=p13, rot=q13),
        game_object(210, "Skirt", [211, 212]), transform(211, 210, 201),
        skinned_mesh(212, 210, [], mesh_guid=OUT_FBX),
    ]), PROP_PREFAB)
    scene = write_asset(proj, "Assets/S.unity", HEADER + "".join([
        prefab_instance(1000, AVA_PREFAB),
        stripped(4, "Transform", 1002, 103, AVA_PREFAB, 1000),
        prefab_instance(2000, PROP_PREFAB, parent=1002),
    ]), "f" * 32)
    res = _read(scene, _registry(ava))
    (c,) = res.clothings
    assert c.bone_transforms == []
    assert not any("shear" in n for n in res.notes), res.notes


def test_duplicate_names_prefer_the_renderer_for_a_mesh(tmp_path: Path):
    """Marette: a bone object and a mesh object both named 'Bag'."""
    models = [
        (10, "Armature", "Null", 0, (0, 0, 0), (-90, 0, 0)),
        (11, "Hips", "LimbNode", 10, (0, 0, 1.0), (90, 0, 0)),
        (12, "Bag", "LimbNode", 11, (0, 0.1, 0), (0, 0, 0)),
        (13, "Bag", "Mesh", 0, (0, 0, 0), (-90, 0, 0)),
        (14, "Strap", "Mesh", 0, (0, 0, 0), (-90, 0, 0)),
    ]
    proj = make_project(tmp_path)
    ava = write_fbx(proj, "Assets/Ava.fbx", fbx_ascii(AVA_MODELS), AVA_FBX)
    write_fbx(proj, "Assets/Bag.fbx", fbx_ascii(models), OUT_FBX)
    write_asset(proj, "Assets/Ava.prefab", HEADER + "".join([
        game_object(100, "Ava", [101]), transform(101, 100, 0, children=[103, 109]),
        game_object(102, "Armature", [103]), transform(103, 102, 101, children=[105]),
        game_object(104, "Hips", [105]), transform(105, 104, 103, pos=(0, 1, 0), children=[107]),
        game_object(106, "Head", [107]), transform(107, 106, 105, pos=(0, 0.5, 0)),
        game_object(108, "Body", [109, 110]), transform(109, 108, 101),
        skinned_mesh(110, 108, [], mesh_guid=AVA_FBX),
    ]), AVA_PREFAB)
    write_asset(proj, "Assets/Bag.prefab", HEADER + "".join([
        game_object(200, "BagOutfit", [201]), transform(201, 200, 0, children=[203, 211, 215]),
        game_object(202, "Armature", [203]), transform(203, 202, 201, children=[205]),
        game_object(204, "Hips", [205]), transform(205, 204, 203, pos=(0, 1, 0), children=[207]),
        game_object(206, "Bag", [207]), transform(207, 206, 205, pos=(0, 0.1, 0)),
        game_object(210, "Bag", [211, 212], active=0), transform(211, 210, 201),
        skinned_mesh(212, 210, [], mesh_guid=OUT_FBX),
        game_object(214, "Strap", [215, 216]), transform(215, 214, 201),
        skinned_mesh(216, 214, [], mesh_guid=OUT_FBX),
    ]), PROP_PREFAB)
    scene = write_asset(proj, "Assets/S.unity", HEADER + "".join([
        prefab_instance(1000, AVA_PREFAB),
        stripped(4, "Transform", 1002, 103, AVA_PREFAB, 1000),
        prefab_instance(2000, PROP_PREFAB, parent=1002),
    ]), "f" * 32)
    res = _read(scene, _registry(ava))
    (c,) = res.clothings
    assert c.drop_mesh_ids == [13]             # the inactive renderer, not the bone
    assert not any("ambiguous" in n for n in res.notes)


def test_an_outfit_whose_meshes_have_no_fbx_beside_them_is_reported(tmp_path: Path):
    """save_08: the Riptide / SimpleLeotard variants draw from MochiFitter's
    .omochi while their exported FBX was deleted: nothing to merge — say so."""
    OMOCHI = "6" * 32
    proj = make_project(tmp_path)
    ava = write_fbx(proj, "Assets/Ava.fbx", fbx_ascii(AVA_MODELS), AVA_FBX)
    write_asset(proj, "Assets/Out/retargeted_Riptide.omochi", "mesh store", OMOCHI)
    write_asset(proj, "Assets/Ava.prefab", HEADER + "".join([
        game_object(100, "Ava", [101]), transform(101, 100, 0, children=[103, 109]),
        game_object(102, "Armature", [103]), transform(103, 102, 101, children=[105]),
        game_object(104, "Hips", [105]), transform(105, 104, 103, pos=(0, 1, 0), children=[107]),
        game_object(106, "Head", [107]), transform(107, 106, 105, pos=(0, 0.5, 0)),
        game_object(108, "Body", [109, 110]), transform(109, 108, 101),
        skinned_mesh(110, 108, [], mesh_guid=AVA_FBX),
    ]), AVA_PREFAB)
    write_asset(proj, "Assets/Out/Riptide.prefab", HEADER + "".join([
        game_object(200, "Riptide_B_Blue", [201]), transform(201, 200, 0, children=[203]),
        game_object(202, "Riptide_Watch", [203, 204]), transform(203, 202, 201),
        skinned_mesh(204, 202, [], mesh_guid=OMOCHI),
    ]), PROP_PREFAB)
    scene = write_asset(proj, "Assets/S.unity", HEADER + "".join([
        prefab_instance(1000, AVA_PREFAB),
        stripped(4, "Transform", 1002, 103, AVA_PREFAB, 1000),
        prefab_instance(2000, PROP_PREFAB, parent=1002),
    ]), "f" * 32)
    res = _read(scene, _registry(ava))
    assert res.clothings == []
    (sk,) = res.skipped
    assert sk["name"] == "Riptide_B_Blue"
    assert "retargeted_Riptide.omochi" in sk["reason"] and "no FBX" in sk["reason"]
