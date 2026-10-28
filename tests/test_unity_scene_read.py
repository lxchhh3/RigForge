"""Reading a Unity scene into a compose pre-fill: which garments, which meshes
are hidden, which blendshape values and bone/prop moves kevin set in Unity —
all resolved to the ids the FE's inspect payload uses."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.avatars.fingerprint import fingerprint
from rigforge.avatars.registry import AvatarRegistry, CuratedAvatar
from rigforge.unity.hierarchy import compose_id
from rigforge.unity.scene_read import SceneReadError, read_scene
from tests.unity_fixtures import (
    HEADER,
    MODEL_ROOT_GO,
    MODEL_ROOT_TRANSFORM,
    fbx_ascii,
    game_object,
    make_project,
    mono,
    prefab_instance,
    skinned_mesh,
    stripped,
    transform,
    write_asset,
    write_fbx,
)

AVA_FBX, AVA_PREFAB = "a" * 32, "b" * 32
OUT_FBX, OUT_PREFAB = "c" * 32, "d" * 32
# MochiFitter: the prefab's renderers use the .omochi mesh asset that sits
# beside the exported FBX of the same name.
OUT_OMOCHI = "c" * 30 + "aa"
PIN_FBX, PIN_PREFAB = "e" * 30 + "01", "e" * 30 + "02"
RIG_FBX = "e" * 30 + "03"
TOGGLE = "a162bb8ec7e24a5abcf457887f1df3fa"
PROXY = "42581d8044b64899834d3d515ab3a144"
SHAPER = "2db441f589c3407bb6fb5f02ff8ab541"
# Set the avatar Body's Sad to 55, delete the vertices under its Smile.
SHAPER_BODY = """\
  m_inverted: 0
  m_shapes:
  - Object:
      referencePath: Body
      targetObject: {fileID: 0}
    ShapeName: Sad
    ChangeType: 1
    Value: 55
  - Object:
      referencePath: Body
      targetObject: {fileID: 0}
    ShapeName: Smile
    ChangeType: 0
    Value: 0
  m_threshold: 0.01
"""

S = math.sqrt(0.5)

# --- the avatar: FBX (Blender-style -90 X Armature) + its flat Unity prefab ---
AVA_MODELS = [
    (1, "Armature", "Null", 0, (0, 0, 0), (-90, 0, 0)),
    (2, "Hips", "LimbNode", 1, (0, 0, 1.0), (90, 0, 0)),     # world (0, 1, 0)
    (3, "Head", "LimbNode", 2, (0, 0.5, 0), (0, 0, 0)),      # world (0, 1.5, 0)
    (6, "Eye_L", "LimbNode", 3, (0.03, 0.05, 0), (0, 0, 0)),
    (4, "Body", "Mesh", 0, (0, 0, 0), (-90, 0, 0)),
    (5, "Cloth", "Mesh", 0, (0, 0, 0), (-90, 0, 0)),
]
AVA_CHANNELS = {4: [("Smile", 0), ("Sad", 30)]}

# --- a MochiFitter-style outfit: same skeleton, `_MFBase` names in the FBX ---
OUT_MODELS = [
    (10, "Armature", "Null", 0, (0, 0, 0), (-90, 0, 0)),
    (11, "Hips_MFBase", "LimbNode", 10, (0, 0, 1.0), (90, 0, 0)),
    (12, "Head", "LimbNode", 11, (0, 0.5, 0), (0, 0, 0)),
    (13, "Ribbon_Root", "LimbNode", 12, (0, 0.1, 0), (0, 0, 0)),
    (16, "Eye_L", "LimbNode", 12, (0.03, 0.05, 0), (0, 0, 0)),
    (17, "Bow", "LimbNode", 12, (0, 0.2, 0), (0, 0, 0)),
    (14, "Skirt", "Mesh", 0, (0, 0, 0), (-90, 0, 0)),
    (15, "Shorts", "Mesh", 0, (0, 0, 0), (-90, 0, 0)),
]
OUT_CHANNELS = {14: [("Long", 0), ("Flare", 0), ("Tight", 20)]}

PIN_MODELS = [(20, "Pin", "Mesh", 0, (0, 0, 0), (0, 0, 0))]
RIG_CHANNELS = {32: [("Eat", 0)]}
RIG_T_ARMATURE = -7185857084041579234     # inner ids of Unity's import (kevin's Toast)
RIG_SMR_TOAST = 5120338455441729675
RIG_MODELS = [
    (30, "Armature_Toast", "Null", 0, (0, 0, 0), (0, 0, 0)),
    (31, "PB_Toast_01", "LimbNode", 30, (0, 0, 0), (0, 0, 0)),
    (32, "Toast", "Mesh", 0, (0, 0, 0), (0, 0, 0)),
]


def _ava_prefab() -> str:
    # Unity bakes the -90 X: Armature is identity, Hips sits at (0, 1, 0).
    return HEADER + "".join([
        game_object(100, "Ava", [101]),
        transform(101, 100, 0, children=[103, 109, 111]),
        game_object(102, "Armature", [103]),
        transform(103, 102, 101, children=[105]),
        game_object(104, "Hips", [105]),
        transform(105, 104, 103, pos=(0, 1, 0), children=[107]),
        game_object(106, "Head", [107]),
        transform(107, 106, 105, pos=(0, 0.5, 0), children=[115]),
        game_object(114, "Eye_L", [115]),
        transform(115, 114, 107, pos=(-0.03, 0.05, 0)),
        game_object(108, "Body", [109, 110]),
        transform(109, 108, 101),
        skinned_mesh(110, 108, [0, 30], mesh_guid=AVA_FBX),
        game_object(112, "Cloth", [111, 113]),
        transform(111, 112, 101),
        skinned_mesh(113, 112, [], mesh_guid=AVA_FBX),
    ])


def _outfit_prefab() -> str:
    return HEADER + "".join([
        game_object(200, "Outfit", [201]),
        transform(201, 200, 0, children=[203, 209, 211]),
        game_object(202, "Armature", [203]),
        transform(203, 202, 201, children=[205]),
        game_object(204, "Hips", [205]),
        transform(205, 204, 203, pos=(0, 1, 0), children=[207]),
        game_object(206, "Head", [207]),
        transform(207, 206, 205, pos=(0, 0.5, 0), children=[213, 217, 219]),
        game_object(216, "Eye_L", [217]),
        transform(217, 216, 207, pos=(-0.03, 0.05, 0)),
        game_object(218, "Bow", [219]),
        transform(219, 218, 207, pos=(0, 0.2, 0)),
        game_object(212, "Ribbon_Root", [213]),
        transform(213, 212, 207, pos=(0, 0.1, 0)),
        game_object(208, "Skirt", [209, 210]),
        transform(209, 208, 201),
        skinned_mesh(210, 208, [0, 0, 20], mesh_guid=OUT_OMOCHI),
        game_object(214, "Shorts", [211, 215]),
        transform(211, 214, 201),
        skinned_mesh(215, 214, [], mesh_guid=OUT_OMOCHI),
    ])


def _pin_prefab() -> str:
    # PinHolder (Bone Proxy -> Head, AsChildAtRoot) -> a Pin.fbx instance at x=0.03
    return HEADER + "".join([
        game_object(300, "PinHolder", [301, 302]),
        transform(301, 300, 0, children=[compose_id(400, MODEL_ROOT_TRANSFORM)]),
        mono(302, 300, PROXY, "  boneReference: 10\n  subPath: \n  attachmentMode: 1\n"),
        prefab_instance(400, PIN_FBX, parent=301, mods=[
            (MODEL_ROOT_TRANSFORM, "m_LocalPosition.x", "0.03"),
        ]),
    ])


def _toggle(fid: int, go: int, entries: list[tuple[int, str, int]]) -> str:
    objs = "".join(
        f"  - Object:\n      referencePath: {p}\n      targetObject: {{fileID: {t}}}\n    Active: {a}\n"
        for t, p, a in entries
    )
    return mono(fid, go, TOGGLE, f"  m_inverted: 0\n  m_objects:\n{objs}")


def _scene() -> str:
    return HEADER + "".join([
        prefab_instance(1000, AVA_PREFAB, mods=[
            (110, "m_BlendShapeWeights.Array.data[0]", "80"),
            (110, "m_BlendShapeWeights.Array.data[1]", "30"),
        ], added_components=[(100, 1003)]),
        stripped(1, "GameObject", 1001, 100, AVA_PREFAB, 1000),
        stripped(4, "Transform", 1002, 103, AVA_PREFAB, 1000),
        stripped(1, "GameObject", 1004, 112, AVA_PREFAB, 1000),
        _toggle(1003, 1001, [(1004, "Cloth", 0)]),
        # the outfit, under the avatar's Armature
        prefab_instance(2000, OUT_PREFAB, parent=1002, mods=[
            (210, "m_BlendShapeWeights.Array.data[1]", "50"),
            (213, "m_LocalPosition.x", "0.02"),
            (207, "m_LocalPosition.y", "0.57"),
        ], added_components=[(200, 2003), (200, 2005)]),
        stripped(1, "GameObject", 2001, 200, OUT_PREFAB, 2000),
        stripped(1, "GameObject", 2004, 214, OUT_PREFAB, 2000),
        _toggle(2003, 2001, [(2004, "Outfit/Shorts", 0)]),
        # the outfit ships an MA Shape Changer on the avatar's Body
        mono(2005, 2001, SHAPER, SHAPER_BODY),
        # a rigid hair pin on a Bone Proxy
        prefab_instance(3000, PIN_PREFAB, parent=1002, mods=[
            (301, "m_LocalPosition.z", "0.9"),   # where the editor left it; build snaps to Head
        ]),
        # a self-rigged prop (its own armature), placed under the Armature at
        # (0, 1.2, 0.1); the scene also moves its inner Armature_Toast and
        # sets its inner Toast mesh's 'Eat' weight (kevin's Toast)
        prefab_instance(4000, RIG_FBX, parent=1002, mods=[
            (MODEL_ROOT_TRANSFORM, "m_LocalPosition.y", "1.2"),
            (MODEL_ROOT_TRANSFORM, "m_LocalPosition.z", "0.1"),
            (RIG_T_ARMATURE, "m_LocalPosition.y", "-0.05"),
            (RIG_SMR_TOAST, "m_BlendShapeWeights.Array.data[0]", "100"),
        ]),
    ])


def build_scene_project(tmp_path: Path):
    """The synthetic Unity project: (scene path, registry, {outfit, pin} FBX)."""
    proj = make_project(tmp_path)
    ava_fbx = write_fbx(proj, "Assets/Ava/Ava.fbx", fbx_ascii(AVA_MODELS, AVA_CHANNELS), AVA_FBX)
    write_asset(proj, "Assets/Ava/Ava.prefab", _ava_prefab(), AVA_PREFAB)
    out_fbx = write_fbx(proj, "Assets/Out/Outfit.fbx", fbx_ascii(OUT_MODELS, OUT_CHANNELS), OUT_FBX)
    write_asset(proj, "Assets/Out/Outfit.prefab", _outfit_prefab(), OUT_PREFAB)
    write_asset(proj, "Assets/Out/Outfit.omochi", "binary mesh store", OUT_OMOCHI)
    pin_fbx = write_fbx(proj, "Assets/Pin/Pin.fbx", fbx_ascii(PIN_MODELS), PIN_FBX)
    write_asset(proj, "Assets/Pin/PinHolder.prefab", _pin_prefab(), PIN_PREFAB)
    write_fbx(proj, "Assets/Rig/SelfRig.fbx", fbx_ascii(RIG_MODELS, RIG_CHANNELS), RIG_FBX)
    # Unity's import of it (Library/Artifacts): names its inner objects.
    from tests.test_unity_library import component, serialized_file
    from tests.test_unity_library import game_object as art_go
    art = proj / "Library" / "Artifacts" / "ab" / "abcdef"
    art.parent.mkdir(parents=True)
    art.write_bytes(serialized_file([
        (MODEL_ROOT_GO, 1, art_go("SelfRig", [MODEL_ROOT_TRANSFORM])),
        (MODEL_ROOT_TRANSFORM, 4, component(MODEL_ROOT_GO)),
        (-4305874778761266443, 1, art_go("Armature_Toast", [RIG_T_ARMATURE])),
        (RIG_T_ARMATURE, 4, component(-4305874778761266443)),
        (-8427158191399166764, 1, art_go("Toast", [RIG_SMR_TOAST])),
        (RIG_SMR_TOAST, 137, component(-8427158191399166764)),
    ], type_tree=True))
    write_asset(proj, "Packages/ma/ModularAvatarObjectToggle.cs", "//", TOGGLE)
    write_asset(proj, "Packages/ma/ModularAvatarBoneProxy.cs", "//", PROXY)
    write_asset(proj, "Packages/ma/ModularAvatarShapeChanger.cs", "//", SHAPER)
    scene = write_asset(proj, "Assets/Scene.unity", _scene(), "f" * 32)

    names = [m[1] for m in AVA_MODELS]
    registry = AvatarRegistry(avatars={"ava": CuratedAvatar(
        id="ava", display_name="Ava", fbx_path=ava_fbx, bone_names=names,
        fingerprint=fingerprint(names),
        canonical_to_name={"Hips": "Hips", "Head": "Head"},
        ascii_fbx_path=ava_fbx,
    )})
    return scene, registry, {"outfit": out_fbx, "pin": pin_fbx}


@pytest.fixture
def setup(tmp_path: Path):
    scene, registry, paths = build_scene_project(tmp_path)
    res = read_scene(
        scene, registry=registry,
        load_view=lambda p: extract(parse(Path(p).read_bytes())),
    )
    return res, paths


def _clothing(res, path: Path):
    (c,) = [c for c in res.clothings if Path(c.clothing_path) == path]
    return c


def test_target_is_the_registry_avatar_the_scene_is_built_on(setup):
    res, _ = setup
    assert res.target_id == "ava"


def test_target_mesh_hidden_by_a_toggle_is_dropped(setup):
    res, _ = setup
    assert res.target.drop_mesh_ids == [5]


def test_target_blendshape_moved_off_the_file_value_is_an_override(setup):
    res, _ = setup
    # Smile 0 -> 80 is kevin's; Sad 30 equals the file and is not sent.
    assert res.target.channel_overrides[4 + 300000] == 80.0


def test_unmoved_bones_send_no_moves(setup):
    res, _ = setup
    assert res.target.bone_transforms == []


def test_outfit_hidden_mesh_and_blendshape(setup):
    res, paths = setup
    c = _clothing(res, paths["outfit"])
    assert c.rigid is False
    assert c.drop_mesh_ids == [15]
    assert c.channel_overrides == {14 + 300001: 50.0}


def test_outfit_ride_along_bone_move_lands_where_unity_shows_it(setup):
    """Ribbon_Root moved +2 cm x; its parent Head moved +7 cm y in the scene.
    Head merges into the avatar's own Head (unmoved), so Ribbon_Root's move
    carries both: (0.02, 0.1 + 0.07, 0) in Unity numbers."""
    res, paths = setup
    c = _clothing(res, paths["outfit"])
    mv = {m["bone_id"]: m for m in c.bone_transforms}[13]
    assert mv["position"] == pytest.approx([0.02, 0.17, 0.0], abs=1e-6)
    assert mv["rotation"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_an_untouched_ride_along_bone_under_an_uncarried_move_still_lands_in_place(setup):
    """Bow was not touched, but rides on the moved Head in Unity: it gets the
    Head's offset so it stays where kevin sees it."""
    res, paths = setup
    c = _clothing(res, paths["outfit"])
    mv = {m["bone_id"]: m for m in c.bone_transforms}[17]
    assert mv["position"] == pytest.approx([0.0, 0.27, 0.0], abs=1e-6)
    assert sorted(m["bone_id"] for m in c.bone_transforms) == [12, 13, 17]


def test_a_shared_bone_riding_on_a_moved_one_is_not_reported(setup):
    """Eye_L only moved because Head did; it is replaced by the avatar's own
    Eye_L anyway, so it is not news."""
    res, _ = setup
    assert not any("Eye_L" in n for n in res.notes)


def test_a_moved_bone_the_merge_replaces_is_sent_for_the_vertex_bake(setup):
    """The outfit's Head merges into the avatar's Head, so Phase C bakes its
    move into the outfit's vertices (kevin's pick 2026-10-28): the reader
    sends the move like any other, in the FBX's parent frame."""
    res, paths = setup
    c = _clothing(res, paths["outfit"])
    mv = {m["bone_id"]: m for m in c.bone_transforms}[12]
    assert mv["position"] == pytest.approx([0.0, 0.57, 0.0], abs=1e-6)
    assert any("Head" in n and "baked into" in n for n in res.notes)
    assert not any("can't carry" in n for n in res.notes)


def test_rigid_prop_on_a_bone_proxy_attaches_to_that_bone(setup):
    res, paths = setup
    c = _clothing(res, paths["pin"])
    assert c.rigid is True
    assert c.attach_to_bone_id == 3          # the avatar's Head


def test_rigid_prop_placement_is_the_bone_plus_its_offset(setup):
    """Built: PinHolder snaps to Head (world (0,1.5,0)), the pin 3 cm along
    +x. In RigForge's move frame (the avatar FBX's -90 X Armature) that is
    Unity numbers (0.03, 0, 1.5) / (90, 0, 0)."""
    res, paths = setup
    c = _clothing(res, paths["pin"])
    (mv,) = c.bone_transforms
    assert mv["bone_id"] == 20
    assert mv["position"] == pytest.approx([0.03, 0.0, 1.5], abs=1e-6)
    assert mv["rotation"] == pytest.approx([90.0, 0.0, 0.0], abs=1e-4)
    assert mv["scale"] == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)


def _selfrig(res):
    (c,) = [c for c in res.clothings if Path(c.clothing_path).name == "SelfRig.fbx"]
    return c


def test_a_self_rigged_prop_is_read_as_a_prop(setup):
    """kevin's pick 2026-10-28: merged as-is, no longer skipped."""
    res, _ = setup
    assert res.skipped == []
    c = _selfrig(res)
    assert c.self_rigged is True and c.rigid is False
    assert c.attach_to_bone_id is None            # under the Armature, no Bone Proxy


def test_a_self_rigged_prop_is_placed_by_its_armature_root_with_inner_moves(setup):
    """Its root stands at (0, 1.2, 0.1) and the scene lowers its inner
    Armature_Toast 5 cm: in RigForge's move frame (the avatar FBX's -90 X
    Armature) that is Unity numbers (0, -0.1, 1.15) / (90, 0, 0)."""
    res, _ = setup
    mv = {m["bone_id"]: m for m in _selfrig(res).bone_transforms}
    # Placed by its top bone PB_Toast_01 (its Null is stripped in the merge).
    assert 30 not in mv
    assert mv[31]["position"] == pytest.approx([0.0, -0.1, 1.15], abs=1e-6)
    assert mv[31]["rotation"] == pytest.approx([90.0, 0.0, 0.0], abs=1e-4)
    # Its unskinned Toast mesh is placed too (it would not follow a bone).
    assert mv[32]["position"] == pytest.approx([0.0, -0.1, 1.2], abs=1e-6)


def test_an_inner_renderers_weight_is_read_through_the_import_artifact(setup):
    res, _ = setup
    assert _selfrig(res).channel_overrides == {32 + 300000: 100.0}


def test_result_serialises_for_the_api(setup):
    res, paths = setup
    d = res.to_dict()
    assert d["target_id"] == "ava"
    assert d["target"]["drop_mesh_ids"] == [5]
    assert d["target"]["blend_shape_channel_overrides"] == [
        {"channel_id": 300004, "deform_percent": 80.0},
        {"channel_id": 300005, "deform_percent": 55.0},
    ]
    paths_out = [c["clothing_path"] for c in d["clothings"]]
    assert str(paths["outfit"]) in paths_out and str(paths["pin"]) in paths_out
    assert isinstance(d["notes"], list) and d["skipped"] == []
    (sr,) = [c for c in d["clothings"] if c["clothing_path"].endswith("SelfRig.fbx")]
    assert sr["self_rigged"] is True


def test_a_scene_with_no_registry_avatar_is_an_error(tmp_path: Path):
    proj = make_project(tmp_path)
    scene = write_asset(proj, "Assets/S.unity", HEADER + game_object(1, "Cube", []), "f" * 32)
    with pytest.raises(SceneReadError):
        read_scene(scene, registry=AvatarRegistry(avatars={}),
                   load_view=lambda p: extract(parse(Path(p).read_bytes())))


def test_outfits_merge_before_props(setup):
    """Merge order names the output (<outfit>+<prop>__<target>.fbx): the
    skinned outfits first, props (rigid or self-rigged) after, each in scene
    order."""
    res, paths = setup
    order = [Path(c.clothing_path).name for c in res.clothings]
    assert order == ["Outfit.fbx", "Pin.fbx", "SelfRig.fbx"]


def test_an_fbx_that_does_not_match_the_scene_copy_is_refused(tmp_path: Path):
    """Every bone's own local differs (a different FBX than the prefab was
    made from): refuse the garment rather than send a move for each bone."""
    scene, registry, paths = build_scene_project(tmp_path)
    stretched = [
        (mid, name, cls, parent, tuple(x * 1.5 for x in t), r) if cls == "LimbNode"
        else (mid, name, cls, parent, t, r)
        for mid, name, cls, parent, t, r in OUT_MODELS
    ]
    paths["outfit"].write_bytes(fbx_ascii(stretched, OUT_CHANNELS))
    res = read_scene(scene, registry=registry,
                     load_view=lambda p: extract(parse(Path(p).read_bytes())))
    sk = {s["name"]: s["reason"] for s in res.skipped}
    assert "does not line up" in sk["Outfit"]


def test_an_ma_shape_changer_sets_the_avatars_shape(setup):
    """The outfit's Shape Changer sets the avatar Body's Sad to 55 (file 30)."""
    res, _ = setup
    assert res.target.channel_overrides[4 + 300001] == 55.0


def test_an_ma_shape_changer_delete_is_reported(setup):
    res, _ = setup
    assert any("Smile" in n and "delete" in n.lower() for n in res.notes)
