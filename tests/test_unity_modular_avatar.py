"""Modular Avatar components as RigForge reads them: Object Toggles decide
which meshes ship, Bone Proxies decide which bone a prop hangs from. The rules
mirror MA's own ReactiveObjectAnalyzer (BuildConditions + the evaluation
loop), read 2026-10-28 from New_Mochi's MA 1.18.7 package."""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.unity.hierarchy import UnityProject, expand
from rigforge.unity.modular_avatar import bone_proxies, effective_active
from tests.unity_fixtures import (
    HEADER,
    game_object,
    make_project,
    mono,
    transform,
    write_asset,
)

TOGGLE = "a162bb8ec7e24a5abcf457887f1df3fa"
MENU = "3b29d45007c5493d926d2cd45a489529"
PROXY = "42581d8044b64899834d3d515ab3a144"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    proj = make_project(tmp_path)
    for name, guid in (("ModularAvatarObjectToggle", TOGGLE),
                       ("ModularAvatarMenuItem", MENU),
                       ("ModularAvatarBoneProxy", PROXY)):
        write_asset(proj, f"Packages/ma/{name}.cs", "// cs", guid)
    return proj


def _toggle(fid: int, go: int, entries: list[tuple[int, str, int]], *, inverted: int = 0,
            enabled: int = 1) -> str:
    objs = "".join(
        f"  - Object:\n      referencePath: {path}\n      targetObject: {{fileID: {tgt}}}\n"
        f"    Active: {active}\n"
        for tgt, path, active in entries
    )
    return mono(fid, go, TOGGLE, f"  m_inverted: {inverted}\n  m_objects:\n{objs}",
                enabled=enabled)


def _menu(fid: int, go: int, *, is_default: int) -> str:
    return mono(fid, go, MENU,
                "  Control:\n    name: \n    type: 102\n"
                f"  isDefault: {is_default}\n  label: Acc\n")


def _avatar(*extra: str) -> str:
    # Avatar root (1/2) -> Armature (3/4) -> Head (5/6); Cloth (7/8); Hat (9/10)
    return HEADER + "".join([
        game_object(1, "Avatar", [2]),
        transform(2, 1, children=[4, 8, 10]),
        game_object(3, "Armature", [4]),
        transform(4, 3, 2, children=[6]),
        game_object(5, "Head", [6]),
        transform(6, 5, 4),
        game_object(7, "Cloth", [8]),
        transform(8, 7, 2),
        game_object(9, "Hat", [10]),
        transform(10, 9, 2),
        *extra,
    ])


def _active(project: Path, text: str) -> dict[str, bool]:
    scene = write_asset(project, "Assets/S.unity", text, "e" * 32)
    proj = UnityProject(project)
    g = expand(proj, scene)
    act = effective_active(proj, g, avatar_root=g.find_go("Avatar"))
    return {g.name(go): v for go, v in act.items()}


def test_a_toggle_on_the_avatar_root_always_applies(project: Path):
    act = _active(project, _avatar(_toggle(50, 1, [(7, "Cloth", 0), (9, "Hat", 1)])))
    assert act["Cloth"] is False
    assert act["Hat"] is True


def test_a_menu_item_that_defaults_on_enables_its_toggle(project: Path):
    act = _active(project, _avatar(
        game_object(20, "Menu", [21]), transform(21, 20, 2),
        _menu(22, 20, is_default=1),
        _toggle(23, 20, [(7, "Cloth", 0)]),
    ))
    assert act["Cloth"] is False


def test_a_menu_item_that_defaults_off_leaves_the_object_as_it_is(project: Path):
    act = _active(project, _avatar(
        game_object(20, "Menu", [21]), transform(21, 20, 2),
        _menu(22, 20, is_default=0),
        _toggle(23, 20, [(7, "Cloth", 0)]),
    ))
    assert act["Cloth"] is True


def test_inverted_flips_the_condition(project: Path):
    """kevin's EclatAndCrumb_Hairpin_MA: menu default ON + inverted toggle
    hiding the pins => the pins show by default."""
    act = _active(project, _avatar(
        game_object(20, "Menu", [21]), transform(21, 20, 2),
        _menu(22, 20, is_default=1),
        _toggle(23, 20, [(7, "Cloth", 0)], inverted=1),
    ))
    assert act["Cloth"] is True


def test_an_inactive_parent_of_the_toggle_disables_it(project: Path):
    act = _active(project, _avatar(
        game_object(20, "Off", [21], active=0), transform(21, 20, 2, children=[25]),
        game_object(24, "Holder", [25]), transform(25, 24, 21),
        _toggle(26, 24, [(7, "Cloth", 0)]),
    ))
    assert act["Cloth"] is True


def test_a_disabled_toggle_component_does_nothing(project: Path):
    act = _active(project, _avatar(_toggle(50, 1, [(7, "Cloth", 0)], enabled=0)))
    assert act["Cloth"] is True


def test_toggle_target_falls_back_to_its_reference_path(project: Path):
    act = _active(project, _avatar(_toggle(50, 1, [(999, "Armature/Head", 0)])))
    assert act["Head"] is False


def test_an_inactive_game_object_stays_inactive(project: Path):
    text = _avatar().replace("m_Name: Hat\n  m_TagString: Untagged\n  m_IsActive: 1",
                             "m_Name: Hat\n  m_TagString: Untagged\n  m_IsActive: 0")
    assert _active(project, text)["Hat"] is False


def test_bone_proxy_reads_humanoid_bone_and_mode(project: Path):
    scene = write_asset(project, "Assets/S.unity", _avatar(
        game_object(20, "Pins", [21]), transform(21, 20, 4),
        mono(22, 20, PROXY, "  boneReference: 10\n  subPath: \n  attachmentMode: 1\n"),
    ), "e" * 32)
    proj = UnityProject(project)
    g = expand(proj, scene)
    (bp,) = bone_proxies(proj, g)
    assert bp.go == g.find_go("Pins")
    assert bp.human_bone == "Head"
    assert bp.sub_path == ""
    assert bp.mode == "AsChildAtRoot"


SHAPER = "2db441f589c3407bb6fb5f02ff8ab541"


def _shaper(fid: int, go: int, entries: list[tuple[str, str, int, float]], *, inverted: int = 0) -> str:
    """entries: (referencePath, ShapeName, ChangeType 0=Delete 1=Set, Value)."""
    shapes = "".join(
        f"  - Object:\n      referencePath: {path}\n      targetObject: {{fileID: 0}}\n"
        f"    ShapeName: {name}\n    ChangeType: {kind}\n    Value: {value}\n"
        for path, name, kind, value in entries
    )
    return mono(fid, go, SHAPER, f"  m_inverted: {inverted}\n  m_shapes:\n{shapes}  m_threshold: 0.01\n")


def _changes(project: Path, text: str):
    from rigforge.unity.modular_avatar import shape_changes
    write_asset(project, "Packages/ma/ModularAvatarShapeChanger.cs", "// cs", SHAPER)
    scene = write_asset(project, "Assets/S.unity", text, "e" * 32)
    proj = UnityProject(project)
    g = expand(proj, scene)
    root = g.find_go("Avatar")
    st = effective_active(proj, g, avatar_root=root)
    return g, shape_changes(proj, g, st, avatar_root=root)


def test_a_shape_changer_sets_the_avatars_shape(project: Path):
    g, ch = _changes(project, _avatar(
        game_object(20, "Stockings", [21]), transform(21, 20, 2),
        _shaper(22, 20, [("Cloth", "For Stockings", 1, 100), ("Cloth", "Thigh", 0, 0)]),
    ))
    assert [(g.name(c.renderer_go), c.shape, c.delete, c.value) for c in ch] == [
        ("Cloth", "For Stockings", False, 100.0), ("Cloth", "Thigh", True, 0.0),
    ]


def test_a_shape_changer_under_an_inactive_object_does_nothing(project: Path):
    """No affected-object exclusion for shapes: every parent must be active."""
    _, ch = _changes(project, _avatar(
        game_object(20, "Off", [21], active=0), transform(21, 20, 2),
        _shaper(22, 20, [("Cloth", "For Stockings", 1, 100)]),
    ))
    assert ch == []


def test_an_inverted_shape_changer_under_a_default_on_menu_does_nothing(project: Path):
    _, ch = _changes(project, _avatar(
        game_object(20, "Menu", [21]), transform(21, 20, 2),
        _menu(23, 20, is_default=1),
        _shaper(22, 20, [("Cloth", "For Stockings", 1, 100)], inverted=1),
    ))
    assert ch == []
