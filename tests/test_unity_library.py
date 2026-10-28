"""Names of the objects inside an imported model, read from Unity's import
artifact (Library/Artifacts, a binary SerializedFile). A prefab override on an
FBX's inner object (kevin's Toast: Armature_Toast moved, EC_Toast 'Eat'=100)
names it only by fileID; the import artifact maps that fileID to the node."""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from rigforge.unity.library import find_model_artifact, model_object_names, read_object_names

MODEL_ROOT_GO = 919132149155446097
MODEL_ROOT_TRANSFORM = -8679921383154817045


def serialized_file(objects: list[tuple[int, int, bytes]], *, type_tree: bool) -> bytes:
    """A minimal SerializedFile v22: objects = (pathID, classID, payload)."""
    classes = sorted({c for _, c, _ in objects})
    meta = b"0.0.0\0" + struct.pack("<i", -2) + bytes([1 if type_tree else 0])
    meta += struct.pack("<i", len(classes))
    for c in classes:
        meta += struct.pack("<i", c) + b"\0" + struct.pack("<h", -1)
        if c == 114:
            meta += b"\0" * 16
        meta += b"\0" * 16
        if type_tree:
            strings = b"Base\0"
            meta += struct.pack("<ii", 1, len(strings)) + b"\0" * 32 + strings
            meta += struct.pack("<i", 0)            # type dependencies
    meta += struct.pack("<i", len(objects))
    body = b""
    table = b""
    for pid, cls, payload in objects:
        while (48 + len(meta) + len(table)) % 4:
            table += b"\0"
        table += struct.pack("<qqIi", pid, len(body), len(payload), classes.index(cls))
        body += payload
        while len(body) % 8:
            body += b"\0"
    meta += table + struct.pack("<i", 0) * 3      # script types, externals, ref types
    data_offset = 48 + len(meta)
    data_offset += (-data_offset) % 16
    header = struct.pack(">IIII", 0, 0, 22, 0) + bytes([0, 0, 0, 0])
    header += struct.pack(">IQQQ", len(meta), data_offset + len(body), data_offset, 0)
    out = header + meta
    out += b"\0" * (data_offset - len(out))
    return out + body


def game_object(name: str, components: list[int]) -> bytes:
    b = b"\0" * 40 + struct.pack("<i", len(components))
    for c in components:
        b += struct.pack("<iq", 0, c)
    raw = name.encode()
    b += struct.pack("<i", 0) + struct.pack("<i", len(raw)) + raw
    return b + b"\0" * ((-len(b)) % 4)


def component(go: int) -> bytes:
    return b"\0" * 40 + struct.pack("<iq", 0, go) + b"\0" * 16


def toast_like(type_tree: bool = True) -> bytes:
    return serialized_file([
        (MODEL_ROOT_GO, 1, game_object("EclatAndCrumb_Toast", [MODEL_ROOT_TRANSFORM])),
        (MODEL_ROOT_TRANSFORM, 4, component(MODEL_ROOT_GO)),
        (-4305874778761266443, 1, game_object("Armature_Toast", [-7185857084041579234])),
        (-7185857084041579234, 4, component(-4305874778761266443)),
        (-8427158191399166764, 1, game_object("EC_Toast", [5120338455441729675])),
        (5120338455441729675, 137, component(-8427158191399166764)),
    ], type_tree=type_tree)


@pytest.mark.parametrize("type_tree", [True, False])
def test_object_names_map_ids_to_their_game_object(tmp_path: Path, type_tree: bool):
    p = tmp_path / "artifact"
    p.write_bytes(toast_like(type_tree))
    names = read_object_names(p)
    assert names[MODEL_ROOT_GO] == (1, "EclatAndCrumb_Toast")
    assert names[-7185857084041579234] == (4, "Armature_Toast")
    assert names[5120338455441729675] == (137, "EC_Toast")


def test_a_non_unity_file_is_rejected(tmp_path: Path):
    p = tmp_path / "junk"
    p.write_bytes(b"not a serialized file at all" * 4)
    with pytest.raises(ValueError):
        read_object_names(p)


def _project_with_artifacts(tmp_path: Path, files: dict[str, bytes]) -> Path:
    proj = tmp_path / "Proj"
    for rel, data in files.items():
        f = proj / "Library" / "Artifacts" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
    (proj / "Assets").mkdir(parents=True, exist_ok=True)
    return proj


def test_the_model_artifact_is_the_one_with_the_model_root(tmp_path: Path):
    """A prefab built on the model also names its objects after the nodes;
    only the import itself has Unity's fixed model-root ids."""
    prefab_like = serialized_file([
        (1552608709405242259, 1, game_object("EclatAndCrumb_Toast", [2215816963118508329])),
        (2215816963118508329, 4, component(1552608709405242259)),
    ], type_tree=True)
    proj = _project_with_artifacts(tmp_path, {
        "be/be7c": prefab_like, "1b/1b39": toast_like(), "00/0000": b"unrelated",
    })
    fbx = proj / "Assets" / "EclatAndCrumb_Toast.fbx"
    fbx.write_bytes(b"; FBX")
    assert find_model_artifact(proj, fbx).name == "1b39"
    names = model_object_names(proj, fbx)
    MASK = 0x7FFF_FFFF_FFFF_FFFF
    assert names[-7185857084041579234 & MASK] == (4, "Armature_Toast")


def test_no_artifact_means_no_names(tmp_path: Path):
    proj = _project_with_artifacts(tmp_path, {"00/0000": b"unrelated"})
    fbx = proj / "Assets" / "Missing.fbx"
    fbx.write_bytes(b"; FBX")
    assert find_model_artifact(proj, fbx) is None
    assert model_object_names(proj, fbx) == {}


NEW_MOCHI = Path("D:/2files/unity_proj/New_Mochi")
TOAST_FBX = NEW_MOCHI / "Assets/sep-neko-ya/EclatAndCrumb/CafeItems/FBX/EclatAndCrumb_Toast.fbx"


@pytest.mark.skipif(not TOAST_FBX.is_file(), reason="kevin's New_Mochi project not present")
def test_kevins_toast_import_names_the_objects_his_scene_overrides():
    MASK = 0x7FFF_FFFF_FFFF_FFFF
    P2 = 1822428105269825218     # the Toast prefab's instance of its FBX
    names = model_object_names(NEW_MOCHI, TOAST_FBX)
    assert names[(363707147108875228 ^ P2) & MASK] == (4, "Armature_Toast")
    assert names[(6792985174677890633 ^ P2) & MASK] == (137, "EC_Toast")
