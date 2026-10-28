"""Unity YAML (scene / prefab) parsing — the text layer under the scene reader."""
from __future__ import annotations

import pytest

from rigforge.unity.yaml_docs import (
    UnityYamlError,
    parse_unity_yaml,
    set_property_path,
)

SCENE_TEXT = """\
%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!1 &100
GameObject:
  m_ObjectHideFlags: 0
  m_Component:
  - component: {fileID: 101}
  m_Name: On
  m_IsActive: 1
--- !u!4 &101
Transform:
  m_GameObject: {fileID: 100}
  m_LocalRotation: {x: 0, y: 0, z: 0, w: 1}
  m_LocalPosition: {x: 0, y: 1, z: -10}
  m_Father: {fileID: 0}
--- !u!1 &286372825 stripped
GameObject:
  m_CorrespondingSourceObject: {fileID: 3724182922056132471, guid: 015a065ba5f8d1547ac99c23be6cc491,
    type: 3}
  m_PrefabInstance: {fileID: 286372823}
--- !u!1001 &286372823
PrefabInstance:
  m_Modification:
    m_TransformParent: {fileID: 101}
    m_Modifications:
    - target: {fileID: 254216598253524121, guid: 015a065ba5f8d1547ac99c23be6cc491,
        type: 3}
      propertyPath: m_BlendShapeWeights.Array.data[2]
      value: 84
      objectReference: {fileID: 0}
    - target: {fileID: 2659936330211088980, guid: 015a065ba5f8d1547ac99c23be6cc491,
        type: 3}
      propertyPath: m_Name
      value: 001
      objectReference: {fileID: 0}
  m_SourcePrefab: {fileID: 100100000, guid: 015a065ba5f8d1547ac99c23be6cc491, type: 3}
--- !u!1660057539 &9223372036854775807
SceneRoots:
  m_Roots:
  - {fileID: 101}
"""


def test_documents_carry_class_id_file_id_and_stripped_flag():
    docs = parse_unity_yaml(SCENE_TEXT)
    by_id = {d.file_id: d for d in docs}
    assert by_id[100].class_id == 1 and by_id[100].type_name == "GameObject"
    assert by_id[101].type_name == "Transform"
    assert by_id[286372825].stripped is True
    assert by_id[100].stripped is False
    assert by_id[286372823].type_name == "PrefabInstance"
    # The largest fileID Unity writes (SceneRoots) survives as an int.
    assert by_id[9223372036854775807].type_name == "SceneRoots"


def test_scalars_stay_strings_so_names_are_never_coerced():
    """YAML 1.1 would read a GameObject named On as True and 001 as 1."""
    docs = {d.file_id: d for d in parse_unity_yaml(SCENE_TEXT)}
    assert docs[100].data["m_Name"] == "On"
    mods = docs[286372823].data["m_Modification"]["m_Modifications"]
    assert mods[1]["value"] == "001"


def test_flow_maps_wrapped_over_two_lines_parse_whole():
    docs = {d.file_id: d for d in parse_unity_yaml(SCENE_TEXT)}
    src = docs[286372825].data["m_CorrespondingSourceObject"]
    assert src == {
        "fileID": "3724182922056132471",
        "guid": "015a065ba5f8d1547ac99c23be6cc491",
        "type": "3",
    }


def test_negative_file_ids_parse():
    text = "--- !u!4 &-8679921383154817045 stripped\nTransform:\n  m_PrefabInstance: {fileID: 5}\n"
    (doc,) = parse_unity_yaml(text)
    assert doc.file_id == -8679921383154817045 and doc.stripped


def test_text_without_unity_documents_is_an_error():
    with pytest.raises(UnityYamlError):
        parse_unity_yaml("just: yaml\n")


# --- property paths (how a prefab override names the field it sets) ---------


def test_set_dotted_field():
    data = {"m_LocalPosition": {"x": "0", "y": "0", "z": "0"}}
    set_property_path(data, "m_LocalPosition.y", "0.07")
    assert data["m_LocalPosition"] == {"x": "0", "y": "0.07", "z": "0"}


def test_set_array_element_extends_the_list():
    data = {"m_BlendShapeWeights": ["0", "0"]}
    set_property_path(data, "m_BlendShapeWeights.Array.data[4]", "100")
    assert data["m_BlendShapeWeights"] == ["0", "0", "0", "0", "100"]


def test_set_field_inside_an_array_element():
    data = {"m_objects": [{"Object": {"referencePath": "a"}, "Active": "0"}]}
    set_property_path(data, "m_objects.Array.data[0].Object.referencePath", "Armature/b")
    assert data["m_objects"][0]["Object"]["referencePath"] == "Armature/b"


def test_array_size_resizes():
    data = {"m_Materials": ["a", "b", "c"]}
    set_property_path(data, "m_Materials.Array.size", "1")
    assert data["m_Materials"] == ["a"]
    set_property_path(data, "m_Materials.Array.size", "3")
    assert data["m_Materials"] == ["a", None, None]


def test_set_creates_missing_parents():
    data: dict = {}
    set_property_path(data, "m_LocalScale.x", "2")
    assert data == {"m_LocalScale": {"x": "2"}}
