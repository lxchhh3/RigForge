"""Builders for tiny synthetic Unity projects (scene + prefabs + .meta files)
used by the scene-reader tests. Text is written the way Unity writes it."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

HEADER = "%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n"


def write_asset(project: Path, rel: str, text: str, guid: str,
                *, meta_extra: str = "") -> Path:
    p = project / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    Path(str(p) + ".meta").write_text(
        f"fileFormatVersion: 2\nguid: {guid}\n{meta_extra}", encoding="utf-8",
    )
    return p


def make_project(tmp_path: Path) -> Path:
    proj = tmp_path / "Proj"
    (proj / "Assets").mkdir(parents=True)
    (proj / "ProjectSettings").mkdir()
    (proj / "Packages").mkdir()
    return proj


def game_object(fid: int, name: str, components: list[int], *, active: int = 1,
                tag: str = "Untagged") -> str:
    comps = "".join(f"  - component: {{fileID: {c}}}\n" for c in components)
    return (
        f"--- !u!1 &{fid}\nGameObject:\n  m_ObjectHideFlags: 0\n"
        f"  m_Component:\n{comps}  m_Layer: 0\n  m_Name: {name}\n"
        f"  m_TagString: {tag}\n  m_IsActive: {active}\n"
    )


def transform(fid: int, go: int, father: int = 0, *,
              pos=(0, 0, 0), rot=(0, 0, 0, 1), scale=(1, 1, 1),
              children: Optional[list[int]] = None) -> str:
    kids = children or []
    kid_lines = "".join(f"  - {{fileID: {k}}}\n" for k in kids)
    kids_text = f"  m_Children:\n{kid_lines}" if kids else "  m_Children: []\n"
    return (
        f"--- !u!4 &{fid}\nTransform:\n  m_ObjectHideFlags: 0\n"
        f"  m_GameObject: {{fileID: {go}}}\n"
        f"  m_LocalRotation: {{x: {rot[0]}, y: {rot[1]}, z: {rot[2]}, w: {rot[3]}}}\n"
        f"  m_LocalPosition: {{x: {pos[0]}, y: {pos[1]}, z: {pos[2]}}}\n"
        f"  m_LocalScale: {{x: {scale[0]}, y: {scale[1]}, z: {scale[2]}}}\n"
        f"{kids_text}  m_Father: {{fileID: {father}}}\n"
    )


def skinned_mesh(fid: int, go: int, weights: list[float], *, mesh_guid: str = "",
                 mesh_fid: int = 4300000, enabled: int = 1) -> str:
    w = "".join(f"  - {x}\n" for x in weights) if weights else ""
    weights_text = f"  m_BlendShapeWeights:\n{w}" if weights else "  m_BlendShapeWeights: []\n"
    mesh = (f"{{fileID: {mesh_fid}, guid: {mesh_guid}, type: 3}}"
            if mesh_guid else "{fileID: 0}")
    return (
        f"--- !u!137 &{fid}\nSkinnedMeshRenderer:\n  m_ObjectHideFlags: 0\n"
        f"  m_GameObject: {{fileID: {go}}}\n  m_Enabled: {enabled}\n"
        f"  m_Mesh: {mesh}\n  m_Bones: []\n{weights_text}"
    )


def mono(fid: int, go: int, script_guid: str, body: str = "", *, enabled: int = 1) -> str:
    return (
        f"--- !u!114 &{fid}\nMonoBehaviour:\n  m_ObjectHideFlags: 0\n"
        f"  m_GameObject: {{fileID: {go}}}\n  m_Enabled: {enabled}\n"
        f"  m_Script: {{fileID: 11500000, guid: {script_guid}, type: 3}}\n"
        f"  m_Name: \n{body}"
    )


def stripped(cls_id: int, type_name: str, fid: int, src_fid: int, src_guid: str,
             instance: int) -> str:
    return (
        f"--- !u!{cls_id} &{fid} stripped\n{type_name}:\n"
        f"  m_CorrespondingSourceObject: {{fileID: {src_fid}, guid: {src_guid}, type: 3}}\n"
        f"  m_PrefabInstance: {{fileID: {instance}}}\n  m_PrefabAsset: {{fileID: 0}}\n"
    )


def prefab_instance(fid: int, src_guid: str, *, parent: int = 0,
                    mods: Optional[list[tuple]] = None,
                    removed_gos: Optional[list[int]] = None,
                    added_components: Optional[list[tuple[int, int]]] = None) -> str:
    """mods: (target fileID, propertyPath, value[, objectReference fileID])."""
    lines = []
    for m in mods or []:
        tgt, prop, val = m[0], m[1], m[2]
        ref = m[3] if len(m) > 3 else 0
        lines.append(
            f"    - target: {{fileID: {tgt}, guid: {src_guid}, type: 3}}\n"
            f"      propertyPath: {prop}\n      value: {val}\n"
            f"      objectReference: {{fileID: {ref}}}\n"
        )
    mods_text = ("    m_Modifications:\n" + "".join(lines)) if lines else "    m_Modifications: []\n"
    rem = removed_gos or []
    rem_text = (
        "    m_RemovedGameObjects:\n"
        + "".join(f"    - {{fileID: {r}, guid: {src_guid}, type: 3}}\n" for r in rem)
    ) if rem else "    m_RemovedGameObjects: []\n"
    add = added_components or []
    add_text = (
        "    m_AddedComponents:\n" + "".join(
            f"    - targetCorrespondingSourceObject: {{fileID: {t}, guid: {src_guid}, type: 3}}\n"
            f"      insertIndex: -1\n      addedObject: {{fileID: {a}}}\n"
            for t, a in add
        )
    ) if add else "    m_AddedComponents: []\n"
    return (
        f"--- !u!1001 &{fid}\nPrefabInstance:\n  m_ObjectHideFlags: 0\n"
        f"  m_Modification:\n    serializedVersion: 3\n"
        f"    m_TransformParent: {{fileID: {parent}}}\n{mods_text}"
        f"    m_RemovedComponents: []\n{rem_text}    m_AddedGameObjects: []\n{add_text}"
        f"  m_SourcePrefab: {{fileID: 100100000, guid: {src_guid}, type: 3}}\n"
    )


# Unity's fixed ids for an imported model's root GameObject / Transform.
MODEL_ROOT_GO = 919132149155446097
MODEL_ROOT_TRANSFORM = -8679921383154817045


def fbx_ascii(models: list[tuple], channels: Optional[dict[int, list[tuple[str, float]]]] = None) -> bytes:
    """A minimal ASCII FBX RigForge's sections parser reads.

    models: (id, name, type_class, parent_id, translation, rotation_deg) with
    type_class LimbNode / Null / Mesh; parent 0 = scene root. Each Mesh gets a
    one-triangle Geometry, and `channels[mesh_id]` = [(name, DeformPercent)]
    become one BlendShape with those channels, connected in list order."""
    channels = channels or {}
    objs: list[str] = []
    conns: list[str] = []
    for mid, name, cls, parent, t, r in models:
        objs.append(
            f'\tModel: {mid}, "Model::{name}", "{cls}" {{\n\t\tVersion: 232\n'
            f'\t\tProperties70:  {{\n'
            f'\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",{t[0]},{t[1]},{t[2]}\n'
            f'\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",{r[0]},{r[1]},{r[2]}\n'
            f'\t\t}}\n\t}}\n'
        )
        conns.append(f'\tC: "OO",{mid},{parent}\n')
        if cls != "Mesh":
            continue
        gid = mid + 100000
        objs.append(
            f'\tGeometry: {gid}, "Geometry::{name}", "Mesh" {{\n'
            f'\t\tVertices: *9 {{\n\t\t\ta: 0,0,0,0.1,0,0,0,0.1,0\n\t\t}}\n'
            f'\t\tPolygonVertexIndex: *3 {{\n\t\t\ta: 0,1,-3\n\t\t}}\n\t}}\n'
        )
        conns.append(f'\tC: "OO",{gid},{mid}\n')
        chans = channels.get(mid) or []
        if not chans:
            continue
        bid = mid + 200000
        objs.append(f'\tDeformer: {bid}, "Deformer::{name}", "BlendShape" {{\n\t\tVersion: 100\n\t}}\n')
        conns.append(f'\tC: "OO",{bid},{gid}\n')
        for i, (cname, pct) in enumerate(chans):
            cid = mid + 300000 + i
            sid = mid + 400000 + i
            objs.append(
                f'\tDeformer: {cid}, "SubDeformer::{cname}", "BlendShapeChannel" {{\n'
                f'\t\tVersion: 100\n\t\tDeformPercent: {pct}\n'
                f'\t\tFullWeights: *1 {{\n\t\t\ta: 100\n\t\t}}\n\t}}\n'
            )
            objs.append(
                f'\tGeometry: {sid}, "Geometry::{cname}", "Shape" {{\n\t\tVersion: 100\n'
                f'\t\tIndexes: *1 {{\n\t\t\ta: 2\n\t\t}}\n'
                f'\t\tVertices: *3 {{\n\t\t\ta: 0,0.05,0\n\t\t}}\n\t}}\n'
            )
            conns.append(f'\tC: "OO",{cid},{bid}\n')
            conns.append(f'\tC: "OO",{sid},{cid}\n')
    text = (
        "; FBX 7.4.0 project file\nObjects:  {\n" + "".join(objs) + "}\n"
        "Connections:  {\n" + "".join(conns) + "}\n"
    )
    return text.encode("utf-8")


def write_fbx(project: Path, rel: str, data: bytes, guid: str) -> Path:
    p = project / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    Path(str(p) + ".meta").write_text(
        f"fileFormatVersion: 2\nguid: {guid}\nModelImporter:\n  serializedVersion: 22200\n",
        encoding="utf-8",
    )
    return p
