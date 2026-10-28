"""A tiny SKINNED ASCII FBX for tests that need real cluster weights,
normals and blend-shape deltas (the vertex bake)."""
from __future__ import annotations

from typing import Optional


def _arr(name: str, values, indent: str = "\t\t") -> str:
    vals = ",".join(repr(float(v)) if not isinstance(v, int) else str(v) for v in values)
    return f"{indent}{name}: *{len(values)} {{\n{indent}\ta: {vals}\n{indent}}}\n"


def _col_major(m: list[list[float]]) -> list[float]:
    return [m[r][c] for c in range(4) for r in range(4)]


def _ident_t(t) -> list[list[float]]:
    return [[1, 0, 0, t[0]], [0, 1, 0, t[1]], [0, 0, 1, t[2]], [0, 0, 0, 1]]


def skinned_fbx(
    bones: list[tuple],
    mesh: tuple,
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    weights: dict[int, list[tuple[int, float]]],
    *,
    normals: Optional[list[tuple[float, float, float]]] = None,
    shape: Optional[tuple[str, list[int], list[tuple[float, float, float]]]] = None,
    bone_worlds: Optional[dict[int, tuple[float, float, float]]] = None,
    extra_meshes: Optional[list[tuple]] = None,
) -> bytes:
    """bones: (id, name, parent_id, translation, rotation[, type_class]); mesh: (id, name,
    translation, rotation). weights: bone id -> [(vertex index, weight)].
    normals are ByPolygonVertex / Direct (one per triangle corner). shape:
    (name, indexes, deltas). bone_worlds: bind translation per bone (identity
    rotation) for TransformLink. extra_meshes: (id, name) meshes with one
    triangle and no skin."""
    objs: list[str] = []
    conns: list[str] = []
    for bone in bones:
        bid, name, parent, t, r = bone[:5]
        cls = bone[5] if len(bone) > 5 else "LimbNode"
        objs.append(
            f'\tModel: {bid}, "Model::{name}", "{cls}" {{\n\t\tVersion: 232\n'
            f'\t\tProperties70:  {{\n'
            f'\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",{t[0]},{t[1]},{t[2]}\n'
            f'\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",{r[0]},{r[1]},{r[2]}\n'
            f'\t\t}}\n\t}}\n'
        )
        conns.append(f'\tC: "OO",{bid},{parent}\n')
    mid, mname, mt, mr = mesh
    gid, skin = mid + 1, mid + 2
    objs.append(
        f'\tModel: {mid}, "Model::{mname}", "Mesh" {{\n\t\tVersion: 232\n'
        f'\t\tProperties70:  {{\n'
        f'\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",{mt[0]},{mt[1]},{mt[2]}\n'
        f'\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",{mr[0]},{mr[1]},{mr[2]}\n'
        f'\t\t}}\n\t}}\n'
    )
    conns.append(f'\tC: "OO",{mid},0\n')
    pvi: list[int] = []
    for a, b, c in triangles:
        pvi += [a, b, -c - 1]
    geo = f'\tGeometry: {gid}, "Geometry::{mname}", "Mesh" {{\n'
    geo += _arr("Vertices", [x for v in vertices for x in v])
    geo += _arr("PolygonVertexIndex", pvi)
    if normals is not None:
        geo += (
            '\t\tLayerElementNormal: 0 {\n\t\t\tVersion: 102\n\t\t\tName: ""\n'
            '\t\t\tMappingInformationType: "ByPolygonVertex"\n'
            '\t\t\tReferenceInformationType: "Direct"\n'
            + _arr("Normals", [x for n in normals for x in n], "\t\t\t")
            + "\t\t}\n"
        )
    geo += "\t}\n"
    objs.append(geo)
    conns.append(f'\tC: "OO",{gid},{mid}\n')
    objs.append(f'\tDeformer: {skin}, "Deformer::", "Skin" {{\n\t\tVersion: 101\n\t}}\n')
    conns.append(f'\tC: "OO",{skin},{gid}\n')
    bone_worlds = bone_worlds or {}
    for i, (bid, vw) in enumerate(sorted(weights.items())):
        cid = mid + 10 + i
        idx = [v for v, _ in vw]
        wts = [w for _, w in vw]
        tl = _ident_t(bone_worlds.get(bid, (0.0, 0.0, 0.0)))
        objs.append(
            f'\tDeformer: {cid}, "SubDeformer::", "Cluster" {{\n\t\tVersion: 100\n'
            + _arr("Indexes", idx) + _arr("Weights", wts)
            + _arr("Transform", _col_major(_ident_t((0, 0, 0))))
            + _arr("TransformLink", _col_major(tl))
            + "\t}\n"
        )
        conns.append(f'\tC: "OO",{cid},{skin}\n')
        conns.append(f'\tC: "OO",{bid},{cid}\n')
    if shape is not None:
        sname, sidx, sdel = shape
        bs, ch, sh = mid + 100, mid + 101, mid + 102
        objs.append(f'\tDeformer: {bs}, "Deformer::{mname}", "BlendShape" {{\n\t\tVersion: 100\n\t}}\n')
        objs.append(
            f'\tDeformer: {ch}, "SubDeformer::{sname}", "BlendShapeChannel" {{\n'
            f'\t\tVersion: 100\n\t\tDeformPercent: 0\n'
            + _arr("FullWeights", [100]) + "\t}\n"
        )
        objs.append(
            f'\tGeometry: {sh}, "Geometry::{sname}", "Shape" {{\n\t\tVersion: 100\n'
            + _arr("Indexes", sidx) + _arr("Vertices", [x for d in sdel for x in d]) + "\t}\n"
        )
        conns += [f'\tC: "OO",{bs},{gid}\n', f'\tC: "OO",{ch},{bs}\n', f'\tC: "OO",{sh},{ch}\n']
    for eid, ename in extra_meshes or []:
        objs.append(
            f'\tModel: {eid}, "Model::{ename}", "Mesh" {{\n\t\tVersion: 232\n\t}}\n'
            f'\tGeometry: {eid + 1}, "Geometry::{ename}", "Mesh" {{\n'
            + _arr("Vertices", [0, 0, 0, 1, 0, 0, 0, 1, 0]) + _arr("PolygonVertexIndex", [0, 1, -3])
            + "\t}\n"
        )
        conns += [f'\tC: "OO",{eid},0\n', f'\tC: "OO",{eid + 1},{eid}\n']
    return ("; FBX 7.4.0 project file\nObjects:  {\n" + "".join(objs) + "}\n"
            "Connections:  {\n" + "".join(conns) + "}\n").encode("utf-8")
