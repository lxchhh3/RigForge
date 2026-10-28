"""Bake a move on a KEPT clothing bone into the clothing's vertices.

In a name-based merge a clothing bone that matches the avatar ("kept") is
replaced by the avatar's own bone, so a move on it has nothing to land on. In
Unity the move still shows (kevin's Eclat_Coffee: the outfit's Head lifted
+2.8 / +4.0 cm carries its hair accessory). Modular Avatar keeps such an
offset by rewriting the mesh's bind pose — but Blender keeps ONE rest per bone
and took that bind as Maya's Head rest, moving her face 3.6 cm (measured
2026-10-28). So the offset goes into the vertices instead, the way moving them
in edit mode would:

    v_world' = v_world + sum_k w_k (M_k - I) v_world,   M_k = W'_k W_k^-1

for every kept bone k whose world changes — the moved bone and the kept
bones riding it (in Unity the outfit's own Eye rode on its moved Head). With a
rotation, normals and blend-shape deltas turn by the same blended linear part.
A ride-along (non-kept) child bone without its own move is re-placed so it
follows its moved parent; one with its own move keeps that move.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

from rigforge.ascii_fbx.edits import TextEdit, iter_oo_connections, set_model_local_transform_edits
from rigforge.ascii_fbx.lexer import FBXNode
from rigforge.ascii_fbx.sections import (
    SectionView,
    _local_matrix,
    _mat_mul,
    model_local_transform,
    model_world_matrix,
)
from rigforge.ascii_fbx.transform import LocalTransform, affine_inverse, affine_to_fbx_local

_I4 = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
_EPS = 1e-12
_TURN_EPS = 1e-4     # rotation-matrix element; ~0.006 deg
_WRAP = 64          # values per line when an array is rewritten


@dataclass
class BakeResult:
    edits: list[TextEdit] = field(default_factory=list)
    # Kept bones whose requested move this pass consumed (no "skip" note).
    baked_bone_ids: set[int] = field(default_factory=set)


# --- array access -------------------------------------------------------------


def _array_node(parent: FBXNode, name: str) -> Optional[FBXNode]:
    child = parent.child(name)
    return child.child("a") if child is not None else None


def _numbers(source: bytes, a: FBXNode) -> tuple[list[bytes], list[float]]:
    raw = a.args_bytes(source)
    toks = [t.strip() for t in raw.split(b",")]
    toks = [t for t in toks if t]
    return toks, [float(t) for t in toks]


def _rewrite(source: bytes, a: FBXNode, toks: list[bytes], values: list[float],
             changed: set[int]) -> TextEdit:
    """Replace an `a:` array's numbers, keeping untouched tokens as written."""
    out = [repr(values[i]).encode() if i in changed else toks[i] for i in range(len(toks))]
    lines = [b",".join(out[i:i + _WRAP]) for i in range(0, len(out), _WRAP)]
    body = b",\n".join(lines)
    s, e = a.args_span
    seg = source[s:e]
    lead = len(seg) - len(seg.lstrip())
    trail = len(seg) - len(seg.rstrip())
    return TextEdit(s + lead, e - trail, body)


def geometry_array(view: SectionView, name: str, field_name: str) -> list[float]:
    """A Geometry's numeric array by geometry name: `Vertices`, or `Normals`
    from its first LayerElementNormal (test / report helper)."""
    src = view.document.source
    objects = view.document.root("Objects")
    for node in objects.children:
        if node.name != "Geometry":
            continue
        head = node.args_bytes(src)
        if f'"Geometry::{name}"'.encode() not in head:
            continue
        if field_name == "Normals":
            layer = node.child("LayerElementNormal")
            a = _array_node(layer, "Normals") if layer is not None else None
        else:
            a = _array_node(node, field_name)
        if a is None:
            return []
        return _numbers(src, a)[1]
    raise KeyError(f"no Geometry named {name!r}")


# --- matrices -----------------------------------------------------------------


def _is_identity(m, eps: float = 1e-10) -> bool:
    return all(abs(m[i][j] - _I4[i][j]) <= eps for i in range(4) for j in range(4))


def _apply(m, p: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + m[i][3] for i in range(3))


def _lin(m, v) -> list[float]:
    return [m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] for i in range(3)]


def _unit(v) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


# --- the bake -----------------------------------------------------------------


def _parents_first(view: SectionView) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()

    def visit(mid: int, stack: frozenset) -> None:
        if mid in seen or mid in stack:
            return
        p = view.bones[mid].parent_id
        if p in view.bones:
            visit(p, stack | {mid})
        seen.add(mid)
        out.append(mid)

    for mid in view.bones:
        visit(mid, frozenset())
    return out


def _full_local(view: SectionView, mid: int, t: LocalTransform) -> LocalTransform:
    f = model_local_transform(view, mid)
    return LocalTransform(
        position=t.position if t.position is not None else f.position,
        rotation=t.rotation if t.rotation is not None else f.rotation,
        scale=t.scale if t.scale is not None else f.scale,
    )


def _mat(t: LocalTransform):
    return _local_matrix(t.position, t.rotation, t.scale)


def bake_kept_bone_moves(
    view: SectionView,
    moves: dict[int, LocalTransform],
    kept: set[int],
    *,
    skip_mesh_ids: Iterable[int] = (),
    skip_bone_ids: Iterable[int] = (),
    explicit_move_ids: Iterable[int] = (),
    notes: list[str],
) -> BakeResult:
    """`moves`: resolved FBX-frame moves by Model id (partial components
    allowed). Only moves on `kept` bones are baked; the rest are left to the
    normal bone-move pass."""
    own = {b: t for b, t in moves.items() if b in kept and b in view.bones
           and view.bones[b].type_class == "LimbNode"}
    if not own:
        return BakeResult()
    skip_mesh = set(skip_mesh_ids)
    skip_bone = set(skip_bone_ids)
    explicit = set(explicit_move_ids)

    worlds: dict[int, tuple] = {}

    def world(mid: Optional[int]):
        if mid is None or mid not in view.bones:
            return _I4
        if mid not in worlds:
            worlds[mid] = model_world_matrix(view, mid)
        return worlds[mid]

    moved: dict[int, tuple] = {}            # kept bone -> its world once moved
    for mid in _parents_first(view):
        b = view.bones[mid]
        if b.type_class != "LimbNode" or mid not in kept:
            continue
        parent_moved = moved.get(b.parent_id)
        if mid in own:
            local = _full_local(view, mid, own[mid])
        elif parent_moved is not None:
            local = model_local_transform(view, mid)
        else:
            continue
        base = parent_moved if parent_moved is not None else world(b.parent_id)
        moved[mid] = _mat_mul(base, _mat(local))

    deltas = {}
    for mid, wm in moved.items():
        d = _mat_mul(wm, affine_inverse(world(mid)))
        if not _is_identity(d):
            deltas[mid] = d
    res = BakeResult(baked_bone_ids=set(own))
    if not deltas:
        return res
    # A real turn, not the ~1e-7 that Unity's 7-digit numbers leave in a
    # translation-only move (rewriting normals/shapes for that is churn).
    rotates = any(
        abs(d[i][j] - _I4[i][j]) > _TURN_EPS for d in deltas.values() for i in range(3) for j in range(3)
    )

    # Ride-along children follow their moved parent.
    followers = []
    for b in view.bones.values():
        if (b.type_class == "LimbNode" and b.parent_id in deltas and b.model_id not in kept
                and b.model_id not in explicit and b.model_id not in skip_bone):
            pw, pm = world(b.parent_id), moved[b.parent_id]
            new_local = affine_to_fbx_local(_mat_mul(_mat_mul(affine_inverse(pw), pm),
                                                     _mat(model_local_transform(view, b.model_id))))
            res.edits.extend(set_model_local_transform_edits(
                view, b.model_id, translation=new_local.position,
                rotation_deg=new_local.rotation, scale=new_local.scale,
            ))
            followers.append(b.name)

    src = view.document.source
    conns = list(iter_oo_connections(view))
    geo_of = {m: g for g, m in view.geometry_owner_model.items()}
    bone_of_cluster = {c: b.model_id for b in view.bones.values() for c in b.cluster_ids}
    skins = set(view.cluster_skin.values())
    baked_meshes: list[str] = []
    n_verts = 0

    for m in view.bones_of_type("Mesh"):
        if m.model_id in skip_mesh or m.model_id not in geo_of:
            continue
        geo = geo_of[m.model_id]
        mesh_skins = {s for s, d in conns if d == geo and s in skins}
        acc: dict[int, list[list[float]]] = {}
        for cid, sk in view.cluster_skin.items():
            bone = bone_of_cluster.get(cid)
            if sk not in mesh_skins or bone not in deltas:
                continue
            node = view.clusters[cid].node_ref
            ia, wa = _array_node(node, "Indexes"), _array_node(node, "Weights")
            if ia is None or wa is None:
                continue
            idx = [int(x) for x in _numbers(src, ia)[1]]
            wts = _numbers(src, wa)[1]
            d = deltas[bone]
            dm = [[d[i][j] - _I4[i][j] for j in range(4)] for i in range(3)]
            for v, w in zip(idx, wts):
                if w == 0.0:
                    continue
                a = acc.setdefault(v, [[0.0] * 4 for _ in range(3)])
                for i in range(3):
                    row, src_row = a[i], dm[i]
                    for j in range(4):
                        row[j] += w * src_row[j]
        if not acc:
            continue

        geo_node = view.geometries[geo].node_ref
        mw = model_world_matrix(view, m.model_id)
        mwi = affine_inverse(mw)
        va = _array_node(geo_node, "Vertices")
        toks, vals = _numbers(src, va)
        changed: set[int] = set()
        for v, a in acc.items():
            k = 3 * v
            if k + 2 >= len(vals):
                continue
            p = _apply(mw, (vals[k], vals[k + 1], vals[k + 2]))
            moved_p = tuple(p[i] + a[i][0] * p[0] + a[i][1] * p[1] + a[i][2] * p[2] + a[i][3]
                            for i in range(3))
            q = _apply(mwi, moved_p)
            for i in range(3):
                if abs(q[i] - vals[k + i]) > _EPS:
                    vals[k + i] = q[i]
                    changed.add(k + i)
        if changed:
            res.edits.append(_rewrite(src, va, toks, vals, changed))
        n_verts += len(acc)
        baked_meshes.append(m.name)

        if rotates:
            res.edits.extend(_turn_normals(view, geo_node, acc, mw, mwi, notes, m.name))
            res.edits.extend(_turn_shapes(view, geo, conns, acc, mw, mwi))

    names = ", ".join(repr(view.bones[b].name) for b in own)
    msg = f"bake: {names} (merged into the target's bone) baked into {n_verts} vertices"
    msg += f" of {', '.join(baked_meshes)}" if baked_meshes else ""
    if followers:
        msg += f"; ride-along {', '.join(followers)} follow"
    notes.append(msg)
    return res


def _blend(a: list[list[float]]):
    """I + the accumulated linear part (3x3) of a vertex's delta."""
    return [[(1.0 if i == j else 0.0) + a[i][j] for j in range(3)] for i in range(3)]


def _turn_normals(view, geo_node, acc, mw, mwi, notes, mesh_name) -> list[TextEdit]:
    src = view.document.source
    out: list[TextEdit] = []
    pvi_node = _array_node(geo_node, "PolygonVertexIndex")
    corner_vertex: Optional[list[int]] = None
    for layer in geo_node.children:
        if layer.name != "LayerElementNormal":
            continue
        mapping = _string_child(src, layer, "MappingInformationType")
        reference = _string_child(src, layer, "ReferenceInformationType")
        if reference not in ("Direct", None):
            notes.append(f"bake: {mesh_name!r} normals are {reference}; not turned")
            continue
        na = _array_node(layer, "Normals")
        if na is None:
            continue
        toks, vals = _numbers(src, na)
        if mapping == "ByPolygonVertex":
            if corner_vertex is None and pvi_node is not None:
                corner_vertex = [int(x) if x >= 0 else -int(x) - 1 for x in _numbers(src, pvi_node)[1]]
            vert_of = corner_vertex or []
        elif mapping in ("ByVertice", "ByVertex"):
            vert_of = list(range(len(vals) // 3))
        else:
            notes.append(f"bake: {mesh_name!r} normals mapped {mapping}; not turned")
            continue
        changed: set[int] = set()
        for c, v in enumerate(vert_of):
            a = acc.get(v)
            if a is None or 3 * c + 2 >= len(vals):
                continue
            n = vals[3 * c:3 * c + 3]
            nw = _lin(mw, n)
            nb = _unit([sum(_blend(a)[i][j] * nw[j] for j in range(3)) for i in range(3)])
            nl = _unit(_lin(mwi, nb))
            for i in range(3):
                if abs(nl[i] - vals[3 * c + i]) > _EPS:
                    vals[3 * c + i] = nl[i]
                    changed.add(3 * c + i)
        if changed:
            out.append(_rewrite(src, na, toks, vals, changed))
    return out


def _turn_shapes(view, geo, conns, acc, mw, mwi) -> list[TextEdit]:
    """Blend-shape deltas live in mesh space: turn them with the vertex."""
    src = view.document.source
    out: list[TextEdit] = []
    bss = {s for s, d in conns if d == geo and s in view.blend_shapes}
    chans = {cid for cid, ch in view.blend_shape_channels.items() if ch.blend_shape_id in bss}
    shapes = [s for s, d in conns if d in chans and s in view.geometries]
    for sid in shapes:
        node = view.geometries[sid].node_ref
        ia, va = _array_node(node, "Indexes"), _array_node(node, "Vertices")
        if ia is None or va is None:
            continue
        idx = [int(x) for x in _numbers(src, ia)[1]]
        toks, vals = _numbers(src, va)
        changed: set[int] = set()
        for i, v in enumerate(idx):
            a = acc.get(v)
            if a is None:
                continue
            dw = _lin(mw, vals[3 * i:3 * i + 3])
            db = [sum(_blend(a)[r][c] * dw[c] for c in range(3)) for r in range(3)]
            dl = _lin(mwi, db)
            for k in range(3):
                if abs(dl[k] - vals[3 * i + k]) > _EPS:
                    vals[3 * i + k] = dl[k]
                    changed.add(3 * i + k)
        if changed:
            out.append(_rewrite(src, va, toks, vals, changed))
    return out


def _string_child(source: bytes, node: FBXNode, name: str) -> Optional[str]:
    c = node.child(name)
    if c is None:
        return None
    return c.args_bytes(source).strip().strip(b'"').decode("utf-8", "replace")


__all__ = ["BakeResult", "bake_kept_bone_moves", "geometry_array"]
