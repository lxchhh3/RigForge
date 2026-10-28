"""A Unity scene with every prefab instance expanded into one object graph.

Unity stores a placed prefab as a `PrefabInstance` (source asset + a list of
overrides) and writes only *stripped* placeholders for the objects it needs to
reference. To see what the Inspector shows, each instance is expanded:

* an object of the source prefab with fileID `s` gets id
  `(instance_id ^ s) & 0x7FFF_FFFF_FFFF_FFFF` in the containing file — the
  rule Unity itself uses for nested prefabs (`compose_id`);
* the instance's overrides (`propertyPath: value`, targeting the source's ids)
  are applied on top of the source values;
* the source's root Transform is parented under `m_TransformParent`;
* stripped placeholders become aliases of the expanded objects they stand for.

An imported model (FBX) is a prefab source too, but its objects live in the
importer's binary artifact, not in YAML. Only its root GameObject/Transform
(fixed ids) are synthesised; overrides on anything deeper are listed in
`Graph.unresolved` for the caller to report.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from rigforge.unity.yaml_docs import (
    UnityDoc,
    as_float,
    file_ref,
    parse_unity_yaml,
    set_property_path,
)

MASK = 0x7FFF_FFFF_FFFF_FFFF
# Unity's fixed fileIDs for an imported model's root GameObject / Transform.
MODEL_ROOT_GO = 919132149155446097
MODEL_ROOT_TRANSFORM = -8679921383154817045
_MODEL_SUFFIXES = {".fbx", ".obj", ".blend", ".dae", ".3ds", ".max", ".ma", ".mb"}

Mat4 = list[list[float]]


def compose_id(instance_id: int, inner_id: int) -> int:
    """The id an object of a nested prefab gets in the file that instances it."""
    return (instance_id ^ inner_id) & MASK


class UnityProject:
    """A Unity project on disk: guid -> asset path index + parsed-file cache."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._guid_to_path: Optional[dict[str, Path]] = None
        self._docs: dict[Path, list[UnityDoc]] = {}
        self._graphs: dict[Path, "Graph"] = {}

    @classmethod
    def for_asset(cls, asset: Path) -> "UnityProject":
        """The project an asset (a scene) belongs to: the nearest ancestor
        holding both Assets/ and ProjectSettings/."""
        p = Path(asset).resolve()
        for d in [p, *p.parents]:
            if (d / "Assets").is_dir() and (d / "ProjectSettings").is_dir():
                return cls(d)
        raise ValueError(f"{asset} is not inside a Unity project (no Assets/ + ProjectSettings/)")

    # --- guids ---------------------------------------------------------------

    _GUID = re.compile(r"^guid: ([0-9a-f]{32})", re.M)

    def _index(self) -> dict[str, Path]:
        if self._guid_to_path is None:
            idx: dict[str, Path] = {}
            for sub in ("Assets", "Packages", "Library/PackageCache"):
                base = self.root / sub
                if not base.is_dir():
                    continue
                for meta in base.rglob("*.meta"):
                    try:
                        with open(meta, "r", encoding="utf-8", errors="ignore") as fh:
                            head = fh.read(256)
                    except OSError:
                        continue
                    m = self._GUID.search(head)
                    if m and m.group(1) not in idx:
                        idx[m.group(1)] = Path(str(meta)[: -len(".meta")])
            self._guid_to_path = idx
        return self._guid_to_path

    def path_for_guid(self, guid: str) -> Optional[Path]:
        return self._index().get(guid)

    def script_name(self, mono: "UObj") -> Optional[str]:
        """The class of a MonoBehaviour = its script asset's file stem
        (ModularAvatarObjectToggle.cs -> ModularAvatarObjectToggle)."""
        _, guid = file_ref(mono.data.get("m_Script"))
        p = self.path_for_guid(guid) if guid else None
        return p.stem if p is not None else None

    # --- files ---------------------------------------------------------------

    def docs(self, path: Path) -> list[UnityDoc]:
        path = Path(path).resolve()
        if path not in self._docs:
            self._docs[path] = parse_unity_yaml(path.read_text(encoding="utf-8"))
        return self._docs[path]


@dataclass
class UObj:
    id: int
    type_name: str
    data: dict


@dataclass
class InnerOverride:
    """An override on an object INSIDE an imported model (its objects are not
    in YAML; `library.model_object_names` names them). `candidates` are
    (model-root GameObject id in this graph's space, model file, the inner
    fileID & MASK): one per model instance it could belong to."""
    candidates: list[tuple[int, Path, int]]
    prop: str
    value: object
    text: str


@dataclass
class Graph:
    """Every object of an expanded scene/prefab, keyed by id in its space."""
    objs: dict[int, UObj] = field(default_factory=dict)
    # Root GameObject id of each imported-model instance -> the model file.
    model_roots: dict[int, Path] = field(default_factory=dict)
    # Model-root GameObject id -> (model file, the id its inner objects are
    # composed with in this space: inner object = (that ^ inner) & MASK).
    model_instances: dict[int, tuple[Path, int]] = field(default_factory=dict)
    # Overrides on objects inside imported models, in Unity's order (deepest
    # prefab first, the scene last).
    inner_overrides: list[InnerOverride] = field(default_factory=list)
    # Overrides that target objects the source no longer has (stale).
    unresolved: list[str] = field(default_factory=list)
    # Prefab instances whose source asset is gone from the project:
    # (instance id, its name override or "", source guid).
    missing: list[tuple[int, str, str]] = field(default_factory=list)
    # Expanded-instance root GameObject -> the prefab/model asset it came from.
    instance_sources: dict[int, Path] = field(default_factory=dict)
    _children: Optional[dict[int, list[int]]] = None
    _by_go: Optional[dict[int, list[int]]] = None

    # --- indexes -------------------------------------------------------------

    def _build(self) -> None:
        children: dict[int, list[int]] = {}
        by_go: dict[int, list[int]] = {}
        for o in self.objs.values():
            go_id, _ = file_ref(o.data.get("m_GameObject"))
            if go_id:
                by_go.setdefault(go_id, []).append(o.id)
            if o.type_name in ("Transform", "RectTransform"):
                father, _ = file_ref(o.data.get("m_Father"))
                if father:
                    children.setdefault(father, []).append(o.id)
        self._children, self._by_go = children, by_go

    def components(self, go_id: int) -> list[UObj]:
        if self._by_go is None:
            self._build()
        return [self.objs[i] for i in self._by_go.get(go_id, []) if i in self.objs]

    def component(self, go_id: int, type_name: str) -> Optional[UObj]:
        for c in self.components(go_id):
            if c.type_name == type_name:
                return c
        return None

    def transform_of(self, go_id: int) -> Optional[int]:
        for c in self.components(go_id):
            if c.type_name in ("Transform", "RectTransform"):
                return c.id
        return None

    def go_of(self, transform_id: int) -> Optional[int]:
        t = self.objs.get(transform_id)
        if t is None:
            return None
        go, _ = file_ref(t.data.get("m_GameObject"))
        return go or None

    def parent(self, transform_id: int) -> Optional[int]:
        t = self.objs.get(transform_id)
        if t is None:
            return None
        father, _ = file_ref(t.data.get("m_Father"))
        return father if father in self.objs else None

    def children(self, transform_id: int) -> list[int]:
        if self._children is None:
            self._build()
        return [c for c in self._children.get(transform_id, []) if c in self.objs]

    def roots(self) -> list[int]:
        """Transforms with no parent (scene roots)."""
        return [o.id for o in self.objs.values()
                if o.type_name in ("Transform", "RectTransform") and self.parent(o.id) is None]

    def subtree(self, transform_id: int) -> Iterable[int]:
        """Pre-order, siblings in file order."""
        stack = [transform_id]
        while stack:
            t = stack.pop()
            yield t
            stack.extend(reversed(self.children(t)))

    # --- GameObjects ---------------------------------------------------------

    def game_objects(self) -> list[int]:
        return [o.id for o in self.objs.values() if o.type_name == "GameObject"]

    def name(self, go_id: int) -> str:
        o = self.objs.get(go_id)
        return str(o.data.get("m_Name", "")) if o is not None else ""

    def active_self(self, go_id: int) -> bool:
        o = self.objs.get(go_id)
        return o is not None and str(o.data.get("m_IsActive", "1")) != "0"

    def tag(self, go_id: int) -> str:
        o = self.objs.get(go_id)
        return str(o.data.get("m_TagString", "")) if o is not None else ""

    def find_go(self, name: str, *, missing_ok: bool = False) -> Optional[int]:
        hits = [g for g in self.game_objects() if self.name(g) == name]
        if len(hits) == 1:
            return hits[0]
        if not hits and missing_ok:
            return None
        raise KeyError(f"{len(hits)} GameObjects named {name!r}")

    def ancestors(self, go_id: int) -> list[int]:
        """GameObjects above `go_id`, nearest first."""
        out: list[int] = []
        t = self.transform_of(go_id)
        t = self.parent(t) if t is not None else None
        while t is not None:
            g = self.go_of(t)
            if g is not None:
                out.append(g)
            t = self.parent(t)
        return out

    def path(self, go_id: int, *, relative_to: Optional[int] = None) -> str:
        names = [self.name(go_id)]
        for a in self.ancestors(go_id):
            if a == relative_to:
                break
            names.append(self.name(a))
        return "/".join(reversed(names))

    def active_in_hierarchy(self, go_id: int) -> bool:
        return self.active_self(go_id) and all(self.active_self(a) for a in self.ancestors(go_id))

    # --- transforms (Unity frame) ------------------------------------------

    def local_position(self, transform_id: int) -> tuple[float, float, float]:
        d = self.objs[transform_id].data.get("m_LocalPosition") or {}
        return (as_float(d.get("x")), as_float(d.get("y")), as_float(d.get("z")))

    def local_rotation(self, transform_id: int) -> tuple[float, float, float, float]:
        """Quaternion (x, y, z, w)."""
        d = self.objs[transform_id].data.get("m_LocalRotation") or {}
        return (as_float(d.get("x")), as_float(d.get("y")), as_float(d.get("z")),
                as_float(d.get("w"), 1.0))

    def local_scale(self, transform_id: int) -> tuple[float, float, float]:
        d = self.objs[transform_id].data.get("m_LocalScale") or {}
        return (as_float(d.get("x"), 1.0), as_float(d.get("y"), 1.0), as_float(d.get("z"), 1.0))

    def local_matrix(self, transform_id: int) -> Mat4:
        return trs_matrix(self.local_position(transform_id),
                          self.local_rotation(transform_id),
                          self.local_scale(transform_id))

    def world_matrix(self, transform_id: int) -> Mat4:
        m = self.local_matrix(transform_id)
        p = self.parent(transform_id)
        while p is not None:
            m = mat_mul(self.local_matrix(p), m)
            p = self.parent(p)
        return m


# --- matrices -----------------------------------------------------------------


def quat_to_matrix3(q: tuple[float, float, float, float]) -> list[list[float]]:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def trs_matrix(pos, quat, scale) -> Mat4:
    r = quat_to_matrix3(quat)
    return [
        [r[0][0] * scale[0], r[0][1] * scale[1], r[0][2] * scale[2], pos[0]],
        [r[1][0] * scale[0], r[1][1] * scale[1], r[1][2] * scale[2], pos[1]],
        [r[2][0] * scale[0], r[2][1] * scale[1], r[2][2] * scale[2], pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat_mul(a: Mat4, b: Mat4) -> Mat4:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


# --- expansion ----------------------------------------------------------------


def _remap_refs(node, mapping) -> None:
    """Rewrite local object references ({fileID: N} with no guid) in place."""
    if isinstance(node, dict):
        if "fileID" in node and not node.get("guid"):
            try:
                fid = int(node["fileID"])
            except (TypeError, ValueError):
                fid = 0
            if fid:
                new = mapping(fid)
                if new is not None:
                    node["fileID"] = str(new)
            return
        for v in node.values():
            _remap_refs(v, mapping)
    elif isinstance(node, list):
        for v in node:
            _remap_refs(v, mapping)


def _model_graph(path: Path) -> Graph:
    """An imported model as a prefab source: just its root GameObject +
    Transform (named after the file, as Unity names an imported model)."""
    g = Graph()
    g.objs[MODEL_ROOT_GO] = UObj(MODEL_ROOT_GO, "GameObject", {
        "m_Name": path.stem, "m_IsActive": "1", "m_TagString": "Untagged",
    })
    g.objs[MODEL_ROOT_TRANSFORM] = UObj(MODEL_ROOT_TRANSFORM, "Transform", {
        "m_GameObject": {"fileID": str(MODEL_ROOT_GO)},
        "m_LocalPosition": {"x": "0", "y": "0", "z": "0"},
        "m_LocalRotation": {"x": "0", "y": "0", "z": "0", "w": "1"},
        "m_LocalScale": {"x": "1", "y": "1", "z": "1"},
        "m_Father": {"fileID": "0"},
    })
    g.model_roots[MODEL_ROOT_GO] = path
    # Inside its own import a model's objects use their raw ids: compose with 0.
    g.model_instances[MODEL_ROOT_GO] = (path, 0)
    return g


def expand(project: UnityProject, path: Path, *, _stack: tuple = ()) -> Graph:
    """Expand a scene or prefab file (see module doc). Prefab graphs are
    cached per project; the caller gets a private deep copy."""
    path = Path(path).resolve()
    if path in _stack:
        raise ValueError(f"prefab cycle through {path}")
    cached = project._graphs.get(path)
    if cached is None:
        cached = _expand_file(project, path, _stack + (path,))
        project._graphs[path] = cached
    return copy.deepcopy(cached)


def _source_graph(project: UnityProject, guid: str, stack: tuple) -> tuple[Optional[Graph], Optional[Path]]:
    src = project.path_for_guid(guid)
    if src is None:
        return None, None
    if src.suffix.lower() in _MODEL_SUFFIXES:
        return _model_graph(src), src
    if src.suffix.lower() == ".prefab":
        return expand(project, src, _stack=stack), src
    return None, src


def _expand_file(project: UnityProject, path: Path, stack: tuple) -> Graph:
    docs = project.docs(path)
    out = Graph()
    aliases: dict[int, int] = {}
    own_ids: set[int] = set()

    for d in docs:
        if d.stripped:
            src_fid, _ = file_ref(d.data.get("m_CorrespondingSourceObject"))
            inst, _ = file_ref(d.data.get("m_PrefabInstance"))
            aliases[d.file_id] = compose_id(inst, src_fid)
        elif d.type_name != "PrefabInstance":
            out.objs[d.file_id] = UObj(d.file_id, d.type_name, copy.deepcopy(d.data))
            own_ids.add(d.file_id)

    for d in docs:
        if d.stripped or d.type_name != "PrefabInstance":
            continue
        _expand_instance(project, d, out, stack)

    # Stripped placeholders -> the expanded objects they stand for, in this
    # file's own objects (inner objects were remapped during expansion).
    def _alias(fid: int) -> Optional[int]:
        return aliases.get(fid)

    for oid in own_ids:
        _remap_refs(out.objs[oid].data, _alias)
    for o in out.objs.values():
        # m_Father / m_GameObject written by an override (objectReference) or
        # by m_TransformParent may also name a placeholder.
        for key in ("m_Father", "m_GameObject"):
            ref = o.data.get(key)
            if isinstance(ref, dict):
                _remap_refs(ref, _alias)
    out._children = out._by_go = None
    return out


def _expand_instance(project: UnityProject, d: UnityDoc, out: Graph, stack: tuple) -> None:
    mod = d.data.get("m_Modification") or {}
    _, src_guid = file_ref(d.data.get("m_SourcePrefab"))
    inner, src_path = _source_graph(project, src_guid, stack)
    if inner is None:
        if src_path is None:
            # Deleted from the project: Unity shows a "Missing Prefab" too.
            name = next((str(m.get("value", "")) for m in mod.get("m_Modifications") or []
                         if m.get("propertyPath") == "m_Name"), "")
            out.missing.append((d.file_id, name, src_guid))
        else:
            out.unresolved.append(
                f"prefab instance {d.file_id}: source {src_path.name} is not a prefab or model"
            )
        return
    inst = d.file_id

    def _map(fid: int) -> Optional[int]:
        return compose_id(inst, fid) if fid in inner.objs else None

    remapped: dict[int, UObj] = {}
    for iid, o in inner.objs.items():
        _remap_refs(o.data, _map)
        nid = compose_id(inst, iid)
        remapped[nid] = UObj(nid, o.type_name, o.data)

    # The source's own inner-model overrides come first (Unity applies the
    # deeper prefab's, then this instance's on top).
    for io in inner.inner_overrides:
        out.inner_overrides.append(InnerOverride(
            [(compose_id(inst, go), p, i) for go, p, i in io.candidates],
            io.prop, io.value, io.text,
        ))
    out.unresolved.extend(inner.unresolved)
    out.missing.extend((compose_id(inst, i), n, g) for i, n, g in inner.missing)

    # Overrides (targets are in the source's id space).
    for m in mod.get("m_Modifications") or []:
        tid, tguid = file_ref(m.get("target"))
        prop = str(m.get("propertyPath", ""))
        obj = remapped.get(compose_id(inst, tid)) if tguid == src_guid else None
        ref_id, ref_guid = file_ref(m.get("objectReference"))
        if ref_id:
            value = {"fileID": str(ref_id)}
            if ref_guid:
                value["guid"] = ref_guid
        else:
            value = m.get("value", "")
            value = "" if value is None else value
        if obj is None:
            text = (f"{src_path.name}: override {prop!r} on object {tid} "
                    f"(inside an imported model, or unknown)")
            cands = [
                (compose_id(inst, go), p, (tid ^ p2) & MASK)
                for go, (p, p2) in inner.model_instances.items()
            ] if tguid == src_guid else []
            if cands:
                out.inner_overrides.append(InnerOverride(cands, prop, value, text))
            else:
                out.unresolved.append(text)
            continue
        try:
            set_property_path(obj.data, prop, value)
        except ValueError as exc:
            out.unresolved.append(f"{src_path.name}: override {prop!r}: {exc}")

    for key in ("m_RemovedGameObjects", "m_RemovedComponents"):
        for r in mod.get(key) or []:
            rid, _ = file_ref(r)
            gone = compose_id(inst, rid)
            remapped.pop(gone, None)
            if key == "m_RemovedGameObjects":
                # A removed GameObject takes its components and its subtree.
                _drop_go_subtree(remapped, gone)

    # The source's root Transform hangs under m_TransformParent.
    parent, _ = file_ref(mod.get("m_TransformParent"))
    root_go: Optional[int] = None
    for o in remapped.values():
        if o.type_name in ("Transform", "RectTransform"):
            father, _ = file_ref(o.data.get("m_Father"))
            if not father or father not in remapped:
                o.data["m_Father"] = {"fileID": str(parent)}
                root_go, _ = file_ref(o.data.get("m_GameObject"))
    out.objs.update(remapped)
    for go, mp in inner.model_roots.items():
        out.model_roots[compose_id(inst, go)] = mp
    for go, (mp, p2) in inner.model_instances.items():
        out.model_instances[compose_id(inst, go)] = (mp, compose_id(inst, p2))
    for go, sp in inner.instance_sources.items():
        out.instance_sources[compose_id(inst, go)] = sp
    if root_go:
        out.instance_sources[root_go] = src_path


def _drop_go_subtree(objs: dict[int, UObj], go_id: int) -> None:
    doomed_go = {go_id}
    changed = True
    while changed:
        changed = False
        for o in list(objs.values()):
            if o.type_name not in ("Transform", "RectTransform"):
                continue
            g, _ = file_ref(o.data.get("m_GameObject"))
            father, _ = file_ref(o.data.get("m_Father"))
            fobj = objs.get(father)
            if fobj is not None:
                fg, _ = file_ref(fobj.data.get("m_GameObject"))
                if fg in doomed_go and g not in doomed_go:
                    doomed_go.add(g)
                    changed = True
    for o in list(objs.values()):
        g, _ = file_ref(o.data.get("m_GameObject"))
        if o.id in doomed_go or g in doomed_go:
            objs.pop(o.id, None)


__all__ = [
    "Graph",
    "InnerOverride",
    "MASK",
    "MODEL_ROOT_GO",
    "MODEL_ROOT_TRANSFORM",
    "UObj",
    "UnityProject",
    "compose_id",
    "expand",
    "mat_mul",
    "quat_to_matrix3",
    "trs_matrix",
]
