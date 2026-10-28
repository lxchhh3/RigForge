"""Modular Avatar components, evaluated the way MA's build does.

Object Toggle (MA 1.18, Editor/ReactiveObjects/AnimationGeneration): a toggle
on GameObject `g` acts on each listed object `t` when every condition holds:

* the nearest MA Menu Item on `g` or above it (up to, not including, the
  avatar root) is in its default state ON (`isDefault`);
* every GameObject from `g` up to (not including) the avatar root that is
  NOT `t` itself or an ancestor of `t` is activeSelf.

`m_inverted` flips the result; when it holds, `t`'s active state becomes the
entry's `Active`. Toggles that switch each other are settled by iterating.

Bone Proxy: the object is moved under a humanoid bone (or a sub-path) at
build time; `attachmentMode` says what is kept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rigforge.unity.hierarchy import Graph, UObj, UnityProject
from rigforge.unity.yaml_docs import file_ref

OBJECT_TOGGLE = "ModularAvatarObjectToggle"
MENU_ITEM = "ModularAvatarMenuItem"
BONE_PROXY = "ModularAvatarBoneProxy"
SHAPE_CHANGER = "ModularAvatarShapeChanger"

# UnityEngine.HumanBodyBones, in enum order.
HUMAN_BODY_BONES = [
    "Hips", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg",
    "LeftFoot", "RightFoot", "Spine", "Chest", "Neck", "Head",
    "LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm",
    "LeftLowerArm", "RightLowerArm", "LeftHand", "RightHand",
    "LeftToes", "RightToes", "LeftEye", "RightEye", "Jaw",
    "LeftThumbProximal", "LeftThumbIntermediate", "LeftThumbDistal",
    "LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal",
    "LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
    "LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal",
    "LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal",
    "RightThumbProximal", "RightThumbIntermediate", "RightThumbDistal",
    "RightIndexProximal", "RightIndexIntermediate", "RightIndexDistal",
    "RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal",
    "RightRingProximal", "RightRingIntermediate", "RightRingDistal",
    "RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal",
    "UpperChest",
]
_PROXY_MODES = ["Unset", "AsChildAtRoot", "AsChildKeepWorldPose",
                "AsChildKeepRotation", "AsChildKeepPosition"]


def _monos(project: UnityProject, graph: Graph, go: int, script: str) -> list[UObj]:
    return [c for c in graph.components(go)
            if c.type_name == "MonoBehaviour" and project.script_name(c) == script]


def _enabled(c: UObj) -> bool:
    return str(c.data.get("m_Enabled", "1")) != "0"


def resolve_reference(graph: Graph, avatar_root: int, entry_object: dict) -> Optional[int]:
    """An MA AvatarObjectReference -> GameObject id: its targetObject when
    that resolves, else its referencePath from the avatar root."""
    tgt, _ = file_ref((entry_object or {}).get("targetObject"))
    if tgt in graph.objs:
        o = graph.objs[tgt]
        if o.type_name == "GameObject":
            return tgt
        go, _ = file_ref(o.data.get("m_GameObject"))
        if go in graph.objs:
            return go
    path = str((entry_object or {}).get("referencePath", "") or "")
    return find_by_path(graph, avatar_root, path) if path else None


def find_by_path(graph: Graph, root_go: int, path: str) -> Optional[int]:
    t = graph.transform_of(root_go)
    for part in path.split("/"):
        if t is None:
            return None
        nxt = [c for c in graph.children(t) if graph.name(graph.go_of(c)) == part]
        if len(nxt) != 1:
            return None
        t = nxt[0]
    return graph.go_of(t) if t is not None else None


@dataclass
class _Rule:
    toggle_go: int
    target: int
    active: bool
    inverted: bool


def effective_active(project: UnityProject, graph: Graph, *, avatar_root: int) -> dict[int, bool]:
    """activeSelf of every GameObject under `avatar_root` after MA Object
    Toggles (default menu states). Use `visible()` for activeInHierarchy."""
    under = _under(graph, avatar_root)
    state = {go: graph.active_self(go) for go in under}

    rules: list[_Rule] = []
    for go in under:
        for t in _monos(project, graph, go, OBJECT_TOGGLE):
            if not _enabled(t):
                continue
            inverted = str(t.data.get("m_inverted", "0")) == "1"
            for e in t.data.get("m_objects") or []:
                tgt = resolve_reference(graph, avatar_root, e.get("Object"))
                if tgt is None or tgt not in state:
                    continue
                rules.append(_Rule(go, tgt, str(e.get("Active", "0")) == "1", inverted))

    for _ in range(8):   # settle toggles that switch each other's holders
        nxt = {go: graph.active_self(go) for go in under}
        for r in rules:
            if _holds(project, graph, avatar_root, r, state) != r.inverted:
                nxt[r.target] = r.active
        if nxt == state:
            break
        state = nxt
    return state


def _holds(project: UnityProject, graph: Graph, avatar_root: int, r: _Rule,
           state: dict[int, bool]) -> bool:
    return _conditions_hold(project, graph, avatar_root, r.toggle_go, r.target, state)


def _conditions_hold(project: UnityProject, graph: Graph, avatar_root: int, holder: int,
                     affected: Optional[int], state: dict[int, bool]) -> bool:
    """MA BuildConditions: the nearest Menu Item at its default, and every
    GameObject from `holder` up to the avatar root active — except the
    affected object and its ancestors (None = no exception, as for shapes)."""
    target_and_above = {affected, *graph.ancestors(affected)} if affected is not None else set()
    cursor: Optional[int] = holder
    did_menu = False
    while cursor is not None and cursor != avatar_root:
        if not did_menu:
            menus = _monos(project, graph, cursor, MENU_ITEM)
            if menus:
                did_menu = True
                if str(menus[0].data.get("isDefault", "0")) != "1":
                    return False
        if cursor not in target_and_above and not state.get(cursor, graph.active_self(cursor)):
            return False
        anc = graph.ancestors(cursor)
        cursor = anc[0] if anc else None
    return True


@dataclass
class ShapeChange:
    renderer_go: int
    shape: str
    delete: bool          # ChangeType Delete: MA removes the vertices under the shape
    value: float          # ChangeType Set: the weight it sets
    changer_go: int


def shape_changes(project: UnityProject, graph: Graph, state: dict[int, bool], *,
                  avatar_root: int) -> list[ShapeChange]:
    """Active MA Shape Changer entries (default menu states), in hierarchy
    order — a later one wins for the same shape, as MA's action groups do."""
    out: list[ShapeChange] = []
    for go in _under(graph, avatar_root):
        for c in _monos(project, graph, go, SHAPE_CHANGER):
            if not _enabled(c):
                continue
            inverted = str(c.data.get("m_inverted", "0")) == "1"
            if _conditions_hold(project, graph, avatar_root, go, None, state) == inverted:
                continue
            for e in c.data.get("m_shapes") or []:
                tgt = resolve_reference(graph, avatar_root, e.get("Object"))
                if tgt is None:
                    continue
                try:
                    value = float(e.get("Value", 0) or 0)
                except (TypeError, ValueError):
                    value = 0.0
                out.append(ShapeChange(
                    renderer_go=tgt, shape=str(e.get("ShapeName", "") or ""),
                    delete=str(e.get("ChangeType", "1")) == "0", value=value, changer_go=go,
                ))
    return out


def visible(graph: Graph, state: dict[int, bool], go: int, *, avatar_root: int) -> bool:
    """activeInHierarchy under the toggled states, and not EditorOnly (VRChat
    strips EditorOnly-tagged objects at build)."""
    chain = [go, *graph.ancestors(go)]
    for g in chain:
        if not state.get(g, graph.active_self(g)):
            return False
        if graph.tag(g) == "EditorOnly":
            return False
        if g == avatar_root:
            break
    return True


def _under(graph: Graph, root_go: int) -> list[int]:
    t = graph.transform_of(root_go)
    if t is None:
        return [root_go]
    out = []
    for tid in graph.subtree(t):
        go = graph.go_of(tid)
        if go is not None:
            out.append(go)
    return out


@dataclass
class BoneProxy:
    go: int
    human_bone: Optional[str]     # HumanBodyBones name, None when unset
    sub_path: str                 # path below that bone ('' = the bone itself)
    mode: str                     # BoneProxyAttachmentMode name


def bone_proxies(project: UnityProject, graph: Graph) -> list[BoneProxy]:
    out: list[BoneProxy] = []
    for go in graph.game_objects():
        for c in _monos(project, graph, go, BONE_PROXY):
            if not _enabled(c):
                continue
            try:
                ref = int(c.data.get("boneReference", -1))
            except (TypeError, ValueError):
                ref = -1
            try:
                mode = int(c.data.get("attachmentMode", 0))
            except (TypeError, ValueError):
                mode = 0
            out.append(BoneProxy(
                go=go,
                human_bone=HUMAN_BODY_BONES[ref] if 0 <= ref < len(HUMAN_BODY_BONES) else None,
                sub_path=str(c.data.get("subPath", "") or ""),
                # MA builds an Unset proxy as AsChildAtRoot.
                mode=_PROXY_MODES[mode] if 0 < mode < len(_PROXY_MODES) else "AsChildAtRoot",
            ))
    return out


__all__ = [
    "BONE_PROXY",
    "BoneProxy",
    "HUMAN_BODY_BONES",
    "MENU_ITEM",
    "OBJECT_TOGGLE",
    "SHAPE_CHANGER",
    "ShapeChange",
    "bone_proxies",
    "shape_changes",
    "effective_active",
    "find_by_path",
    "resolve_reference",
    "visible",
]
