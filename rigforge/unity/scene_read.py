"""A saved Unity scene -> a RigForge compose pre-fill.

kevin sets an avatar up in Unity (outfits and props on Maya, meshes switched
off with Modular Avatar toggles, blendshape sliders, bones nudged) and should
not have to re-enter any of it in the Compose page. `read_scene` reads the
scene and returns, in the ids the FE's inspect payloads use:

* the target avatar (the registry avatar whose bones the scene is built on)
  and its hidden meshes / changed blendshapes / moved bones;
* one clothing per garment FBX under the avatar, with the same three, and
  for a rigid prop the bone it hangs from and its placement;
* what could not be carried, and why (`notes`, `skipped`).

Values are what the build would ship: MA Object Toggles at their default
menu state, inactive / EditorOnly objects hidden, blendshape weights =
prefab values with the scene's overrides, Bone Proxies snapped to their bone.

Bones are compared by WORLD placement, not by local numbers: a MochiFitter
"finalized" prefab is not an instance of its FBX — Unity folded the FBX's
-90 X Armature into the top transforms and renamed `Hips_MFBase` to `Hips`
— so locals differ at the root while every bone sits where the FBX puts it.
A move is emitted when the local the scene implies, in the FBX's own parent
frame, differs from the file's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from rigforge.ascii_fbx.edits import iter_oo_connections
from rigforge.ascii_fbx.sections import SectionView, bone_to_mesh_names, model_world_matrix
from rigforge.ascii_fbx.transform import (
    LocalTransform,
    affine_inverse,
    affine_to_fbx_local,
    fbx_to_unity,
)
from rigforge.avatars.fingerprint import normalize_bone_name, strip_rig_suffix
from rigforge.avatars.registry import (
    AvatarRegistry,
    DonorIdentificationError,
    identify_donor,
)
from rigforge.pipeline.phase_a import is_self_rigged
from rigforge.unity.hierarchy import Graph, UnityProject, expand, mat_mul, trs_matrix
from rigforge.unity.library import model_object_names
from rigforge.unity.modular_avatar import (
    BoneProxy,
    ShapeChange,
    bone_proxies,
    effective_active,
    shape_changes,
    visible,
)
from rigforge.unity.yaml_docs import as_float, file_ref

Mat4 = list[list[float]]

# A Unity avatar hierarchy "is" a registry avatar when it holds this share of
# that avatar's bone names (containment, not Jaccard: a Unity prefab adds
# its own objects — colliders, the root, props — that the FBX never had).
AVATAR_CONTAINMENT = 0.9
POS_TOL = 1e-5        # m
ROT_TOL = 1e-4        # rotation-matrix element
SCALE_TOL = 1e-5
WEIGHT_TOL = 1e-3     # blendshape percent
# More than this share of bones "moved" = the frames do not line up; refuse
# rather than emit a move for every bone.
MAX_MOVED_SHARE = 0.5
PLACEMENT_NOTE_M = 0.005

# HumanBodyBones name -> RigForge canonical role (Bone Proxy targets).
_HUMAN_TO_ROLE = {
    "Hips": "Hips", "Spine": "Spine", "Chest": "Chest", "Neck": "Neck", "Head": "Head",
    "LeftShoulder": "Shoulder.L", "RightShoulder": "Shoulder.R",
    "LeftUpperArm": "UpperArm.L", "RightUpperArm": "UpperArm.R",
    "LeftLowerArm": "LowerArm.L", "RightLowerArm": "LowerArm.R",
    "LeftHand": "Hand.L", "RightHand": "Hand.R",
    "LeftUpperLeg": "UpperLeg.L", "RightUpperLeg": "UpperLeg.R",
    "LeftLowerLeg": "LowerLeg.L", "RightLowerLeg": "LowerLeg.R",
    "LeftFoot": "Foot.L", "RightFoot": "Foot.R",
    "LeftToes": "Toes.L", "RightToes": "Toes.R",
}
for _side, _s in (("Left", "L"), ("Right", "R")):
    for _f in ("Thumb", "Index", "Middle", "Ring", "Little"):
        for _i, _seg in enumerate(("Proximal", "Intermediate", "Distal"), start=1):
            _HUMAN_TO_ROLE[f"{_side}{_f}{_seg}"] = f"{_f}{_i}.{_s}"


class SceneReadError(RuntimeError):
    """The scene cannot be read into a compose (no avatar, frames disagree)."""


@dataclass
class SourceRead:
    """One side's pre-fill: the target, or one clothing."""
    drop_mesh_ids: list[int] = field(default_factory=list)
    channel_overrides: dict[int, float] = field(default_factory=dict)
    bone_transforms: list[dict] = field(default_factory=list)

    def _base_dict(self) -> dict:
        return {
            "drop_mesh_ids": list(self.drop_mesh_ids),
            "blend_shape_channel_overrides": [
                {"channel_id": cid, "deform_percent": v}
                for cid, v in self.channel_overrides.items()
            ],
            "bone_transforms": [dict(m) for m in self.bone_transforms],
        }


@dataclass
class ClothingRead(SourceRead):
    clothing_path: str = ""
    source: str = ""                  # the Unity object it came from
    rigid: bool = False
    self_rigged: bool = False         # its own armature, no bone of the target
    attach_to_bone_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "clothing_path": self.clothing_path,
            "source": self.source,
            "rigid": self.rigid,
            "self_rigged": self.self_rigged,
            "attach_to_bone_id": self.attach_to_bone_id,
            **self._base_dict(),
        }


@dataclass
class SceneRead:
    scene: str
    target_id: str
    target: SourceRead
    clothings: list[ClothingRead]
    skipped: list[dict]
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "scene": self.scene,
            "target_id": self.target_id,
            "target": self.target._base_dict(),
            "clothings": [c.to_dict() for c in self.clothings],
            "skipped": [dict(s) for s in self.skipped],
            "notes": list(self.notes),
        }


# --- small matrix helpers (Unity <-> FBX frame) -------------------------------

_MIRROR = [[-1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]


def _mirror(m: Mat4) -> list[list[float]]:
    """Unity-frame matrix <-> FBX-frame matrix (X mirror, both ways)."""
    return mat_mul(_MIRROR, mat_mul([list(r) for r in m], _MIRROR))


def _t(m) -> list[list[float]]:
    return [list(r) for r in m]


def _close(a: LocalTransform, b: LocalTransform) -> bool:
    from rigforge.ascii_fbx.transform import fbx_euler_to_matrix
    if max(abs(x - y) for x, y in zip(a.position, b.position)) > POS_TOL:
        return False
    if max(abs(x - y) for x, y in zip(a.scale, b.scale)) > SCALE_TOL:
        return False
    ra, rb = fbx_euler_to_matrix(*a.rotation), fbx_euler_to_matrix(*b.rotation)
    return max(abs(ra[i][j] - rb[i][j]) for i in range(3) for j in range(3)) <= ROT_TOL


def _local_fbx(view: SectionView, mid: int) -> LocalTransform:
    from rigforge.ascii_fbx.sections import model_local_transform
    return model_local_transform(view, mid)


def _lt_matrix(t: LocalTransform) -> list[list[float]]:
    from rigforge.ascii_fbx.sections import _local_matrix
    return _t(_local_matrix(t.position, t.rotation, t.scale))


def _unity_dict(t_fbx: LocalTransform) -> dict:
    u = fbx_to_unity(t_fbx)
    return {"position": [float(x) for x in u.position],
            "rotation": [float(x) for x in u.rotation],
            "scale": [float(x) for x in u.scale]}


def _cm(v: tuple) -> str:
    return "(" + ", ".join(f"{x * 100:+.1f}" for x in v) + ") cm"


# --- FBX-side lookups ---------------------------------------------------------


def mesh_channels(view: SectionView, mesh_id: int) -> list[int]:
    """A mesh's blendshape channel ids in the order Unity numbers them
    (`m_BlendShapeWeights` index): BlendShape deformers on its geometry, then
    their channels, each in FBX connection order."""
    geo = next((g for g, m in view.geometry_owner_model.items() if m == mesh_id), None)
    if geo is None:
        return []
    conns = list(iter_oo_connections(view))
    out: list[int] = []
    for s, d in conns:
        if d == geo and s in view.blend_shapes:
            out.extend(cs for cs, cd in conns if cd == s and cs in view.blend_shape_channels)
    return out


def deform_percent(view: SectionView, channel_id: int) -> float:
    ch = view.blend_shape_channels[channel_id]
    for child in ch.node_ref.children:
        if child.name == "DeformPercent":
            raw = child.args_bytes(view.document.source).strip()
            try:
                return float(raw.split()[0]) if raw else 0.0
            except (ValueError, IndexError):
                return 0.0
    return 0.0


# --- the read -----------------------------------------------------------------


_PROJECTS: dict[Path, UnityProject] = {}


def _project_for(scene: Path) -> UnityProject:
    proj = UnityProject.for_asset(scene)
    cached = _PROJECTS.get(proj.root)
    if cached is None:
        _PROJECTS[proj.root] = proj
        return proj
    # Re-read files and the guid index each call: kevin saves in Unity
    # between reads, and MochiFitter writes new prefabs.
    cached._docs.clear()
    cached._graphs.clear()
    cached._guid_to_path = None
    return cached


def read_scene(
    scene: Path,
    *,
    registry: AvatarRegistry,
    load_view: Callable[[Path], SectionView],
) -> SceneRead:
    scene = Path(scene)
    if not scene.is_file():
        raise SceneReadError(f"scene not found: {scene}")
    try:
        project = _project_for(scene)
    except ValueError as exc:
        raise SceneReadError(str(exc)) from exc
    graph = expand(project, scene)
    notes: list[str] = []
    skipped: list[dict] = []

    avatar_go, target_id = _find_avatar(graph, registry)
    av = registry.get(target_id)
    target_view = av.load_ascii_view()
    state = effective_active(project, graph, avatar_root=avatar_go)
    changes: dict[int, list[ShapeChange]] = {}
    for ch in shape_changes(project, graph, state, avatar_root=avatar_go):
        changes.setdefault(ch.renderer_go, []).append(ch)

    garment_roots = _garment_roots(graph, avatar_go)
    garment_gos: set[int] = set()
    for r in garment_roots:
        garment_gos.update(_subtree_gos(graph, r))
    avatar_gos = [g for g in _subtree_gos(graph, avatar_go) if g not in garment_gos]

    # Garments: one job per FBX of each prefab/model instance under the
    # avatar, plus each extra FBX inside the avatar's OWN hierarchy (a
    # vendor's avatar+outfit prefab, Chrome_Maya_LilToon: two FBXs' meshes
    # on one armature). Those embedded ones share the avatar's bones.
    jobs: list[tuple[str, int, Path, list[int], bool]] = []
    for root in garment_roots:
        unbound: dict[Path, list[int]] = {}
        for fbx, gos in _fbx_groups(project, graph, root, unbound).items():
            jobs.append((graph.name(root), root, fbx, gos, False))
        for asset, gos in unbound.items():
            if any(visible(graph, state, g, avatar_root=avatar_go) for g in gos):
                skipped.append({
                    "name": graph.name(root), "path": str(asset),
                    "reason": (f"its meshes come from {asset.name} with no FBX of that name "
                               f"beside it (the export was deleted?), so there is nothing to "
                               f"merge; add its FBX on the page by hand"),
                })
    target_gos, embedded = _split_embedded(project, graph, avatar_gos, target_view)
    plain = [g for g in avatar_gos if not _has_renderer(graph, g)]
    for fbx, renderers in embedded.items():
        jobs.append((fbx.stem, avatar_go, fbx, renderers + plain, True))

    target = SourceRead()
    _read_skinned(
        graph, state, avatar_go, target_gos, target_view, target, notes,
        label=f"{av.id} (target)", root_go=avatar_go, shared_names=set(), changes=changes,
    )
    unity_target_bones = {graph.name(g): g for g in avatar_gos}
    shared = {b.name for b in target_view.bones.values() if b.type_class == "LimbNode"}

    clothings: list[ClothingRead] = []
    proxies = bone_proxies(project, graph)
    for name, root, fbx, gos, is_embedded in jobs:
        label = f"{name} -> {fbx.name}" if not is_embedded else f"{name} (inside {graph.name(avatar_go)})"
        try:
            view = load_view(fbx)
        except Exception as exc:  # noqa: BLE001 - surface every reader failure
            skipped.append({"name": name, "path": str(fbx), "reason": f"could not read the FBX: {exc}"})
            continue
        source = graph.path(root, relative_to=avatar_go) if not is_embedded else graph.name(avatar_go)
        c = ClothingRead(clothing_path=str(fbx), source=source)
        if view.is_rigid() or is_self_rigged(view, target_view):
            c.rigid = view.is_rigid()
            c.self_rigged = not c.rigid
            ok = _read_prop(
                project, graph, state, avatar_go, gos, fbx, view, target_view, av, proxies,
                unity_target_bones, c, notes, label=label, self_rigged=c.self_rigged,
                changes=changes,
            )
            if not ok:
                skipped.append({"name": name, "path": str(fbx), "reason": "every mesh is hidden in the scene"})
                continue
        else:
            names = [strip_rig_suffix(b.name) for b in view.limb_bones() if b.name]
            try:
                identify_donor(names, registry, target_id=target_id)
            except DonorIdentificationError:
                shown = ", ".join(sorted(b.name for b in view.limb_bones())[:4])
                skipped.append({
                    "name": name, "path": str(fbx),
                    "reason": (f"its bones ({shown}) match no registry avatar well "
                               f"enough to merge it onto {target_id}"),
                })
                continue
            try:
                _read_skinned(
                    graph, state, avatar_go, gos, view, c, notes, label=label,
                    root_go=root, shared_names=shared, changes=changes,
                    send_shared=not is_embedded,
                )
            except SceneReadError as exc:
                skipped.append({"name": name, "path": str(fbx), "reason": str(exc)})
                continue
            meshes = {m.model_id for m in view.bones_of_type("Mesh")}
            if meshes and meshes <= set(c.drop_mesh_ids):
                skipped.append({"name": name, "path": str(fbx), "reason": "every mesh is hidden in the scene"})
                continue
            if not is_embedded:
                _placement_note(graph, gos, unity_target_bones, label, notes)
        clothings.append(c)
        if not is_embedded and not visible(graph, state, root, avatar_root=avatar_go):
            notes.append(f"{label}: its root is hidden in the scene")

    for iid, iname, guid in graph.missing:
        skipped.append({
            "name": iname or f"prefab instance {iid}", "path": f"guid {guid}",
            "reason": ("its prefab is missing from the Unity project (deleted?); Unity "
                       "shows it as a Missing Prefab"),
        })
    _unresolved_notes(graph, skipped, notes)
    # Merge order names the output: skinned outfits first, then props (rigid
    # or self-rigged); sorted() is stable, so each keeps scene order.
    clothings = sorted(clothings, key=lambda c: c.rigid or c.self_rigged)
    return SceneRead(
        scene=str(scene), target_id=target_id, target=target,
        clothings=clothings, skipped=skipped, notes=notes,
    )


def _split_embedded(project: UnityProject, graph: Graph, avatar_gos: list[int],
                    target_view: SectionView) -> tuple[list[int], dict[Path, list[int]]]:
    """The avatar's own objects, minus renderers of OTHER FBXs, and those
    renderers by FBX. The avatar's FBX is the one whose renderer names best
    match the target's meshes."""
    target_meshes = {m.name for m in target_view.bones_of_type("Mesh")}
    by_file: dict[Path, list[int]] = {}
    for g in avatar_gos:
        for kind in ("SkinnedMeshRenderer", "MeshFilter"):
            c = graph.component(g, kind)
            if c is None:
                continue
            _, guid = file_ref(c.data.get("m_Mesh"))
            f = project.path_for_guid(guid) if guid else None
            f = _model_file_for(f) if f is not None else None
            if f is not None:
                by_file.setdefault(f, []).append(g)
            break
    if len(by_file) <= 1:
        return list(avatar_gos), {}
    score = {f: sum(graph.name(g) in target_meshes for g in gs) for f, gs in by_file.items()}
    best = max(score.values())
    others = {f: gs for f, gs in by_file.items() if score[f] < best or best == 0}
    if best == 0:
        return list(avatar_gos), {}
    moved = {g for gs in others.values() for g in gs}
    return [g for g in avatar_gos if g not in moved], others


def _find_avatar(graph: Graph, registry: AvatarRegistry) -> tuple[int, str]:
    best: Optional[tuple[float, int, str]] = None
    for t in graph.roots():
        go = graph.go_of(t)
        if go is None:
            continue
        names = {normalize_bone_name(graph.name(g)) for g in _subtree_gos(graph, go)}
        for av in registry.avatars.values():
            want = {normalize_bone_name(n) for n in av.bone_names}
            if not want:
                continue
            share = len(want & names) / len(want)
            if share >= AVATAR_CONTAINMENT and (best is None or share > best[0]):
                best = (share, go, av.id)
    if best is None:
        known = ", ".join(sorted(registry.avatars)) or "none"
        raise SceneReadError(
            f"no object in the scene is built on a registry avatar ({known}): "
            f"none holds {AVATAR_CONTAINMENT:.0%} of an avatar's bones"
        )
    return best[1], best[2]


def _subtree_gos(graph: Graph, go: int) -> list[int]:
    t = graph.transform_of(go)
    if t is None:
        return [go]
    out = []
    for tid in graph.subtree(t):
        g = graph.go_of(tid)
        if g is not None:
            out.append(g)
    return out


def _garment_roots(graph: Graph, avatar_go: int) -> list[int]:
    """Top-level prefab/model instances under the avatar (not the avatar's
    own prefab, not instances nested inside another garment)."""
    instance_roots = set(graph.instance_sources) | set(graph.model_roots)
    out = []
    for go in _subtree_gos(graph, avatar_go):
        if go == avatar_go or go not in instance_roots:
            continue
        above = graph.ancestors(go)
        between = above[: above.index(avatar_go)] if avatar_go in above else above
        if not any(a in instance_roots for a in between):
            out.append(go)
    return out


def _fbx_groups(project: UnityProject, graph: Graph, root: int,
                unbound: Optional[dict[Path, list[int]]] = None) -> dict[Path, list[int]]:
    """The garment FBXs under one instance and the GameObjects of each: an
    imported-model instance root is its FBX; a renderer is its mesh's FBX."""
    groups: dict[Path, list[int]] = {}
    gos = _subtree_gos(graph, root)
    for go in gos:
        if go in graph.model_roots:
            groups.setdefault(graph.model_roots[go], []).append(go)
            continue
        for kind in ("SkinnedMeshRenderer", "MeshFilter"):
            c = graph.component(go, kind)
            if c is None:
                continue
            _, guid = file_ref(c.data.get("m_Mesh"))
            asset = project.path_for_guid(guid) if guid else None
            p = _model_file_for(asset) if asset is not None else None
            if p is not None:
                groups.setdefault(p, []).append(go)
            elif asset is not None and unbound is not None:
                unbound.setdefault(asset, []).append(go)
    # Every non-renderer GameObject of the instance (the bones) belongs to
    # each skinned group: bones are shared by the garment's meshes.
    for p, members in groups.items():
        if p not in graph.model_roots.values():
            members.extend(g for g in gos if g not in members and _is_plain(graph, g))
    return groups


_MODEL_SUFFIXES = (".fbx", ".FBX")


def _model_file_for(mesh_asset: Path) -> Optional[Path]:
    """The FBX a renderer's mesh comes from. A mesh stored in another asset
    (MochiFitter's `.omochi`, a baked `.asset`) stands for the FBX of the
    same name beside it — the file RigForge merges."""
    if mesh_asset.suffix in _MODEL_SUFFIXES:
        return mesh_asset
    for suf in _MODEL_SUFFIXES:
        sib = mesh_asset.with_suffix(suf)
        if sib.is_file():
            return sib
    return None


def _is_plain(graph: Graph, go: int) -> bool:
    return (graph.component(go, "SkinnedMeshRenderer") is None
            and graph.component(go, "MeshFilter") is None
            and go not in graph.model_roots)


def _unity_by_name(graph: Graph, gos: list[int]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for g in gos:
        out.setdefault(graph.name(g), []).append(g)
    return out


def _has_renderer(graph: Graph, go: int) -> bool:
    return any(graph.component(go, k) is not None
               for k in ("SkinnedMeshRenderer", "MeshFilter", "MeshRenderer"))


def _match_go(by_name: dict[str, list[int]], fbx_name: str, *, graph: Optional[Graph] = None,
              renderer: Optional[bool] = None) -> tuple[Optional[int], bool]:
    """(GameObject, ambiguous) for an FBX node: its own name, else the name
    with a fitting tool's suffix stripped (MochiFitter `Hips_MFBase`). Two
    objects of one name (Marette: a bone and a mesh both 'Bag') are told
    apart by `renderer`: True for a mesh, False for a bone."""
    for n in (fbx_name, strip_rig_suffix(fbx_name)):
        hits = by_name.get(n) or []
        if len(hits) > 1 and graph is not None and renderer is not None:
            hits = [h for h in hits if _has_renderer(graph, h) == renderer]
        if len(hits) == 1:
            return hits[0], False
        if len(hits) > 1:
            return None, True
    return None, False


def _bind_meshes(graph: Graph, gos: list[int], view: SectionView
                 ) -> tuple[dict[int, int], set[int]]:
    """FBX mesh id -> its renderer object, and the ambiguous mesh ids. By
    name first; then a lone unmatched renderer pairs with a lone unmatched
    mesh (GG.unity: the prop prefab's root IS the renderer, named after the
    prefab — GG_headset_Maya_3 — not after the FBX's mesh)."""
    by_name = _unity_by_name(graph, gos)
    bound: dict[int, int] = {}
    ambiguous: set[int] = set()
    for m in view.bones_of_type("Mesh"):
        go, amb = _match_go(by_name, m.name, graph=graph, renderer=True)
        if go is not None:
            bound[m.model_id] = go
        elif amb:
            ambiguous.add(m.model_id)
    left_meshes = [m.model_id for m in view.bones_of_type("Mesh")
                   if m.model_id not in bound and m.model_id not in ambiguous]
    left_gos = [g for g in gos if _has_renderer(graph, g) and g not in bound.values()]
    if len(left_meshes) == 1 and len(left_gos) == 1:
        bound[left_meshes[0]] = left_gos[0]
    return bound, ambiguous


def _to_local(m) -> LocalTransform:
    """affine_to_fbx_local, tolerating the hair of shear that FBX scales a
    hair off 1.0 leave (Marette: 0.999998 / 0.999995 above rotated skirt
    bones): orthonormalise when it is tiny, refuse real shear."""
    try:
        return affine_to_fbx_local(m)
    except ValueError as exc:
        if "shear" not in str(exc):
            raise
    cols = [[m[i][j] for i in range(3)] for j in range(3)]
    scale = [math.sqrt(sum(v * v for v in c)) for c in cols]
    unit = [[v / s for v in c] for c, s in zip(cols, scale)]
    worst = max(abs(sum(unit[a][k] * unit[b][k] for k in range(3)))
                for a in range(3) for b in range(3) if a < b)
    if worst > 1e-3:
        raise ValueError("matrix has shear; it is not expressible as T . R . S")
    e0 = unit[0]
    e1 = [u - sum(unit[1][k] * e0[k] for k in range(3)) * e for u, e in zip(unit[1], e0)]
    n1 = math.sqrt(sum(v * v for v in e1))
    e1 = [v / n1 for v in e1]
    e2 = [e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2], e0[0] * e1[1] - e0[1] * e1[0]]
    if sum(a * b for a, b in zip(e2, unit[2])) < 0:
        e2 = [-v for v in e2]
    fixed = [[e0[i] * scale[0], e1[i] * scale[1], e2[i] * scale[2], m[i][3]] for i in range(3)]
    fixed.append([0.0, 0.0, 0.0, 1.0])
    return affine_to_fbx_local(_tuple(fixed))


def _rel_world(graph: Graph, go: int, root_go: int) -> list[list[float]]:
    t, rt = graph.transform_of(go), graph.transform_of(root_go)
    w = graph.world_matrix(t)
    return mat_mul(_t(affine_inverse(_tuple(graph.world_matrix(rt)))), w)


def _tuple(m) -> tuple:
    return tuple(tuple(r) for r in m)


def _read_skinned(
    graph: Graph, state: dict[int, bool], avatar_go: int, gos: list[int],
    view: SectionView, out: SourceRead, notes: list[str], *, label: str,
    root_go: int, shared_names: set[str], changes: dict[int, list[ShapeChange]],
    send_shared: bool = True,
) -> None:
    hidden_names: list[str] = []
    weight_count = 0

    bound, ambiguous = _bind_meshes(graph, gos, view)
    stray = sorted(
        graph.name(g) for g in gos
        if graph.component(g, "SkinnedMeshRenderer") is not None
        and g not in bound.values()
        and visible(graph, state, g, avatar_root=avatar_go)
    )
    if stray:
        notes.append(
            f"{label}: {', '.join(stray)} show in the scene but are not meshes of this FBX; "
            f"not carried"
        )

    for m in view.bones_of_type("Mesh"):
        go, amb = bound.get(m.model_id), m.model_id in ambiguous
        if go is None:
            if amb:
                notes.append(f"{label}: mesh {m.name!r} is ambiguous in the scene; left as the file has it")
            continue
        if not visible(graph, state, go, avatar_root=avatar_go):
            out.drop_mesh_ids.append(m.model_id)
            hidden_names.append(m.name)
            continue
        smr = graph.component(go, "SkinnedMeshRenderer")
        if smr is None:
            continue
        if str(smr.data.get("m_Enabled", "1")) == "0":
            out.drop_mesh_ids.append(m.model_id)
            hidden_names.append(m.name)
            continue
        raw = smr.data.get("m_BlendShapeWeights") or []
        chans = mesh_channels(view, m.model_id)
        if len(raw) > len(chans):
            notes.append(
                f"{label}: {m.name!r} has {len(raw)} blendshape slots in Unity but "
                f"{len(chans)} in the FBX; its blendshapes were not read"
            )
            continue
        weights = {i: as_float(w) for i, w in enumerate(raw)}
        _apply_shape_changes(weights, view, chans, changes.get(go, []), m.name, label, notes)
        for i, val in weights.items():
            if abs(val - deform_percent(view, chans[i])) > WEIGHT_TOL:
                out.channel_overrides[chans[i]] = round(val, 4)
                weight_count += 1
    out.drop_mesh_ids.sort()

    moves, cant = _bone_moves(graph, gos, view, root_go, shared_names, label,
                              send_shared=send_shared)
    out.bone_transforms.extend(moves)
    notes.extend(cant)
    parts = []
    if hidden_names:
        parts.append(f"hide {', '.join(hidden_names)}")
    if weight_count:
        parts.append(f"{weight_count} blendshape value(s)")
    if moves:
        parts.append(f"{len(moves)} bone move(s)")
    notes.append(f"{label}: " + ("; ".join(parts) if parts else "nothing changed from the FBX"))


def _apply_shape_changes(weights: dict[int, float], view: SectionView, chans: list[int],
                         changes: list[ShapeChange], mesh: str, label: str,
                         notes: list[str]) -> None:
    """MA Shape Changer entries on this renderer: Set writes the weight of
    every channel by that name; Delete removes vertices — not carried."""
    for ch in changes:
        idx = [i for i, c in enumerate(chans) if view.blend_shape_channels[c].name == ch.shape]
        if not idx:
            notes.append(f"{label}: MA Shape Changer names {ch.shape!r}, which {mesh!r} lacks")
            continue
        if ch.delete:
            notes.append(
                f"{label}: an MA Shape Changer deletes the vertices under {ch.shape!r} on "
                f"{mesh!r}; RigForge can't delete vertices, so that is not carried"
            )
            continue
        for i in idx:
            weights[i] = ch.value


def _bone_moves(
    graph: Graph, gos: list[int], view: SectionView, root_go: int,
    shared_names: set[str], label: str, *, send_shared: bool = True,
) -> tuple[list[dict], list[str]]:
    """Moves for the FBX's bones whose placement in the scene differs from
    the file (see module doc). Parents first, so a child of a moved bone
    only moves if its own local changed."""
    by_name = _unity_by_name(graph, gos)
    carried: dict[int, list[list[float]]] = {}       # FBX id -> world it will have
    moves: list[dict] = []
    cant: list[str] = []
    checked = 0
    own_count = 0

    def world_of(mid: Optional[int]) -> list[list[float]]:
        if mid is None or mid not in view.bones:
            return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        if mid not in carried:
            b = view.bones[mid]
            carried[mid] = mat_mul(world_of(b.parent_id if b.parent_id else None),
                                   _lt_matrix(_local_fbx(view, mid)))
        return carried[mid]

    order = _parents_first(view)
    for mid in order:
        b = view.bones[mid]
        if b.type_class != "LimbNode":
            continue
        go, _ = _match_go(by_name, b.name, graph=graph, renderer=False)
        if go is None:
            continue
        checked += 1
        parent_world = world_of(b.parent_id if b.parent_id else None)
        want_world = _mirror(_rel_world(graph, go, root_go))
        try:
            want_local = _to_local(_tuple(mat_mul(_t(affine_inverse(_tuple(parent_world))), want_world)))
        except ValueError as exc:
            cant.append(f"{label}: {b.name!r}: {exc}")
            continue
        file_local = _local_fbx(view, mid)
        if _close(want_local, file_local):
            continue
        # Did ITS OWN local change, or does it only ride on a moved parent?
        own_changed = True
        pb = view.bones.get(b.parent_id) if b.parent_id else None
        pgo = (_match_go(by_name, pb.name, graph=graph, renderer=False)[0]
               if pb is not None and pb.type_class == "LimbNode" else None)
        if pgo is not None:
            own = mat_mul(_t(affine_inverse(_tuple(_mirror(_rel_world(graph, pgo, root_go))))),
                          want_world)
            try:
                own_changed = not _close(_to_local(_tuple(own)), file_local)
            except ValueError:
                pass
        if own_changed:
            own_count += 1
        if strip_rig_suffix(b.name) in shared_names or b.name in shared_names:
            # Merged into the avatar's bone: news only if its own local
            # changed (Phase C carries the ride along itself). An outfit
            # embedded in the avatar's prefab shares the avatar's very bone
            # objects: their moves are the target's, read there.
            if not own_changed or not send_shared:
                continue
            # Sent like any move; Phase C bakes it into the outfit's vertices
            # on that bone (the bone itself stays the avatar's). The output
            # bone does not move, so `carried` keeps its file world and its
            # ride-along children are placed against that.
            delta = tuple(want_world[i][3] - world_of(mid)[i][3] for i in range(3))
            meshes = bone_to_mesh_names(view, mid)
            on = f" ({', '.join(meshes)})" if meshes else ""
            cant.append(
                f"{label}: {b.name!r} is moved {_cm((-delta[0], delta[1], delta[2]))} in the "
                f"scene; it merges into the avatar's own {strip_rig_suffix(b.name)!r}, so the "
                f"move is baked into the vertices on it{on}"
            )
            moves.append({"bone_id": mid, **_unity_dict(want_local)})
            continue
        carried[mid] = want_world
        moves.append({"bone_id": mid, **_unity_dict(want_local)})

    # Only bones whose own local changed count: a moved parent's ride-along
    # children follow it and are not evidence of a frame mismatch.
    if checked and own_count > MAX_MOVED_SHARE * checked:
        raise SceneReadError(
            f"{label}: {own_count} of {checked} bones differ from the FBX — the scene's "
            f"copy does not line up with this file (wrong FBX or a different frame)"
        )
    return moves, cant


def _parents_first(view: SectionView) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()

    def visit(mid: int, stack: set[int]) -> None:
        if mid in seen or mid in stack:
            return
        b = view.bones[mid]
        if b.parent_id in view.bones:
            visit(b.parent_id, stack | {mid})
        seen.add(mid)
        out.append(mid)

    for mid in view.bones:
        visit(mid, set())
    return out


def _placement_note(graph: Graph, gos: list[int], target_bones: dict[str, int],
                    label: str, notes: list[str]) -> None:
    """The merge snaps a garment onto the avatar's bones; say so when the
    scene shows it somewhere else."""
    for name in ("Hips", "Spine", "Chest"):
        mine = [g for g in gos if strip_rig_suffix(graph.name(g)) == name]
        theirs = target_bones.get(name)
        if len(mine) != 1 or theirs is None:
            continue
        a = graph.world_matrix(graph.transform_of(mine[0]))
        b = graph.world_matrix(graph.transform_of(theirs))
        d = math.dist([a[i][3] for i in range(3)], [b[i][3] for i in range(3)])
        if d > PLACEMENT_NOTE_M:
            notes.append(
                f"{label}: its {name} sits {d * 100:.1f} cm from the avatar's in the scene; "
                f"the merge puts it on her bones"
            )
        return


@dataclass
class _Inner:
    """Overrides on ONE imported-model instance's inner objects, by FBX node
    name (read through the model's import artifact, see unity/library.py)."""
    transforms: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    weights: dict[str, dict[int, float]] = field(default_factory=dict)
    active: dict[str, bool] = field(default_factory=dict)
    enabled: dict[str, bool] = field(default_factory=dict)
    read: int = 0
    unread: int = 0


_TRANSFORM_CLASSES = (4, 224)
_RENDERER_CLASSES = (137, 23)


def _inner_state(project: UnityProject, graph: Graph, model_root: Optional[int], fbx: Path,
                 collapsed_mesh: Optional[str]) -> _Inner:
    st = _Inner()
    if model_root is None:
        return st
    mine = [io for io in graph.inner_overrides
            if any(go == model_root for go, _p, _i in io.candidates)]
    if not mine:
        return st
    names = model_object_names(project.root, fbx)
    carried = ("m_LocalPosition.", "m_LocalRotation.", "m_LocalScale.",
               "m_BlendShapeWeights.", "m_IsActive", "m_Enabled")
    for io in mine:
        hit = next((names[i] for go, _p, i in io.candidates if go == model_root and i in names), None)
        if hit is None:
            # Bounds / material overrides on a vanished object change nothing
            # RigForge carries; only count the ones that would.
            if io.prop.startswith(carried):
                st.unread += 1
            continue
        cls, name = hit
        if name == fbx.stem and collapsed_mesh is not None:
            name = collapsed_mesh            # a collapsed root IS the model's one mesh
        prop, val = io.prop, io.value
        if cls in _TRANSFORM_CLASSES and prop.split(".")[0] in (
                "m_LocalPosition", "m_LocalRotation", "m_LocalScale"):
            fld, comp = prop.split(".", 1)
            st.transforms.setdefault(name, {}).setdefault(fld, {})[comp] = str(val)
        elif cls in _RENDERER_CLASSES and prop.startswith("m_BlendShapeWeights.Array.data["):
            st.weights.setdefault(name, {})[int(prop.split("[")[1].rstrip("]"))] = as_float(val)
        elif cls == 1 and prop == "m_IsActive":
            st.active[name] = str(val) != "0"
        elif cls in _RENDERER_CLASSES and prop == "m_Enabled":
            st.enabled[name] = str(val) != "0"
        else:
            continue                         # bounds, materials: nothing RigForge carries
        st.read += 1
    return st


def _quat_from_matrix(m) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation matrix (Shepperd)."""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return (0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ((m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s)


def _inner_local_unity(view: SectionView, mid: int, inner: _Inner) -> list[list[float]]:
    """An inner node's local as Unity holds it: the FBX node's (as Unity
    imports it) with the scene's / prefab's overrides on top."""
    from rigforge.ascii_fbx.transform import unity_euler_to_matrix
    base = fbx_to_unity(_local_fbx(view, mid))
    pos, scl = list(base.position), list(base.scale)
    q = list(_quat_from_matrix(unity_euler_to_matrix(*base.rotation)))
    ov = inner.transforms.get(view.bones[mid].name) or {}
    p, r, sc = ov.get("m_LocalPosition", {}), ov.get("m_LocalRotation", {}), ov.get("m_LocalScale", {})
    pos = [as_float(p.get(k), pos[i]) for i, k in enumerate("xyz")]
    q = [as_float(r.get(k), q[i]) for i, k in enumerate("xyzw")]
    scl = [as_float(sc.get(k), scl[i]) for i, k in enumerate("xyz")]
    return trs_matrix(pos, tuple(q), scl)


def _inner_world_fbx(view: SectionView, mid: int, inner: _Inner) -> list[list[float]]:
    """FBX-frame world of a node relative to its model's root, overrides in."""
    m = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    cur: Optional[int] = mid
    while cur is not None and cur in view.bones:
        m = mat_mul(_mirror(_inner_local_unity(view, cur, inner)), m)
        parent = view.bones[cur].parent_id
        cur = parent if parent else None
    return m


def _imports_deform_percent(fbx: Path) -> bool:
    """Unity starts an imported renderer's weights at the FBX's DeformPercent
    unless the importer's `importBlendShapeDeformPercent` is off."""
    try:
        meta = Path(str(fbx) + ".meta").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return True
    return "importBlendShapeDeformPercent: 0" not in meta


def _read_prop(
    project: UnityProject, graph: Graph, state: dict[int, bool], avatar_go: int,
    gos: list[int], fbx: Path, view: SectionView, target_view: SectionView, av,
    proxies: list[BoneProxy], target_bones: dict[str, int], out: ClothingRead,
    notes: list[str], *, label: str, self_rigged: bool,
    changes: dict[int, list[ShapeChange]],
) -> bool:
    """A prop — rigid (no bones) or self-rigged (its own armature): where the
    build places it (a Bone Proxy snaps it to its bone), the bone it hangs
    from, its hidden meshes and blendshapes, and (self-rigged) moves on its
    own bones. Objects inside an imported model are read through the import
    artifact. Returns False when every mesh is hidden."""
    from rigforge.pipeline.phase_c import prop_roots, rigid_move_frame

    roots = prop_roots(view, self_rigged=self_rigged)
    tops = [b for b in view.bones.values() if b.parent_id in (None, 0)]
    model_root = next((g for g in gos if g in graph.model_roots), None)
    # Unity reuses a single top-level node as the model's root object.
    collapsed = model_root is not None and len(tops) == 1
    by_name = _unity_by_name(graph, gos)
    bound, _amb = _bind_meshes(graph, gos, view)
    inner = _inner_state(project, graph, model_root, fbx,
                         tops[0].name if collapsed and tops[0].type_class == "Mesh" else None)

    # Where the build puts it: a Bone Proxy above it snaps it to its bone.
    anchor_go = model_root if model_root is not None else (gos[0] if gos else None)
    proxy, proxy_go = _nearest_proxy(graph, anchor_go, proxies, avatar_go)
    attach_name: Optional[str] = None
    if proxy is not None:
        role = _HUMAN_TO_ROLE.get(proxy.human_bone or "")
        attach_name = av.canonical_to_name.get(role) if role else None
        if attach_name and proxy.sub_path:
            attach_name = proxy.sub_path.rstrip("/").split("/")[-1]
        if attach_name is None:
            notes.append(f"{label}: Bone Proxy on {proxy.human_bone!r} has no RigForge bone; left at the Armature")
    else:
        attach_name = _nearest_target_bone(graph, anchor_go, target_bones, avatar_go)
    attach_id = None
    if attach_name is not None:
        hits = [b.model_id for b in target_view.bones.values()
                if b.name == attach_name and b.type_class == "LimbNode"]
        attach_id = hits[0] if len(hits) == 1 else None
    out.attach_to_bone_id = attach_id

    def go_and_rel(mid: int) -> tuple[Optional[int], list[list[float]]]:
        """The scene object a node is read through, and the node's FBX-frame
        pose relative to it (identity when the object IS the node)."""
        ident = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        if collapsed:
            return model_root, ident
        if view.bones[mid].type_class == "Mesh":
            go = bound.get(mid)
        else:
            go, _ = _match_go(by_name, view.bones[mid].name, graph=graph, renderer=False)
        if go is not None:
            return go, ident
        if model_root is not None:          # inside a wrapper root
            return model_root, _inner_world_fbx(view, mid, inner)
        return None, ident

    def shown(mid: int) -> bool:
        go, _ = go_and_rel(mid)
        if go is None:
            return True
        name = view.bones[mid].name
        return (visible(graph, state, go, avatar_root=avatar_go)
                and inner.active.get(name, True) and inner.enabled.get(name, True))

    _, frame_world = rigid_move_frame(target_view)
    frame_inv = _t(affine_inverse(frame_world))
    root_ids = {r.model_id for r in roots}
    for r in roots:
        if r.type_class == "Mesh" and not shown(r.model_id):
            continue
        go, rel = go_and_rel(r.model_id)
        if go is None:
            continue
        built = _built_world(graph, go, avatar_go, proxy, proxy_go, target_view, attach_id)
        world = mat_mul(built, rel)
        local = _to_local(_tuple(mat_mul(frame_inv, world)))
        out.bone_transforms.append({"bone_id": r.model_id, **_unity_dict(local)})
        editor = mat_mul(_mirror(_rel_world(graph, go, avatar_go)), rel)
        gap = math.dist([editor[i][3] for i in range(3)], [world[i][3] for i in range(3)])
        turn = _angle_deg(editor, world)
        if gap > 0.001 or turn > 0.5:
            notes.append(
                f"{label}: {r.name!r} sits {gap * 100:.1f} cm / {turn:.1f} deg from where the "
                f"build puts it (the Bone Proxy snaps it to {attach_name}); read at the built place"
            )

    # Its own bones (self-rigged): an ordinary move where the scene changed one.
    if self_rigged:
        for b in view.bones.values():
            if b.model_id in root_ids or b.type_class == "Mesh":
                continue
            if b.type_class == "Null" and b.parent_id in (None, 0):
                continue                      # stripped in the merge; its pose is in the roots'
            go, _ = _match_go(by_name, b.name, graph=graph, renderer=False)
            if b.name not in inner.transforms and go is None:
                continue
            if go is not None:
                want = _mirror(graph.local_matrix(graph.transform_of(go)))
            else:
                want = _mirror(_inner_local_unity(view, b.model_id, inner))
            local = _to_local(_tuple(want))
            if not _close(local, _local_fbx(view, b.model_id)):
                out.bone_transforms.append({"bone_id": b.model_id, **_unity_dict(local)})

    # Hidden meshes and blendshapes.
    any_visible = False
    base_from_file = _imports_deform_percent(fbx)
    for m in view.bones_of_type("Mesh"):
        if not shown(m.model_id):
            out.drop_mesh_ids.append(m.model_id)
            continue
        any_visible = True
        chans = mesh_channels(view, m.model_id)
        go, _ = go_and_rel(m.model_id)
        smr = graph.component(go, "SkinnedMeshRenderer") if go is not None and go != model_root else None
        if smr is not None:
            weights = {i: as_float(w) for i, w in enumerate(smr.data.get("m_BlendShapeWeights") or [])}
        else:
            weights = {i: (deform_percent(view, c) if base_from_file else 0.0) for i, c in enumerate(chans)}
            weights.update(inner.weights.get(m.name, {}))
        if go is not None and (go != model_root or collapsed):
            _apply_shape_changes(weights, view, chans, changes.get(go, []), m.name, label, notes)
        for i, w in weights.items():
            if i < len(chans) and abs(w - deform_percent(view, chans[i])) > WEIGHT_TOL:
                out.channel_overrides[chans[i]] = round(w, 4)
    out.drop_mesh_ids.sort()

    kind = "self-rigged prop (its own armature)" if self_rigged else "rigid"
    where = f"hangs from {attach_name}" if attach_id is not None else "sits on the Armature"
    how = " (Bone Proxy)" if proxy is not None else ""
    msg = f"{label}: {kind}, {where}{how}; {len(out.bone_transforms)} placement/move(s)"
    if out.drop_mesh_ids:
        msg += f"; hide {len(out.drop_mesh_ids)} mesh(es)"
    if out.channel_overrides:
        msg += f"; {len(out.channel_overrides)} blendshape value(s)"
    if inner.read:
        msg += f"; {inner.read} override(s) read inside the FBX's import"
    notes.append(msg)
    if inner.unread:
        notes.append(
            f"{label}: {inner.unread} override(s) point at objects its Unity import does not have "
            f"(stale, or no import artifact in Library/); not read"
        )
    return any_visible


def _angle_deg(a: list[list[float]], b: list[list[float]]) -> float:
    """Angle of the rotation taking a's orientation to b's (scale removed)."""
    def unit(m):
        cols = []
        for j in range(3):
            n = math.sqrt(sum(m[i][j] ** 2 for i in range(3))) or 1.0
            cols.append([m[i][j] / n for i in range(3)])
        return cols   # column vectors
    ca, cb = unit(a), unit(b)
    # trace(A^T B) = sum of column dot products
    tr = sum(sum(ca[j][i] * cb[j][i] for i in range(3)) for j in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


def _nearest_proxy(graph: Graph, go: Optional[int], proxies: list[BoneProxy],
                   avatar_go: int) -> tuple[Optional[BoneProxy], Optional[int]]:
    if go is None:
        return None, None
    by_go = {p.go: p for p in proxies}
    for g in [go, *graph.ancestors(go)]:
        if g == avatar_go:
            break
        if g in by_go:
            return by_go[g], g
    return None, None


def _nearest_target_bone(graph: Graph, go: Optional[int], target_bones: dict[str, int],
                         avatar_go: int) -> Optional[str]:
    if go is None:
        return None
    own = set(target_bones.values())
    for g in graph.ancestors(go):
        if g == avatar_go:
            return None
        if g in own and graph.name(g) not in ("Armature",):
            return graph.name(g)
    return None


def _built_world(graph: Graph, go: int, avatar_go: int, proxy: Optional[BoneProxy],
                 proxy_go: Optional[int], target_view: SectionView,
                 attach_id: Optional[int]) -> list[list[float]]:
    """FBX-frame world of `go` once the build has placed it: a Bone Proxy
    re-places its object on the target bone (by mode); everything else stays
    where the scene shows it."""
    editor = _mirror(_rel_world(graph, go, avatar_go))
    if proxy is None or proxy_go is None or attach_id is None or proxy.mode == "AsChildKeepWorldPose":
        return editor
    bone_world = _t(model_world_matrix(target_view, attach_id))
    proxy_editor = _mirror(_rel_world(graph, proxy_go, avatar_go))
    pt = graph.transform_of(proxy_go)
    sx, sy, sz = graph.local_scale(pt)
    if proxy.mode == "AsChildAtRoot":
        placed = mat_mul(bone_world, [[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1.0]])
    elif proxy.mode == "AsChildKeepRotation":
        placed = [row[:] for row in proxy_editor]
        for i in range(3):
            placed[i][3] = bone_world[i][3]
    elif proxy.mode == "AsChildKeepPosition":
        placed = mat_mul(bone_world, [[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1.0]])
        for i in range(3):
            placed[i][3] = proxy_editor[i][3]
    else:
        return editor
    # go's pose relative to the proxy object is unchanged by the snap.
    rel = mat_mul(_t(affine_inverse(_tuple(proxy_editor))), editor)
    return mat_mul(placed, rel)


def _unresolved_notes(graph: Graph, skipped: list[dict], notes: list[str]) -> None:
    skipped_files = {Path(s["path"]).name for s in skipped}
    by_file: dict[str, int] = {}
    for u in graph.unresolved:
        f = u.split(":", 1)[0]
        stem = Path(f).stem
        if any(Path(s).stem == stem or stem.startswith(Path(s).stem) for s in skipped_files):
            continue
        by_file[f] = by_file.get(f, 0) + 1
    for f, n in by_file.items():
        notes.append(f"{f}: {n} override(s) point at objects the prefab no longer has (stale); not read")


__all__ = [
    "ClothingRead",
    "SceneRead",
    "SceneReadError",
    "SourceRead",
    "deform_percent",
    "mesh_channels",
    "read_scene",
]
