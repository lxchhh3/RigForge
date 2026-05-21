"""Section extractor: produces per-bone JSON records from a parsed ASCII FBX.

This is the input shape for the LLM (Rail L). The lexer gives us a tree with
byte offsets; this module pulls the semantic data we care about (bone names,
hierarchy, translations, skin attachments) out of that tree and emits a
deterministic JSON view.

Per PLAN.md Phase B step 1, augmented with world-space anchors (v1.1):
    {model_id, name, parent_name, translation_xyz, child_names,
     has_skin_cluster, cluster_weight_count,
     world_xyz, height_pct, lateral, front_back}

`world_xyz` comes from each bone's `Cluster.TransformLink` (the bind-pose
world matrix) when available, else by walking the parent chain with proper
4x4 T·R·S composition (FBX rotation order eXYZ). The bucketed fields
(`height_pct`, `lateral`, `front_back`) are derived from scene-wide bone
bounds so the LLM can disambiguate name-ambiguous parts spatially — e.g. an
`OuterB` panel whose name doesn't say front-vs-back.

Raw text offsets are retained on the side (`BoneRecord.node_ref`) so edits.py
can do byte-surgery on the lexer's offsets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .lexer import FBXDocument, FBXNode


# --- args parsing ------------------------------------------------------------

ArgValue = tuple[str, Any]  # ("str", str) | ("num", int|float)


def parse_args(text: bytes) -> list[ArgValue]:
    """Split a comma-separated args text into typed values.

    Returns a list of (kind, value) tuples where kind is "str" for quoted
    strings and "num" for ints/floats. Unrecognized tokens (e.g. "*4782")
    are returned as ("str", token_text).

    The lexer extends a leaf node's text to the start of the next sibling,
    which can include `;`-style comments. We skip comments inline here so
    they don't bleed into the previous token's value.
    """
    out: list[ArgValue] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        # whitespace
        if c in b" \t\r\n":
            i += 1
            continue
        # comment: ';' to end of line
        if c == 0x3B:
            while i < n and text[i] != 0x0A:
                i += 1
            continue
        # string literal
        if c == 0x22:
            i += 1
            j = i
            while j < n and text[j] != 0x22:
                j += 1
            s = text[i:j].decode("utf-8", errors="replace")
            out.append(("str", s))
            i = j + 1 if j < n else j
            continue
        # comma separator
        if c == 0x2C:
            i += 1
            continue
        # bare token: stop at structural break (comma, semicolon, whitespace)
        j = i
        while j < n and text[j] not in b",; \t\r\n":
            j += 1
        tok = text[i:j].decode("utf-8", errors="replace")
        i = j
        if not tok:
            continue
        try:
            if "." in tok or "e" in tok or "E" in tok:
                out.append(("num", float(tok)))
            else:
                out.append(("num", int(tok)))
        except ValueError:
            out.append(("str", tok))
    return out


def _num(v: Optional[ArgValue]) -> Optional[float]:
    """Coerce an ArgValue to a number; return None if missing/not-numeric."""
    if v is None:
        return None
    if v[0] == "num":
        return v[1]
    return None


def _str(v: Optional[ArgValue]) -> Optional[str]:
    if v is None:
        return None
    if v[0] == "str":
        return v[1]
    return None


# --- record types ------------------------------------------------------------


@dataclass
class BoneRecord:
    """One Model node in the Objects section.

    `name` has the "Model::" prefix stripped. `full_name` keeps it intact for
    edits that need to rewrite the quoted FBX-side name.
    """
    model_id: int
    name: str
    full_name: str
    type_class: str             # "LimbNode" | "Mesh" | "Null" | ...
    parent_id: Optional[int]
    parent_name: Optional[str]
    children_ids: list[int]
    child_names: list[str]
    translation_xyz: tuple[float, float, float]
    has_skin_cluster: bool
    cluster_weight_count: int
    cluster_ids: list[int]
    node_ref: FBXNode = field(repr=False)
    # World-space anchors (v1.1). Populated post-construction by
    # `_populate_world_positions`. world_xyz is in scene units (typically
    # meters). height_pct is 0..1 normalized to the scene's Y bounds.
    # lateral is "L"/"R"/"C", front_back is "F"/"B"/"C".
    world_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    height_pct: float = 0.0
    lateral: str = "C"
    front_back: str = "C"

    def to_json_record(self) -> dict:
        """LLM-input JSON view (Phase B step 1, v1.1 schema)."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "parent_name": self.parent_name,
            "translation_xyz": list(self.translation_xyz),
            "child_names": list(self.child_names),
            "has_skin_cluster": self.has_skin_cluster,
            "cluster_weight_count": self.cluster_weight_count,
            "world_xyz": list(self.world_xyz),
            "height_pct": round(self.height_pct, 3),
            "lateral": self.lateral,
            "front_back": self.front_back,
        }


@dataclass
class ClusterRecord:
    cluster_id: int
    name: str                   # short name (e.g., "Hips" from "SubDeformer::Hips")
    weight_count: int           # *N from Indexes (0 if absent)
    node_ref: FBXNode = field(repr=False)


@dataclass
class MaterialRecord:
    """One Material node in the Objects section.

    `name` has the "Material::" prefix stripped. `full_name` keeps it intact for
    edits that need to rewrite or duplicate the quoted FBX-side name. Merge
    dedup uses `name` as the natural key (FBX material names are unique within
    a document by SDK convention).
    """
    material_id: int
    name: str
    full_name: str
    node_ref: FBXNode = field(repr=False)


@dataclass
class BlendShapeRecord:
    """One `Deformer: id, "Deformer::Name", "BlendShape"` node.

    A BlendShape Deformer is the parent container that owns one or more
    BlendShapeChannel SubDeformers, attached to a Geometry via Connections.
    """
    deformer_id: int
    name: str
    full_name: str
    node_ref: FBXNode = field(repr=False)


@dataclass
class BlendShapeChannelRecord:
    """One `Deformer: id, "SubDeformer::Name", "BlendShapeChannel"` node.

    The channel is the dedup key for blendshape merging — its `name` is the
    user-facing morph name (e.g. "Smile", "Wink"). `blend_shape_id` is the
    BlendShape Deformer that owns it (resolved from Connections, None if the
    document lacks the OO link).
    """
    channel_id: int
    name: str
    full_name: str
    blend_shape_id: Optional[int]
    node_ref: FBXNode = field(repr=False)


@dataclass
class GeometryRecord:
    """One `Geometry: id, "Geometry::Name", "Mesh"` node.

    Holds the actual mesh vertex data. Linked to a Mesh-type Model via OO
    connection (Geometry → owning Model). Linked to a Skin Deformer via OO
    connection (Skin → Geometry) when skinned.
    """
    geometry_id: int
    name: str
    full_name: str
    node_ref: FBXNode = field(repr=False)


@dataclass
class SkinRecord:
    """One `Deformer: id, "Deformer::Name", "Skin"` node.

    A Skin deforms one Geometry and owns multiple Cluster SubDeformers
    (one per influencing bone). `geometry_id` is resolved from Connections.
    """
    skin_id: int
    name: str
    full_name: str
    geometry_id: Optional[int]
    node_ref: FBXNode = field(repr=False)


@dataclass
class SectionView:
    bones: dict[int, BoneRecord]
    clusters: dict[int, ClusterRecord]
    document: FBXDocument
    materials: dict[int, MaterialRecord] = field(default_factory=dict)
    blend_shapes: dict[int, BlendShapeRecord] = field(default_factory=dict)
    blend_shape_channels: dict[int, BlendShapeChannelRecord] = field(default_factory=dict)
    geometries: dict[int, GeometryRecord] = field(default_factory=dict)
    skins: dict[int, SkinRecord] = field(default_factory=dict)
    # Derived affinity indexes — populated by extract().
    # cluster_skin: cluster_id -> skin_id
    cluster_skin: dict[int, int] = field(default_factory=dict)
    # geometry_owner_model: geometry_id -> Mesh-type Model id (owner)
    geometry_owner_model: dict[int, int] = field(default_factory=dict)

    def bones_of_type(self, type_class: str) -> list[BoneRecord]:
        return [b for b in self.bones.values() if b.type_class == type_class]

    def limb_bones(self) -> list[BoneRecord]:
        return self.bones_of_type("LimbNode")

    def json_records(self, *, type_classes: tuple[str, ...] = ("LimbNode",)) -> list[dict]:
        """Stable-ordered JSON records for LLM consumption."""
        recs = [b.to_json_record() for b in self.bones.values()
                if b.type_class in type_classes]
        recs.sort(key=lambda r: r["model_id"])
        return recs


class SectionError(RuntimeError):
    """Raised when the FBX structure is missing required sections."""


# --- extraction --------------------------------------------------------------


_MODEL_PREFIX = "Model::"
_SUBDEFORMER_PREFIX = "SubDeformer::"
_DEFORMER_PREFIX = "Deformer::"
_MATERIAL_PREFIX = "Material::"
_GEOMETRY_PREFIX = "Geometry::"


def _strip_prefix(s: str, prefix: str) -> str:
    return s[len(prefix):] if s.startswith(prefix) else s


def _parse_object_args(args: list[ArgValue]) -> tuple[Optional[int], str, str]:
    """For Model/Geometry/Deformer style: `id, "ClassPrefix::Name", "TypeClass"`."""
    if len(args) < 3:
        return (None, "", "")
    mid = _num(args[0])
    name = _str(args[1]) or ""
    type_class = _str(args[2]) or ""
    return (int(mid) if mid is not None else None, name, type_class)


def _extract_translation(model: FBXNode, source: bytes) -> tuple[float, float, float]:
    """Pull Lcl Translation from a Model's Properties70 block. Default (0,0,0)."""
    props = model.child("Properties70")
    if props is None:
        return (0.0, 0.0, 0.0)
    for p in props.child_all("P"):
        args = parse_args(p.args_bytes(source))
        # P args: "PropName", "Type1", "Type2", "Flag", X, Y, Z
        if not args or _str(args[0]) != "Lcl Translation":
            continue
        # Translation values are the last 3 numeric args.
        nums = [_num(a) for a in args if a[0] == "num"]
        if len(nums) >= 3:
            return (float(nums[-3]), float(nums[-2]), float(nums[-1]))
    return (0.0, 0.0, 0.0)


def _extract_props_vec3(model: FBXNode, source: bytes, prop_name: str,
                         default: tuple[float, float, float]) -> tuple[float, float, float]:
    """Read any Properties70 P entry whose name == prop_name as a 3-vector.
    Used for Lcl Translation / Lcl Rotation / Lcl Scaling extraction.
    """
    props = model.child("Properties70")
    if props is None:
        return default
    for p in props.child_all("P"):
        args = parse_args(p.args_bytes(source))
        if not args or _str(args[0]) != prop_name:
            continue
        nums = [_num(a) for a in args if a[0] == "num"]
        if len(nums) >= 3:
            return (float(nums[-3]), float(nums[-2]), float(nums[-1]))
    return default


_IDENTITY_4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _mat_mul(a: tuple, b: tuple) -> tuple:
    """4x4 matrix multiply: a @ b. Pure-Python, no numpy dependency."""
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


def _local_matrix(translation: tuple[float, float, float],
                   rotation_deg: tuple[float, float, float],
                   scale: tuple[float, float, float]) -> tuple:
    """Build a 4x4 local transform M = T · R · S.
    FBX default RotationOrder is eXYZ → R = Rz · Ry · Rx (column-vector
    convention; rotations applied X, then Y, then Z).
    """
    tx, ty, tz = translation
    rx, ry, rz = (math.radians(a) for a in rotation_deg)
    sx, sy, sz = scale
    cx, sx_ = math.cos(rx), math.sin(rx)
    cy, sy_ = math.cos(ry), math.sin(ry)
    cz, sz_ = math.cos(rz), math.sin(rz)
    # R = Rz @ Ry @ Rx, then T · R · S; precomputed for speed
    r00 = cz * cy
    r01 = cz * sy_ * sx_ - sz_ * cx
    r02 = cz * sy_ * cx + sz_ * sx_
    r10 = sz_ * cy
    r11 = sz_ * sy_ * sx_ + cz * cx
    r12 = sz_ * sy_ * cx - cz * sx_
    r20 = -sy_
    r21 = cy * sx_
    r22 = cy * cx
    return (
        (r00 * sx, r01 * sy, r02 * sz, tx),
        (r10 * sx, r11 * sy, r12 * sz, ty),
        (r20 * sx, r21 * sy, r22 * sz, tz),
        (0.0,     0.0,     0.0,     1.0),
    )


def _parse_transform_link(cluster_node: FBXNode, source: bytes) -> Optional[tuple[float, float, float]]:
    """Read cluster bind-pose world matrix (`TransformLink: *16 { a: ... }`)
    and return its translation column. None if absent."""
    tl = cluster_node.child("TransformLink")
    if tl is None:
        return None
    a = tl.child("a")
    if a is None:
        return None
    args = parse_args(a.args_bytes(source))
    nums = [_num(arg) for arg in args if arg[0] == "num"]
    if len(nums) < 16:
        return None
    # FBX TransformLink is column-major: translation is at flat indices 12,13,14.
    return (float(nums[12]), float(nums[13]), float(nums[14]))


def _cluster_weight_count(cluster: FBXNode, source: bytes) -> int:
    """Read `Indexes: *N` from a Cluster body. 0 if absent (some clusters have
    no actual skinned vertices)."""
    indexes = cluster.child("Indexes")
    if indexes is None:
        return 0
    # args text is something like "*266"
    raw = indexes.args_bytes(source).strip()
    if raw.startswith(b"*"):
        # Strip '*' and any trailing whitespace/garbage before '{'
        s = raw[1:].split()[0] if raw[1:].strip() else b"0"
        try:
            return int(s)
        except ValueError:
            return 0
    return 0


def extract(doc: FBXDocument) -> SectionView:
    """Build a SectionView from a parsed ASCII FBX document.

    Walks Objects (Models + Deformers) and Connections (OO links). Computes
    per-bone parent/child/cluster info in three passes.
    """
    source = doc.source
    objects = doc.root("Objects")
    if objects is None:
        raise SectionError("FBX has no Objects section")

    models: dict[int, dict] = {}     # model_id -> {full_name, type_class, translation, node}
    clusters: dict[int, ClusterRecord] = {}
    materials: dict[int, MaterialRecord] = {}
    blend_shapes: dict[int, BlendShapeRecord] = {}
    blend_shape_channels: dict[int, BlendShapeChannelRecord] = {}
    geometries: dict[int, GeometryRecord] = {}
    skins: dict[int, SkinRecord] = {}

    for node in objects.children:
        if node.name == "Model":
            mid, full_name, type_class = _parse_object_args(parse_args(node.args_bytes(source)))
            if mid is None:
                continue
            translation = _extract_translation(node, source)
            models[mid] = {
                "full_name": full_name,
                "type_class": type_class,
                "translation": translation,
                "node": node,
            }
        elif node.name == "Deformer":
            did, full_name, type_class = _parse_object_args(parse_args(node.args_bytes(source)))
            if did is None:
                continue
            if type_class == "Cluster":
                short = _strip_prefix(full_name, _SUBDEFORMER_PREFIX)
                short = _strip_prefix(short, _DEFORMER_PREFIX)
                clusters[did] = ClusterRecord(
                    cluster_id=did,
                    name=short,
                    weight_count=_cluster_weight_count(node, source),
                    node_ref=node,
                )
            elif type_class == "BlendShape":
                short = _strip_prefix(full_name, _DEFORMER_PREFIX)
                blend_shapes[did] = BlendShapeRecord(
                    deformer_id=did,
                    name=short,
                    full_name=full_name,
                    node_ref=node,
                )
            elif type_class == "BlendShapeChannel":
                short = _strip_prefix(full_name, _SUBDEFORMER_PREFIX)
                short = _strip_prefix(short, _DEFORMER_PREFIX)
                blend_shape_channels[did] = BlendShapeChannelRecord(
                    channel_id=did,
                    name=short,
                    full_name=full_name,
                    blend_shape_id=None,   # resolved in pass 2
                    node_ref=node,
                )
            elif type_class == "Skin":
                short = _strip_prefix(full_name, _DEFORMER_PREFIX)
                skins[did] = SkinRecord(
                    skin_id=did,
                    name=short,
                    full_name=full_name,
                    geometry_id=None,   # resolved in pass 2
                    node_ref=node,
                )
        elif node.name == "Geometry":
            gid, full_name, _ = _parse_object_args(parse_args(node.args_bytes(source)))
            if gid is None:
                continue
            short = _strip_prefix(full_name, _GEOMETRY_PREFIX)
            geometries[gid] = GeometryRecord(
                geometry_id=gid,
                name=short,
                full_name=full_name,
                node_ref=node,
            )
        elif node.name == "Material":
            mid, full_name, _ = _parse_object_args(parse_args(node.args_bytes(source)))
            if mid is None:
                continue
            short = _strip_prefix(full_name, _MATERIAL_PREFIX)
            materials[mid] = MaterialRecord(
                material_id=mid,
                name=short,
                full_name=full_name,
                node_ref=node,
            )

    # Pass 2: Connections — derive parent_of[child_model] and clusters_on[model]
    parent_of: dict[int, int] = {}
    clusters_on: dict[int, list[int]] = {}
    cluster_skin: dict[int, int] = {}
    geometry_owner_model: dict[int, int] = {}

    connections = doc.root("Connections")
    if connections is not None:
        for c in connections.children:
            if c.name != "C":
                continue
            args = parse_args(c.args_bytes(source))
            if not args:
                continue
            ctype = _str(args[0])
            if ctype != "OO":
                continue
            if len(args) < 3:
                continue
            src_id = _num(args[1])
            dst_id = _num(args[2])
            if src_id is None or dst_id is None:
                continue
            src_id_i = int(src_id)
            dst_id_i = int(dst_id)
            # FBX OO conventions:
            #   parent-child:  src=child Model, dst=parent Model (dst=0 means RootNode)
            #   skin attach:   src=bone Model, dst=Cluster (FBX SDK convention)
            #                  src=Cluster,    dst=bone Model (some exporters)
            #   mesh chain:    src=Geometry,   dst=Mesh-type Model (owner)
            #                  src=Skin,       dst=Geometry         (skin attached)
            #                  src=Cluster,    dst=Skin             (cluster in skin)
            #                  src=BlendShape, dst=Geometry         (blend on mesh)
            if src_id_i in models and (dst_id_i == 0 or dst_id_i in models):
                parent_of[src_id_i] = dst_id_i
            elif src_id_i in models and dst_id_i in clusters:
                clusters_on.setdefault(src_id_i, []).append(dst_id_i)
            elif src_id_i in clusters and dst_id_i in models:
                clusters_on.setdefault(dst_id_i, []).append(src_id_i)
            elif src_id_i in clusters and dst_id_i in skins:
                cluster_skin[src_id_i] = dst_id_i
            elif src_id_i in skins and dst_id_i in geometries:
                skins[src_id_i].geometry_id = dst_id_i
            elif src_id_i in geometries and dst_id_i in models:
                geometry_owner_model[src_id_i] = dst_id_i
            elif src_id_i in blend_shape_channels and dst_id_i in blend_shapes:
                # channel owned-by BlendShape
                blend_shape_channels[src_id_i].blend_shape_id = dst_id_i

    # Pass 3: build BoneRecords
    children_of: dict[int, list[int]] = {}
    for child, parent in parent_of.items():
        children_of.setdefault(parent, []).append(child)

    bones: dict[int, BoneRecord] = {}
    for mid, mi in models.items():
        short_name = _strip_prefix(mi["full_name"], _MODEL_PREFIX)
        pid = parent_of.get(mid)
        parent_name: Optional[str] = None
        if pid is not None and pid in models:
            parent_name = _strip_prefix(models[pid]["full_name"], _MODEL_PREFIX)
        # pid == 0 → RootNode, parent_name stays None
        kids = children_of.get(mid, [])
        child_names = [_strip_prefix(models[k]["full_name"], _MODEL_PREFIX)
                       for k in kids if k in models]
        attached = clusters_on.get(mid, [])
        weight_count = sum(clusters[cid].weight_count for cid in attached if cid in clusters)

        bones[mid] = BoneRecord(
            model_id=mid,
            name=short_name,
            full_name=mi["full_name"],
            type_class=mi["type_class"],
            parent_id=pid,
            parent_name=parent_name,
            children_ids=kids,
            child_names=child_names,
            translation_xyz=mi["translation"],
            has_skin_cluster=len(attached) > 0,
            cluster_weight_count=weight_count,
            cluster_ids=attached,
            node_ref=mi["node"],
        )

    _populate_world_positions(bones, clusters, models, source)

    return SectionView(
        bones=bones,
        clusters=clusters,
        document=doc,
        materials=materials,
        blend_shapes=blend_shapes,
        blend_shape_channels=blend_shape_channels,
        geometries=geometries,
        skins=skins,
        cluster_skin=cluster_skin,
        geometry_owner_model=geometry_owner_model,
    )


# --- mesh affinity (bone → meshes deformed) -------------------------------


def bone_to_mesh_names(view: SectionView, bone_id: int) -> list[str]:
    """Return the display names of meshes that `bone_id` (or any descendant)
    actually deforms.

    Two FBX quirks shape this:
      (1) Maya rigs add every bone to every mesh's Skin even with zero weights.
          A cluster with `weight_count == 0` is not a real binding — skip it.
      (2) Chain roots (PB_skirt_root, Hat_root, etc.) carry zero direct weights
          but their descendants deform real meshes. Walk the subtree so the
          root reports the meshes its chain influences. This matches the user's
          mental model: dropping a chain root drops the whole chain.

    Mesh-type Models hold a Geometry directly — they report their own name.

    Deduplicated; stable order = first-seen order from a pre-order walk.
    """
    bone = view.bones.get(bone_id)
    if bone is None:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name and name not in seen:
            seen.add(name)
            out.append(name)

    def _affinity_for(b: BoneRecord) -> None:
        if b.type_class == "Mesh":
            # The owning Mesh-Model display name IS the mesh's user-facing label.
            _add(b.name)
        for cid in b.cluster_ids:
            cluster = view.clusters.get(cid)
            if cluster is None or cluster.weight_count <= 0:
                continue
            skin_id = view.cluster_skin.get(cid)
            if skin_id is None:
                continue
            skin = view.skins.get(skin_id)
            if skin is None or skin.geometry_id is None:
                continue
            owner_model_id = view.geometry_owner_model.get(skin.geometry_id)
            if owner_model_id is None:
                geom = view.geometries.get(skin.geometry_id)
                if geom is not None:
                    _add(geom.name)
                continue
            owner = view.bones.get(owner_model_id)
            if owner is not None:
                _add(owner.name)

    # Pre-order subtree walk with cycle guard.
    visited: set[int] = set()
    stack: list[int] = [bone_id]
    while stack:
        bid = stack.pop()
        if bid in visited:
            continue
        visited.add(bid)
        b = view.bones.get(bid)
        if b is None:
            continue
        _affinity_for(b)
        # Push children in reverse so pop order matches FBX child order.
        for cid in reversed(b.children_ids):
            stack.append(cid)

    return out


# --- world-position pass -----------------------------------------------------

# Dead-zone around scene center for lateral/front_back bucketing. Tuned to
# scene units (meters); a chest-centered "C" bone (e.g. Spine) should not
# pick up an "L"/"R" tag from float noise.
_AXIS_DEADZONE = 0.02


def _populate_world_positions(
    bones: dict[int, BoneRecord],
    clusters: dict[int, ClusterRecord],
    models: dict[int, dict],
    source: bytes,
) -> None:
    """Fill world_xyz + bucketed fields on every BoneRecord, in place.

    Primary source: each bone's cluster `TransformLink` translation (bind-pose
    world matrix). Fallback: walk the parent chain composing T·R·S local
    matrices into a world matrix. Bones with no clusters and no resolvable
    parent stay at the default (0,0,0) — they end up bucketed as "C".
    """
    # Cluster world translations (cheap lookup).
    cluster_world: dict[int, tuple[float, float, float]] = {}
    for cid, c in clusters.items():
        pos = _parse_transform_link(c.node_ref, source)
        if pos is not None:
            cluster_world[cid] = pos

    # Pre-extract Lcl Rotation / Scaling per model — used only for the
    # parent-chain fallback path.
    locals_by_id: dict[int, tuple] = {}
    def _local_for(mid: int) -> tuple:
        if mid in locals_by_id:
            return locals_by_id[mid]
        mi = models.get(mid)
        if mi is None:
            return _IDENTITY_4
        node = mi["node"]
        t = mi["translation"]
        r = _extract_props_vec3(node, source, "Lcl Rotation", (0.0, 0.0, 0.0))
        s = _extract_props_vec3(node, source, "Lcl Scaling", (1.0, 1.0, 1.0))
        m = _local_matrix(t, r, s)
        locals_by_id[mid] = m
        return m

    # Memoized world transforms. Used only for the fallback path; bones with a
    # cluster short-circuit to that translation.
    world_by_id: dict[int, tuple[float, float, float]] = {}

    def _world_for(mid: int, seen: Optional[set] = None) -> tuple[float, float, float]:
        if mid in world_by_id:
            return world_by_id[mid]
        b = bones.get(mid)
        if b is None:
            return (0.0, 0.0, 0.0)
        # Trusted path: any cluster on this bone has the bind-pose world matrix.
        for cid in b.cluster_ids:
            if cid in cluster_world:
                world_by_id[mid] = cluster_world[cid]
                return world_by_id[mid]
        # Fallback path: walk up through local matrices.
        seen = seen if seen is not None else set()
        if mid in seen:
            world_by_id[mid] = (0.0, 0.0, 0.0)
            return world_by_id[mid]
        seen.add(mid)
        local = _local_for(mid)
        pid = b.parent_id
        if pid is None or pid == 0 or pid not in bones:
            # No resolvable parent — local IS world.
            world_by_id[mid] = (local[0][3], local[1][3], local[2][3])
            return world_by_id[mid]
        # If the parent has a cluster, use its trusted world matrix as the
        # base; otherwise recurse.
        parent_world_t = _world_for(pid, seen)
        # Compose: world = parent_world @ local. We only need the translation
        # column, but we need the full parent rotation to transform local's
        # translation correctly. Reconstruct parent world matrix:
        parent_local = _local_for(pid)
        # If parent_world_t came from a cluster, the parent's *rotation* in the
        # world frame may differ from parent_local's. We accept the small
        # inconsistency: the dominant signal is translation; bucketing is
        # robust to a few cm of rotation-induced offset. The exact case where
        # this matters (parent has cluster, child doesn't, parent has
        # non-trivial rotation) is rare for the `_end` markers that take this
        # path. Better-than-translation-sum, worse-than-numpy.
        pw00, pw01, pw02 = parent_local[0][0], parent_local[0][1], parent_local[0][2]
        pw10, pw11, pw12 = parent_local[1][0], parent_local[1][1], parent_local[1][2]
        pw20, pw21, pw22 = parent_local[2][0], parent_local[2][1], parent_local[2][2]
        lt0, lt1, lt2 = local[0][3], local[1][3], local[2][3]
        wt = (
            parent_world_t[0] + pw00 * lt0 + pw01 * lt1 + pw02 * lt2,
            parent_world_t[1] + pw10 * lt0 + pw11 * lt1 + pw12 * lt2,
            parent_world_t[2] + pw20 * lt0 + pw21 * lt1 + pw22 * lt2,
        )
        world_by_id[mid] = wt
        return wt

    for mid in bones:
        _world_for(mid)

    # Scene Y bounds for height_pct normalization.
    ys = [w[1] for w in world_by_id.values()]
    y_min = min(ys) if ys else 0.0
    y_max = max(ys) if ys else 0.0
    y_span = y_max - y_min

    for mid, b in bones.items():
        wx, wy, wz = world_by_id.get(mid, (0.0, 0.0, 0.0))
        b.world_xyz = (wx, wy, wz)
        b.height_pct = (wy - y_min) / y_span if y_span > 0 else 0.0
        b.lateral = (
            "L" if wx > _AXIS_DEADZONE
            else "R" if wx < -_AXIS_DEADZONE
            else "C"
        )
        b.front_back = (
            "F" if wz > _AXIS_DEADZONE
            else "B" if wz < -_AXIS_DEADZONE
            else "C"
        )
