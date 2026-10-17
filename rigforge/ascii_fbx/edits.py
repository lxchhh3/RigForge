"""Atomic ASCII FBX edits.

All edits are byte-level text operations on the original source bytes. They
preserve unrelated nodes verbatim (round-trip identity for everything we
didn't touch).

Edit model
----------
- `TextEdit(start, end, replacement)` describes a splice: source[start:end] is
  replaced with `replacement` (bytes).
- A list of TextEdits is sorted by start offset, checked for non-overlap, and
  applied in a single pass via `apply_edits()`.
- Two convention edits:
    * `drop_node_edit`   removes a Model/Cluster/Connection body+line cleanly
    * `rename_bone_edits` rewrites the bone-side display name everywhere
                          (Model arg + every cluster's SubDeformer arg)

PLAN.md guarantees the LLM never sees the text it modifies; this module is the
sole writer of FBX bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Iterable, Optional

from .lexer import FBXDocument, FBXNode
from .sections import BoneRecord, ClusterRecord, SectionView, parse_args, _num, _str


class EditError(RuntimeError):
    """Raised on invalid or conflicting edits."""


@dataclass(frozen=True)
class TextEdit:
    start: int
    end: int             # exclusive
    replacement: bytes


def apply_edits(source: bytes, edits: Iterable[TextEdit]) -> bytes:
    """Apply a set of TextEdits to source bytes, producing a new bytes object.

    Edits must be non-overlapping. Order of application is by start offset.
    """
    sorted_edits = sorted(edits, key=lambda e: e.start)
    last_end = -1
    for e in sorted_edits:
        if e.start < 0 or e.end > len(source) or e.start > e.end:
            raise EditError(f"out-of-bounds edit: {e}")
        if e.start < last_end:
            raise EditError(
                f"overlapping edits at offset {e.start}; previous edit ended at {last_end}"
            )
        last_end = e.end
    parts: list[bytes] = []
    cursor = 0
    for e in sorted_edits:
        parts.append(source[cursor : e.start])
        parts.append(e.replacement)
        cursor = e.end
    parts.append(source[cursor:])
    return b"".join(parts)


# ---------------------------------------------------------------------------
# Drop helpers
# ---------------------------------------------------------------------------


def _line_drop_range(node: FBXNode, source: bytes) -> tuple[int, int]:
    """Compute the byte range to delete to remove `node` (and the line(s) it
    occupies) cleanly.

    Strategy:
      - drop_start: walk back from node.full_start past `\\t`/` ` until we hit
        a `\\n` or start-of-file. Stops just AFTER the prior newline.
      - natural_end: for body nodes, just past `}`. For leaves, the first
        `\\n` at or after `intro_end`.
      - drop_end: walk forward from natural_end past `\\t`/` ` and consume one
        `\\n` so the trailing line break is removed too.
    """
    n = len(source)

    # Walk backward to start-of-line.
    i = node.name_span[0]
    while i > 0 and source[i - 1] in b" \t":
        i -= 1
    drop_start = i

    # Compute natural end.
    if node.has_body:
        assert node.body_close is not None
        natural_end = node.body_close + 1
    else:
        j = node.intro_end
        while j < n and source[j] != 0x0A:
            j += 1
        natural_end = j

    # Walk forward to consume trailing horizontal whitespace + one newline.
    j = natural_end
    while j < n and source[j] in b" \t":
        j += 1
    if j < n and source[j] == 0x0A:
        j += 1
    drop_end = j
    return (drop_start, drop_end)


def drop_node_edit(node: FBXNode, source: bytes) -> TextEdit:
    """Edit that removes a node and the whitespace/newline owning its line."""
    s, e = _line_drop_range(node, source)
    return TextEdit(s, e, b"")


# ---------------------------------------------------------------------------
# High-level operations on the section view
# ---------------------------------------------------------------------------


def drop_bone_edits(view: SectionView, bone: BoneRecord) -> list[TextEdit]:
    """Edits required to fully remove `bone` from the document:
       - the Model node itself
       - every Cluster attached to it
       - every Connection involving the bone or its clusters
    """
    source = view.document.source
    edits: list[TextEdit] = [drop_node_edit(bone.node_ref, source)]

    for cid in bone.cluster_ids:
        cluster = view.clusters.get(cid)
        if cluster is not None:
            edits.append(drop_node_edit(cluster.node_ref, source))

    edits.extend(_drop_connections_referencing(
        view, ids={bone.model_id, *bone.cluster_ids}
    ))
    return edits


def drop_cluster_edits(view: SectionView, cluster: ClusterRecord) -> list[TextEdit]:
    """Drop just the Cluster and connections that reference it."""
    source = view.document.source
    edits: list[TextEdit] = [drop_node_edit(cluster.node_ref, source)]
    edits.extend(_drop_connections_referencing(view, ids={cluster.cluster_id}))
    return edits


def mesh_geometry_ids(view: SectionView, mesh_model_id: int) -> set[int]:
    """Geometry ids owned by a Mesh-type Model."""
    return {
        gid for gid, mid in view.geometry_owner_model.items()
        if mid == mesh_model_id
    }


def mesh_skin_ids(view: SectionView, mesh_model_id: int) -> set[int]:
    """Skin Deformer ids bound to a Mesh-type Model's geometries."""
    geom_ids = mesh_geometry_ids(view, mesh_model_id)
    return {
        s.skin_id for s in view.skins.values()
        if s.geometry_id is not None and s.geometry_id in geom_ids
    }


def mesh_cluster_ids(view: SectionView, mesh_model_id: int) -> set[int]:
    """Cluster ids that skin a Mesh-type Model (Model -> Geometry -> Skin ->
    Cluster).

    Shared by `drop_mesh_edits` (which removes them) and by callers that must
    avoid emitting edits INSIDE a Cluster block another pass is deleting — see
    `rename_bone_edits(skip_cluster_ids=...)`.
    """
    skin_ids = mesh_skin_ids(view, mesh_model_id)
    return {cid for cid, sid in view.cluster_skin.items() if sid in skin_ids}


def mesh_blend_shape_channel_ids(view: SectionView, mesh_model_id: int) -> set[int]:
    """BlendShapeChannel ids that `drop_mesh_edits` would cascade-delete for
    this mesh (Model -> Geometry -> BlendShape deformer -> channels).

    Callers that emit edits INSIDE a channel node (DeformPercent overrides,
    channel renames) must skip these when the mesh is being dropped in the
    same pass — otherwise the two edits overlap and `apply_edits` raises.
    """
    geom_ids = mesh_geometry_ids(view, mesh_model_id)
    if not geom_ids:
        return set()
    bs_ids = {
        s for s, d in iter_oo_connections(view)
        if d in geom_ids and s in view.blend_shapes
    }
    if not bs_ids:
        return set()
    return {
        s for s, d in iter_oo_connections(view)
        if d in bs_ids and s in view.blend_shape_channels
    }


def drop_mesh_edits(view: SectionView, mesh_model_id: int) -> list[TextEdit]:
    """Edits required to remove a whole mesh chain from the document:
       - the Mesh-type Model node
       - every Geometry connected to that Model
       - every Skin Deformer connected to that Geometry
       - every Cluster SubDeformer connected to that Skin
       - every BlendShape Deformer attached to that Geometry, plus its
         BlendShapeChannels and the Shape geometries those channels reference
       - every Connection referencing any of the above

    Returns [] if `mesh_model_id` isn't a Mesh-type Model in the view — this
    keeps the API call site safe against stale ids without raising.

    Bones are intentionally untouched: a bone often deforms multiple meshes
    (Maya rigs every bone into every skin), so cascading bone removal here
    would corrupt sibling meshes. Bone cleanup is a separate concern.

    Blendshape cascade rationale: without it, the channel/shape decls survive
    in the file but the Geometry→BlendShape connection is severed when the
    mesh drops, leaving "orphan" morphs that never render in Blender/Maya.
    The cascade only fires when the BlendShape is attached to the dropped
    Geometry — BlendShapes on other meshes are untouched.
    """
    bone = view.bones.get(mesh_model_id)
    if bone is None or bone.type_class != "Mesh":
        return []
    source = view.document.source
    edits: list[TextEdit] = [drop_node_edit(bone.node_ref, source)]

    geom_ids = mesh_geometry_ids(view, mesh_model_id)
    skin_ids = mesh_skin_ids(view, mesh_model_id)
    cluster_ids = mesh_cluster_ids(view, mesh_model_id)

    # Cascade into the blendshape graph attached to the dropped Geometry.
    # FBX convention: src=BlendShape, dst=Geometry  (deformer attached to mesh).
    # We walk one pass over Connections, bucketing by direction, so we hit
    # each row at most twice (BS-on-geom, channel-on-BS, shape-on-channel).
    bs_ids: set[int] = set()
    channel_ids: set[int] = set()
    shape_geom_ids: set[int] = set()
    oo_rows = list(iter_oo_connections(view))
    for s, d in oo_rows:
        if d in geom_ids and s in view.blend_shapes:
            bs_ids.add(s)
    if bs_ids:
        for s, d in oo_rows:
            if d in bs_ids and s in view.blend_shape_channels:
                channel_ids.add(s)
    if channel_ids:
        for s, d in oo_rows:
            # Shape geometries point at channels and live in `geometries` too,
            # but the dropped mesh geom is already in geom_ids — skip those.
            if (d in channel_ids and s in view.geometries
                    and s not in geom_ids):
                shape_geom_ids.add(s)

    for gid in geom_ids | shape_geom_ids:
        g = view.geometries.get(gid)
        if g is not None:
            edits.append(drop_node_edit(g.node_ref, source))
    for sid in skin_ids:
        s = view.skins.get(sid)
        if s is not None:
            edits.append(drop_node_edit(s.node_ref, source))
    for cid in cluster_ids:
        c = view.clusters.get(cid)
        if c is not None:
            edits.append(drop_node_edit(c.node_ref, source))
    for bid in bs_ids:
        bs = view.blend_shapes.get(bid)
        if bs is not None:
            edits.append(drop_node_edit(bs.node_ref, source))
    for cid in channel_ids:
        ch = view.blend_shape_channels.get(cid)
        if ch is not None:
            edits.append(drop_node_edit(ch.node_ref, source))

    edits.extend(_drop_connections_referencing(
        view, ids={
            mesh_model_id, *geom_ids, *skin_ids, *cluster_ids,
            *bs_ids, *channel_ids, *shape_geom_ids,
        },
    ))
    return edits


def iter_oo_connections(view: SectionView):
    """Yield (src_id, dst_id) for every `C: "OO", src, dst` row in Connections.

    Skips rows that don't have two numeric ids after the relation tag.
    """
    source = view.document.source
    connections = view.document.root("Connections")
    if connections is None:
        return
    for c in connections.children:
        if c.name != "C":
            continue
        args = parse_args(c.args_bytes(source))
        if len(args) < 3:
            continue
        src = _num(args[1])
        dst = _num(args[2])
        if src is None or dst is None:
            continue
        yield int(src), int(dst)


def set_blend_shape_channel_deform_percent_edits(
    view: SectionView, channel_id: int, percent: float,
) -> list[TextEdit]:
    """Edit the `DeformPercent` value of a BlendShapeChannel SubDeformer.

    `percent` is the morph intensity baked into the assembled FBX. Range is
    nominally 0-100 (matches Maya's slider UI) but the FBX format itself
    accepts any float. Values outside [0, 100] are passed through verbatim;
    the caller is responsible for clamping if it cares.

    Returns [] when:
      - `channel_id` doesn't resolve to a channel in the view (stale FE id)
      - the channel node has no DeformPercent child (malformed FBX or older
        exporter that uses a different property name)

    Both failures are silent so a stale or partial id list doesn't break the
    whole assemble run.
    """
    ch = view.blend_shape_channels.get(channel_id)
    if ch is None:
        return []
    source = view.document.source
    for child in ch.node_ref.children:
        if child.name != "DeformPercent":
            continue
        args_start, args_end = child.args_span
        region = source[args_start:args_end]
        # Reuse the positioned-args tokenizer from merge.py so we hit the
        # numeric token exactly and preserve surrounding whitespace.
        from .merge import positioned_args
        for tok in positioned_args(region):
            if tok.kind == "num":
                return [TextEdit(
                    args_start + tok.start,
                    args_start + tok.end,
                    _format_percent_value(percent).encode("ascii"),
                )]
        return []
    return []


def _format_percent_value(value: float) -> str:
    """Format a DeformPercent value the way Maya itself does: bare integers
    when whole, %g for fractions. Avoids '50.0' vs '50' churn that bloats
    diffs without changing semantics."""
    if value == int(value):
        return str(int(value))
    return f"{value:.6g}"


# ---------------------------------------------------------------------------
# Add a scaled copy of a blendshape channel
# ---------------------------------------------------------------------------


def _object_header_id(node: FBXNode, source: bytes) -> Optional[int]:
    args = parse_args(node.args_bytes(source))
    if args and args[0][0] == "num":
        return int(args[0][1])
    return None


def _array_values(node: FBXNode, source: bytes) -> Optional[list]:
    """The numbers of an `X: *N { a: ... }` child, or None if absent."""
    a = node.child("a")
    if a is None:
        return None
    return [_num(t) for t in parse_args(a.args_bytes(source)) if t[0] == "num"]


def _fmt_num(v: float) -> str:
    return str(int(v)) if v == int(v) else f"{v:.15g}"


def add_scaled_blend_shape_channel_edits(
    view: SectionView,
    source_channel_id: int,
    new_name: str,
    factor: float,
    *,
    deform_percent: float = 0.0,
) -> list[TextEdit]:
    """Add a BlendShapeChannel `new_name` on the same BlendShape deformer as
    `source_channel_id`, whose Shape is the source's: the same vertex
    Indexes, its Vertices and Normals deltas times `factor`. At 100 it does
    what the source would at `factor` x 100 — more of the author's own
    motion, as a key of its own (GG_headset's 内 -> 内_neck, 2026-10-15).

    The Shape and channel go in right after the source channel's node, their
    two `C: "OO"` links at the end of Connections, with ids above every
    Objects id. Raises EditError for an unknown source, a source without
    exactly one Shape, or a name the deformer already has.
    """
    src_ch = view.blend_shape_channels.get(source_channel_id)
    if src_ch is None:
        raise EditError(f"no blendshape channel with id {source_channel_id}")
    if src_ch.blend_shape_id is None:
        raise EditError(f"channel {src_ch.name!r} is not linked to a BlendShape deformer")
    if any(c.name == new_name and c.blend_shape_id == src_ch.blend_shape_id
           for c in view.blend_shape_channels.values()):
        raise EditError(f"the deformer of {src_ch.name!r} already has a channel {new_name!r}")

    source = view.document.source
    objects = view.document.root("Objects")
    connections = view.document.root("Connections")
    if objects is None or connections is None or connections.body_close is None:
        raise EditError("document has no Objects / Connections section")
    by_id = {}
    for obj in objects.children:
        oid = _object_header_id(obj, source)
        if oid is not None:
            by_id[oid] = obj
    shapes = [s for s, d in iter_oo_connections(view)
              if d == source_channel_id and s in by_id and by_id[s].name == "Geometry"]
    if len(shapes) != 1:
        raise EditError(f"channel {src_ch.name!r} has {len(shapes)} shapes, expected 1")
    shape = by_id[shapes[0]]

    arrays = {}
    for child in shape.children:
        if child.name in ("Indexes", "Vertices", "Normals"):
            arrays[child.name] = _array_values(child, source)
    if arrays.get("Indexes") is None or arrays.get("Vertices") is None:
        raise EditError(f"shape of {src_ch.name!r} has no Indexes / Vertices")

    new_shape_id = max(by_id) + 1
    new_ch_id = max(by_id) + 2
    line_start = source.rfind(b"\n", 0, shape.name_span[0]) + 1
    ind = source[line_start:shape.name_span[0]].decode("ascii")

    def block(name: str, values, *, ints: bool = False) -> str:
        nums = ",".join(str(int(v)) if ints else _fmt_num(v * factor) for v in values)
        return (f"{ind}\t{name}: *{len(values)} {{\n{ind}\t\ta: {nums}\n{ind}\t}} \n")

    text = (
        f'{ind}Geometry: {new_shape_id}, "Geometry::{new_name}", "Shape" {{\n'
        f"{ind}\tVersion: 100\n"
        + block("Indexes", arrays["Indexes"], ints=True)
        + block("Vertices", arrays["Vertices"])
        + (block("Normals", arrays["Normals"]) if arrays.get("Normals") is not None else "")
        + f"{ind}}}\n"
        f'{ind}Deformer: {new_ch_id}, "SubDeformer::{new_name}", "BlendShapeChannel" {{\n'
        f"{ind}\tVersion: 100\n"
        f"{ind}\tDeformPercent: {_format_percent_value(deform_percent)}\n"
        f"{ind}\tFullWeights: *1 {{\n{ind}\t\ta: 100\n{ind}\t}} \n"
        f"{ind}}}\n"
    )
    if src_ch.node_ref.body_close is None:
        raise EditError(f"channel {src_ch.name!r} node has no body")
    after_channel = source.find(b"\n", src_ch.node_ref.body_close)
    after_channel = len(source) if after_channel == -1 else after_channel + 1

    close_line = source.rfind(b"\n", 0, connections.body_close) + 1
    conn_ind = "\t"
    for c in connections.children:
        if c.name == "C":
            ls = source.rfind(b"\n", 0, c.name_span[0]) + 1
            conn_ind = source[ls:c.name_span[0]].decode("ascii")
            break
    links = (f'{conn_ind}C: "OO",{new_shape_id},{new_ch_id}\n'
             f'{conn_ind}C: "OO",{new_ch_id},{src_ch.blend_shape_id}\n')
    return [
        TextEdit(after_channel, after_channel, text.encode("utf-8")),
        TextEdit(close_line, close_line, links.encode("ascii")),
    ]


# ---------------------------------------------------------------------------
# Bone move: rest-pose local transform of a Model node
# ---------------------------------------------------------------------------

_LCL_PROPERTY_NAMES = ("Lcl Translation", "Lcl Rotation", "Lcl Scaling")


def set_model_local_transform_edits(
    view: SectionView,
    model_id: int,
    *,
    translation: Optional[tuple[float, float, float]] = None,
    rotation_deg: Optional[tuple[float, float, float]] = None,
    scale: Optional[tuple[float, float, float]] = None,
) -> list[TextEdit]:
    """Edits that set a Model's rest-pose local transform (FBX frame).

    Rewrites the numeric values of `Lcl Translation` / `Lcl Rotation` /
    `Lcl Scaling` in the Model's Properties70 block; a property the exporter
    elided (bones at the default rotation/scale routinely have no line) is
    INSERTED with the same indentation as its siblings. Components passed as
    None are left alone.

    Why this moves the mesh: the skin clusters' bind pose (`TransformLink`)
    is deliberately untouched, so every vertex weighted to the bone is
    evaluated against the old bind and the new rest transform — the mesh
    follows the bone rigidly. Identical to dragging the bone in Unity.

    Returns [] for an unknown id or when nothing is requested. Raises
    EditError when the Model has no Properties70 block or a property line
    is malformed (fewer than three numbers) — those are file problems the
    caller should surface, not silently skip.
    """
    bone = view.bones.get(model_id)
    if bone is None:
        return []
    wanted: list[tuple[str, tuple[float, float, float]]] = [
        (name, vec) for name, vec in zip(
            _LCL_PROPERTY_NAMES, (translation, rotation_deg, scale),
        ) if vec is not None
    ]
    if not wanted:
        return []

    source = view.document.source
    props = bone.node_ref.child("Properties70")
    if props is None:
        raise EditError(
            f"Model {model_id} ({bone.name!r}) has no Properties70 block; "
            f"cannot set its local transform"
        )
    p_children = props.child_all("P")
    existing: dict[str, FBXNode] = {}
    for p in p_children:
        args = parse_args(p.args_bytes(source))
        name = _str(args[0]) if args else None
        if name in _LCL_PROPERTY_NAMES and name not in existing:
            existing[name] = p

    from .merge import positioned_args

    edits: list[TextEdit] = []
    inserts: list[bytes] = []
    insert_at, indent = _properties70_insert_point(props, p_children, source)
    for prop_name, vec in wanted:
        formatted = _format_vec3(vec)
        p = existing.get(prop_name)
        if p is None:
            inserts.append(
                indent + f'P: "{prop_name}", "{prop_name}", "", "A",'.encode("ascii")
                + formatted + b"\n"
            )
            continue
        args_start, args_end = p.args_span
        nums = [t for t in positioned_args(source[args_start:args_end]) if t.kind == "num"]
        if len(nums) < 3:
            raise EditError(
                f"Model {model_id} ({bone.name!r}): {prop_name} has "
                f"{len(nums)} numeric values, expected 3"
            )
        first, last = nums[-3], nums[-1]
        edits.append(TextEdit(args_start + first.start, args_start + last.end, formatted))
    if inserts:
        edits.append(TextEdit(insert_at, insert_at, b"".join(inserts)))
    return edits


def _properties70_insert_point(
    props: FBXNode, p_children: list[FBXNode], source: bytes,
) -> tuple[int, bytes]:
    """Where a new `P:` line goes inside a Properties70 block, and the
    indentation to give it: right after the last existing `P:` line (copying
    its indent), or, for an empty block, on the line after `{` one tab deeper
    than the closing brace."""
    if p_children:
        last = p_children[-1]
        line_start = source.rfind(b"\n", 0, last.name_span[0]) + 1
        indent = source[line_start:last.name_span[0]]
        # A leaf's extent_end swallows trailing whitespace (up to the next
        # token), so anchor on the first newline after its args instead.
        nl = source.find(b"\n", last.intro_end)
        if nl == -1:
            raise EditError("Properties70 block is not newline-terminated")
        return nl + 1, indent
    assert props.body_open is not None and props.body_close is not None
    close_line_start = source.rfind(b"\n", 0, props.body_close) + 1
    indent = source[close_line_start:props.body_close] + b"\t"
    nl = source.find(b"\n", props.body_open)
    if nl == -1 or nl > props.body_close:
        raise EditError("Properties70 block is not newline-terminated")
    return nl + 1, indent


def _format_vec3(vec: tuple[float, float, float]) -> bytes:
    """`x,y,z` the way the FBX SDK writes them: bare ints when whole,
    shortest round-trip repr otherwise (so 0.0787041112780571 survives)."""
    parts = []
    for v in vec:
        f = float(v)
        parts.append(str(int(f)) if f.is_integer() else repr(f))
    return ",".join(parts).encode("ascii")


def drop_blend_shape_channel_edits(
    view: SectionView, channel_id: int
) -> list[TextEdit]:
    """Edits required to remove a BlendShapeChannel SubDeformer cleanly:
       - the channel node itself
       - every Connection referencing the channel

    The owning BlendShape Deformer is intentionally NOT cascaded — it may
    still own other channels (FBX docs commonly group all morphs of a mesh
    under one BlendShape container). An empty BlendShape Deformer is benign
    in the FBX format; dropping it would risk corrupting other surviving
    morphs.

    Returns [] if the id isn't a channel in the view, so the API can pass a
    stale id without breaking the run.
    """
    ch = view.blend_shape_channels.get(channel_id)
    if ch is None:
        return []
    source = view.document.source
    edits: list[TextEdit] = [drop_node_edit(ch.node_ref, source)]
    edits.extend(_drop_connections_referencing(view, ids={channel_id}))
    return edits


def _drop_connections_referencing(view: SectionView, ids: set[int]) -> list[TextEdit]:
    """Find every `C: "OO", src, dst` leaf in the Connections section that
    touches any id in `ids`, and emit a drop edit for it."""
    edits: list[TextEdit] = []
    source = view.document.source
    connections = view.document.root("Connections")
    if connections is None:
        return edits
    for c in connections.children:
        if c.name != "C":
            continue
        args = parse_args(c.args_bytes(source))
        if len(args) < 3:
            continue
        src = _num(args[1])
        dst = _num(args[2])
        if src is None or dst is None:
            continue
        if int(src) in ids or int(dst) in ids:
            edits.append(drop_node_edit(c, source))
    return edits


def reparent_bone_edits(
    view: SectionView,
    bone: BoneRecord,
    new_parent_id: int,
) -> list[TextEdit]:
    """Edits to change `bone`'s parent in the Connections section.

    Flips the parent_id slot in the bone's `C: "OO", <bone_id>, <old_parent>`
    leaf. FBX parent-child convention is src=child, dst=parent (see
    sections.extract). We locate the leaf by scanning Connections for an OO
    row whose src equals `bone.model_id` AND whose dst equals the bone's
    current `parent_id`; we then splice just the dst token.

    No-op short-circuit: if the bone is already parented to `new_parent_id`,
    returns `[]` so apply_edits is a pure pass-through.

    Raises EditError if no parent connection is found (a free-floating
    Model with no OO row — caller should drop or wire it manually).
    """
    if bone.parent_id is not None and bone.parent_id == new_parent_id:
        return []

    source = view.document.source
    connections = view.document.root("Connections")
    if connections is None:
        raise EditError("FBX has no Connections section")

    target_conn: FBXNode | None = None
    dst_token_span: tuple[int, int] | None = None
    for c in connections.children:
        if c.name != "C":
            continue
        args_start, args_end = c.args_span
        region = source[args_start:args_end]
        toks = list(_positioned_args(region))
        if len(toks) < 3:
            continue
        ctype = toks[0]
        src_tok = toks[1]
        dst_tok = toks[2]
        if ctype.kind != "str" or ctype.value != "OO":
            continue
        if src_tok.kind != "num" or dst_tok.kind != "num":
            continue
        if int(src_tok.value) != bone.model_id:
            continue
        # Match the row that points at the CURRENT parent. This guards
        # against accidentally rewriting a cluster-attachment row (where the
        # bone is also the src, but the dst is a Cluster id).
        if bone.parent_id is None or int(dst_tok.value) != bone.parent_id:
            continue
        target_conn = c
        dst_token_span = (
            args_start + dst_tok.start,
            args_start + dst_tok.end,
        )
        break

    if target_conn is None or dst_token_span is None:
        raise EditError(
            f"no parent OO connection found for bone {bone.model_id} "
            f"(current parent_id={bone.parent_id})"
        )

    return [TextEdit(dst_token_span[0], dst_token_span[1],
                     str(new_parent_id).encode("ascii"))]


@dataclass(frozen=True)
class _ArgToken:
    kind: str           # "str" | "num"
    value: object
    start: int          # offset within the args region
    end: int            # exclusive


def _positioned_args(text: bytes) -> Iterable[_ArgToken]:
    """Like `parse_args` but yields per-token (start, end) offsets so we can
    splice a single argument slot in place. Mirrors the comment/whitespace
    handling in parse_args."""
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in b" \t\r\n":
            i += 1
            continue
        if c == 0x3B:  # ';' comment
            while i < n and text[i] != 0x0A:
                i += 1
            continue
        if c == 0x22:  # '"'
            start = i
            i += 1
            j = i
            while j < n and text[j] != 0x22:
                j += 1
            value = text[i:j].decode("utf-8", errors="replace")
            end = j + 1 if j < n else j
            yield _ArgToken("str", value, start, end)
            i = end
            continue
        if c == 0x2C:  # ','
            i += 1
            continue
        # bare token
        j = i
        while j < n and text[j] not in b",; \t\r\n":
            j += 1
        tok = text[i:j].decode("utf-8", errors="replace")
        start, end = i, j
        i = j
        if not tok:
            continue
        try:
            if "." in tok or "e" in tok or "E" in tok:
                yield _ArgToken("num", float(tok), start, end)
            else:
                yield _ArgToken("num", int(tok), start, end)
        except ValueError:
            yield _ArgToken("str", tok, start, end)


def rename_bone_edits(
    view: SectionView,
    bone: BoneRecord,
    new_short_name: str,
    *,
    skip_cluster_ids: Optional[AbstractSet[int]] = None,
) -> list[TextEdit]:
    """Edits to change the bone's display name from `bone.name` to
    `new_short_name`, touching:
       - the Model's quoted name `"Model::<old>"`
       - every attached Cluster's `"SubDeformer::<old>"` (if it matches)

    `skip_cluster_ids` suppresses the cluster-side rename for clusters another
    pass is removing wholesale. A mesh drop deletes entire Cluster blocks, and
    a bone that skins that mesh would otherwise queue a name rewrite INSIDE a
    deleted range — which `apply_edits` rejects as an overlap. The deletion
    wins; the rename there is moot. Callers that drop meshes pass those meshes'
    `mesh_cluster_ids`.
    """
    skip_cluster_ids = skip_cluster_ids or frozenset()
    source = view.document.source
    edits: list[TextEdit] = []

    edits.append(_replace_quoted_in_args(
        bone.node_ref, source,
        f'"{bone.full_name}"'.encode("utf-8"),
        f'"Model::{new_short_name}"'.encode("utf-8"),
    ))

    for cid in bone.cluster_ids:
        if cid in skip_cluster_ids:
            continue
        cluster = view.clusters.get(cid)
        if cluster is None:
            continue
        # Cluster's short-name might already differ (e.g., "SubDeformer::FooBone"
        # while the bone name was "Bar"). We only rename when the short names match.
        if cluster.name != bone.name:
            continue
        edits.append(_replace_quoted_in_args(
            cluster.node_ref, source,
            f'"SubDeformer::{cluster.name}"'.encode("utf-8"),
            f'"SubDeformer::{new_short_name}"'.encode("utf-8"),
        ))
    return edits


def rename_blend_shape_channel_edits(
    view: SectionView,
    channel_id: int,
    new_name: str,
) -> list[TextEdit]:
    """Edits to rename a BlendShapeChannel SubDeformer (a morph) in place.

    Rewrites the channel's quoted `"SubDeformer::<old>"` to `<...>::<new_name>`
    (preserving whatever prefix the file used). Morphs are wired into the FBX by
    object id, not by name, so the name is a cosmetic label — renaming it keeps
    the file self-consistent (same as `rename_bone_edits` for bones). The paired
    shape Geometry is intentionally NOT touched: it's an internal node the user
    never sees, and leaving it keeps the edit minimal.

    Returns [] if `channel_id` isn't a channel in the view, so a stale FE id
    can't break the run.
    """
    ch = view.blend_shape_channels.get(channel_id)
    if ch is None:
        return []
    source = view.document.source
    return [_replace_quoted_in_args(
        ch.node_ref, source,
        f'"{ch.full_name}"'.encode("utf-8"),
        f'"{_renamed_full(ch.full_name, "SubDeformer", new_name)}"'.encode("utf-8"),
    )]


def rename_mesh_edits(
    view: SectionView,
    mesh_model_id: int,
    new_name: str,
) -> list[TextEdit]:
    """Edits to rename a mesh in place: the Mesh-type Model's `"Model::<old>"`
    plus every Geometry it owns (`"Geometry::<old>"`), all set to `new_name`.

    Naming the owned Geometry to match the Model keeps the two consistent (the
    Model name is what shows in the outliner; the Geometry name is internal) —
    same spirit as `rename_bone_edits` renaming a bone's clusters to match.

    Returns [] if `mesh_model_id` isn't a Mesh-type Model in the view, so a
    stale id no-ops rather than raising.
    """
    bone = view.bones.get(mesh_model_id)
    if bone is None or bone.type_class != "Mesh":
        return []
    source = view.document.source
    edits: list[TextEdit] = [_replace_quoted_in_args(
        bone.node_ref, source,
        f'"{bone.full_name}"'.encode("utf-8"),
        f'"{_renamed_full(bone.full_name, "Model", new_name)}"'.encode("utf-8"),
    )]
    for gid, owner in view.geometry_owner_model.items():
        if owner != mesh_model_id:
            continue
        g = view.geometries.get(gid)
        if g is None:
            continue
        edits.append(_replace_quoted_in_args(
            g.node_ref, source,
            f'"{g.full_name}"'.encode("utf-8"),
            f'"{_renamed_full(g.full_name, "Geometry", new_name)}"'.encode("utf-8"),
        ))
    return edits


def rename_geometry_edits(
    view: SectionView,
    geometry_id: int,
    new_name: str,
) -> list[TextEdit]:
    """Edits to rename a single Geometry node's `"Geometry::<old>"` in place.

    Use when a mesh's Model name is already readable but its mesh-DATA name
    isn't (common: a Korean/Japanese Blender user exports a mesh whose object
    name is ASCII but whose mesh-data keeps Blender's localized default like
    `평면.050` / `平面.050`). Renames only the Geometry; the owning Model is
    untouched. Returns [] for a stale id.
    """
    g = view.geometries.get(geometry_id)
    if g is None:
        return []
    source = view.document.source
    return [_replace_quoted_in_args(
        g.node_ref, source,
        f'"{g.full_name}"'.encode("utf-8"),
        f'"{_renamed_full(g.full_name, "Geometry", new_name)}"'.encode("utf-8"),
    )]


def _renamed_full(full_name: str, default_prefix: str, new_short: str) -> str:
    """Build a new `Prefix::<new_short>` full name, preserving the original
    node's prefix (e.g. 'SubDeformer', 'Model', 'Geometry') when present."""
    prefix = full_name.rsplit("::", 1)[0] if "::" in full_name else default_prefix
    return f"{prefix}::{new_short}"


def _replace_quoted_in_args(
    node: FBXNode,
    source: bytes,
    needle: bytes,
    replacement: bytes,
) -> TextEdit:
    args_start, args_end = node.args_span
    region = source[args_start:args_end]
    idx = region.find(needle)
    if idx == -1:
        raise EditError(
            f"could not locate {needle!r} in {node.name} args at offset {args_start}"
        )
    return TextEdit(args_start + idx, args_start + idx + len(needle), replacement)
