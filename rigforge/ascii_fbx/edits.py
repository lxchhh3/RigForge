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
from typing import Iterable

from .lexer import FBXDocument, FBXNode
from .sections import BoneRecord, ClusterRecord, SectionView, parse_args, _num


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


def drop_mesh_edits(view: SectionView, mesh_model_id: int) -> list[TextEdit]:
    """Edits required to remove a whole mesh chain from the document:
       - the Mesh-type Model node
       - every Geometry connected to that Model
       - every Skin Deformer connected to that Geometry
       - every Cluster SubDeformer connected to that Skin
       - every Connection referencing any of the above

    Returns [] if `mesh_model_id` isn't a Mesh-type Model in the view — this
    keeps the API call site safe against stale ids without raising.

    Bones are intentionally untouched: a bone often deforms multiple meshes
    (Maya rigs every bone into every skin), so cascading bone removal here
    would corrupt sibling meshes. Bone cleanup is a separate concern.
    """
    bone = view.bones.get(mesh_model_id)
    if bone is None or bone.type_class != "Mesh":
        return []
    source = view.document.source
    edits: list[TextEdit] = [drop_node_edit(bone.node_ref, source)]

    # Find Geometries owned by this Mesh-Model (reverse of geometry_owner_model).
    geom_ids: set[int] = {
        gid for gid, mid in view.geometry_owner_model.items()
        if mid == mesh_model_id
    }
    # Find Skins deforming those geometries.
    skin_ids: set[int] = {
        s.skin_id for s in view.skins.values()
        if s.geometry_id is not None and s.geometry_id in geom_ids
    }
    # Find Clusters belonging to those skins (reverse of cluster_skin).
    cluster_ids: set[int] = {
        cid for cid, sid in view.cluster_skin.items()
        if sid in skin_ids
    }

    for gid in geom_ids:
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

    edits.extend(_drop_connections_referencing(
        view, ids={mesh_model_id, *geom_ids, *skin_ids, *cluster_ids},
    ))
    return edits


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
) -> list[TextEdit]:
    """Edits to change the bone's display name from `bone.name` to
    `new_short_name`, touching:
       - the Model's quoted name `"Model::<old>"`
       - every attached Cluster's `"SubDeformer::<old>"` (if it matches)
    """
    source = view.document.source
    edits: list[TextEdit] = []

    edits.append(_replace_quoted_in_args(
        bone.node_ref, source,
        f'"{bone.full_name}"'.encode("utf-8"),
        f'"Model::{new_short_name}"'.encode("utf-8"),
    ))

    for cid in bone.cluster_ids:
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
