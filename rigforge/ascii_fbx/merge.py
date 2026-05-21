"""Cross-avatar armature merge primitives (Phase C step bodies).

Composes on top of edits.py's TextEdit + apply_edits. Phase C runs these in
sequence when donor_id != target_id:

  1. cluster_repoint_edits(clothing_view, table)
        Rewrite the bone-side arg of bone↔cluster connections so clusters
        reference the target armature's bone ids (not the clothing's).

  2. bone_keep_strip_edits(clothing_view, kept_bone) for every kept bone
        Drop the kept bone's Model node + its parent-child connection,
        but leave cluster bodies intact (their skin weights live on).

  3. drop_bone_edits (existing, in edits.py) for every dropped bone
        Same as Phase B's pass-through case: bone + clusters + connections.

  4. id_offset_edits(clothing_view, offset, repoint)
        Shift every clothing object id (and every connection arg) by
        `offset`, EXCEPT where `repoint` maps a clothing id to a target id
        (those args get the target id directly, no offset).

  5. splice_into_section_edit(target_doc, "Objects", clothing_objects_bytes)
        Append surviving clothing nodes into target's Objects { ... }.
        Same for Connections.

The lexer offsets stay valid because each step builds its TextEdits against
the same ORIGINAL clothing source — they're collected then applied once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .edits import TextEdit, drop_node_edit
from .lexer import FBXDocument, FBXNode
from .sections import BoneRecord, SectionView


# ---------------------------------------------------------------------------
# Positioned arg tokenizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArgToken:
    """A single arg token with byte-span relative to the args-region start."""
    kind: str           # "str" | "num"
    value: Any          # str for "str", int|float for "num"
    start: int          # offset within the args region
    end: int            # exclusive


def positioned_args(text: bytes) -> list[ArgToken]:
    """Tokenize an arg region, returning each token with byte span.

    Like sections.parse_args but retains positions. Same handling of
    comments + strings + bare tokens.
    """
    out: list[ArgToken] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in b" \t\r\n":
            i += 1
            continue
        if c == 0x3B:  # ';' comment to EOL
            while i < n and text[i] != 0x0A:
                i += 1
            continue
        if c == 0x22:  # '"' string
            start = i
            i += 1
            j = i
            while j < n and text[j] != 0x22:
                j += 1
            value = text[i:j].decode("utf-8", errors="replace")
            end = j + 1 if j < n else j
            out.append(ArgToken("str", value, start, end))
            i = end
            continue
        if c == 0x2C:  # ',' separator
            i += 1
            continue
        # bare token: digit, sign, dot, '*', identifier-like
        start = i
        j = i
        while j < n and text[j] not in b",; \t\r\n":
            j += 1
        tok_bytes = text[start:j]
        s = tok_bytes.decode("utf-8", errors="replace")
        i = j
        if not s:
            continue
        try:
            if "." in s or "e" in s or "E" in s:
                out.append(ArgToken("num", float(s), start, j))
            else:
                out.append(ArgToken("num", int(s), start, j))
        except ValueError:
            # opaque token like "*4782" or "T" — keep as str
            out.append(ArgToken("str", s, start, j))
    return out


# ---------------------------------------------------------------------------
# ID offset
# ---------------------------------------------------------------------------


_OBJECTS_SECTION = "Objects"
_CONNECTIONS_SECTION = "Connections"


def _first_numeric_token_span(
    node: FBXNode, source: bytes
) -> tuple[int, int, int] | None:
    """Locate the first numeric arg of `node` in source bytes.

    Returns (absolute_start, absolute_end, current_value) or None if no
    numeric arg exists.
    """
    args_start, args_end = node.args_span
    region = source[args_start:args_end]
    for tok in positioned_args(region):
        if tok.kind == "num":
            return (args_start + tok.start, args_start + tok.end, int(tok.value))
    return None


def id_offset_edits(
    view: SectionView,
    *,
    offset: int,
    repoint: dict[int, int],
) -> list[TextEdit]:
    """Shift every clothing-side object id by `offset`, with `repoint`
    overriding specific ids.

    Touches:
      (a) every object's header in Objects { ... } — rewrites the first
          numeric arg
      (b) every C: leaf in Connections { ... } — rewrites src + dst args

    `repoint` is consulted FIRST for every id it knows about; misses fall
    through to `id + offset`. Used by cross-merge to point repurposed
    clusters at the target avatar's actual bone ids.
    """
    edits: list[TextEdit] = []
    source = view.document.source

    objects = view.document.root(_OBJECTS_SECTION)
    if objects is not None:
        for obj in objects.children:
            spec = _first_numeric_token_span(obj, source)
            if spec is None:
                continue
            abs_start, abs_end, val = spec
            new_val = repoint.get(val, val + offset)
            edits.append(TextEdit(abs_start, abs_end, str(new_val).encode("ascii")))

    connections = view.document.root(_CONNECTIONS_SECTION)
    if connections is not None:
        for conn in connections.children:
            if conn.name != "C":
                continue
            args_start, args_end = conn.args_span
            region = source[args_start:args_end]
            toks = positioned_args(region)
            # C args: ["OO"|"OP", src_id, dst_id, optional "PropName"]
            # Rewrite every numeric token in the args. RootNode (id=0) is
            # special — it's the scene root, shared with target after splice,
            # and must never be offset. Treat 0 as identity unless the caller
            # has explicitly remapped it via `repoint`.
            for t in toks:
                if t.kind != "num":
                    continue
                val = int(t.value)
                if val == 0 and 0 not in repoint:
                    continue
                new_val = repoint.get(val, val + offset)
                if new_val == val:
                    continue
                edits.append(TextEdit(
                    args_start + t.start,
                    args_start + t.end,
                    str(new_val).encode("ascii"),
                ))

    return edits


# ---------------------------------------------------------------------------
# Cluster repoint (bone-side of bone↔cluster connections only)
# ---------------------------------------------------------------------------


def cluster_repoint_edits(
    view: SectionView,
    repoint_table: dict[int, int],
) -> list[TextEdit]:
    """For each connection where one endpoint is a kept-bone (in
    repoint_table) and the other is a Cluster (in view.clusters), rewrite
    the bone-side arg to the target avatar's bone id.

    Leaves parent-child (bone↔bone) connections untouched — those are
    handled separately when the bone Model is stripped.
    """
    edits: list[TextEdit] = []
    source = view.document.source
    connections = view.document.root(_CONNECTIONS_SECTION)
    if connections is None:
        return edits

    cluster_ids = set(view.clusters.keys())
    for conn in connections.children:
        if conn.name != "C":
            continue
        args_start, args_end = conn.args_span
        region = source[args_start:args_end]
        toks = positioned_args(region)
        if len(toks) < 3:
            continue
        ctype = toks[0].value if toks[0].kind == "str" else None
        if ctype != "OO":
            continue
        if toks[1].kind != "num" or toks[2].kind != "num":
            continue
        src = int(toks[1].value)
        dst = int(toks[2].value)
        # Identify bone-cluster pairing
        if src in repoint_table and dst in cluster_ids:
            edits.append(TextEdit(
                args_start + toks[1].start,
                args_start + toks[1].end,
                str(repoint_table[src]).encode("ascii"),
            ))
        elif dst in repoint_table and src in cluster_ids:
            edits.append(TextEdit(
                args_start + toks[2].start,
                args_start + toks[2].end,
                str(repoint_table[dst]).encode("ascii"),
            ))

    return edits


# ---------------------------------------------------------------------------
# Bone strip (kept bone in cross-merge)
# ---------------------------------------------------------------------------


def bone_keep_strip_edits(view: SectionView, bone: BoneRecord) -> list[TextEdit]:
    """Strip a KEPT clothing bone whose role is owned by the target:
      - drop the bone's Model node
      - drop the connection that names THIS bone as a child (src=bone_id, dst=parent),
        i.e. the bone's own upward parent reference — that link disappears
        with the Model.

    Connections that reference this bone as a PARENT (src=child, dst=bone_id)
    are LEFT IN PLACE so the id_offset_edits pass can repoint them via the
    `repoint` table to the target avatar's matching bone id. That way the
    bone's child sub-tree (secondary chains, clusters) stays attached to the
    correct point in the target armature.
    """
    edits: list[TextEdit] = [drop_node_edit(bone.node_ref, view.document.source)]

    source = view.document.source
    connections = view.document.root(_CONNECTIONS_SECTION)
    if connections is None:
        return edits

    bone_id = bone.model_id
    cluster_ids = set(view.clusters.keys())
    for conn in connections.children:
        if conn.name != "C":
            continue
        toks = positioned_args(conn.args_bytes(source))
        if len(toks) < 3 or toks[0].kind != "str" or toks[0].value != "OO":
            continue
        if toks[1].kind != "num" or toks[2].kind != "num":
            continue
        src = int(toks[1].value)
        dst = int(toks[2].value)
        # Drop only the bone's own upward parent reference: src=this bone,
        # dst is another Model or the scene RootNode (0). Skin-attach
        # connections (dst is a Cluster) stay so they can be repointed.
        # Connections where THIS bone is the dst (children pointing at it
        # as parent) also stay — id_offset's repoint redirects them to the
        # target bone id, so the child sub-tree (secondary chains,
        # ornaments) lands at the right place in the target armature.
        if src == bone_id and dst not in cluster_ids:
            edits.append(drop_node_edit(conn, source))

    return edits


# ---------------------------------------------------------------------------
# Zero-weight bone sweep (post-merge cleanup)
# ---------------------------------------------------------------------------


def zero_weight_leaf_bone_ids(view: SectionView) -> set[int]:
    """Identify LimbNode bones that contribute no skin weight to any vertex
    AND have no surviving non-zero-weight descendants. Iterative leaf prune:
    a bone enters the drop set when every cluster attached to it has
    `weight_count == 0` (or it has no clusters at all) AND every one of its
    children is already in the drop set. The pass repeats until stable, so a
    chain of dead end-markers collapses entirely.

    Why leaf-only: dropping a mid-chain weightless bone would orphan its
    weighted descendants (parent connection goes away, children float). Leaf
    pruning is the safe form of the collaborator's "aggressively wipe every
    0-weight bone" guidance — almost every zero-weight bone in a real rig is
    a chain tip (`_end` markers, IK helpers, unused secondary chains), and
    iterating catches the chain wholesale even when intermediate nodes have
    no weight themselves.

    Does NOT touch Mesh / Null type Models, only LimbNodes.
    """
    drop: set[int] = set()
    while True:
        progress = False
        for bid, bone in view.bones.items():
            if bid in drop:
                continue
            if bone.type_class != "LimbNode":
                continue
            kept_children = [c for c in bone.children_ids if c not in drop]
            if kept_children:
                continue
            has_weight = False
            for cid in bone.cluster_ids:
                cluster = view.clusters.get(cid)
                if cluster is not None and cluster.weight_count > 0:
                    has_weight = True
                    break
            if has_weight:
                continue
            drop.add(bid)
            progress = True
        if not progress:
            return drop


# ---------------------------------------------------------------------------
# Splice
# ---------------------------------------------------------------------------


def splice_into_section_edit(
    doc: FBXDocument,
    section_name: str,
    payload: bytes,
) -> TextEdit:
    """Build an edit that inserts `payload` just before the closing '}' of the
    named top-level section (e.g. "Objects", "Connections"). The payload is
    inserted verbatim — caller is responsible for indentation + trailing
    newline.
    """
    section = doc.root(section_name)
    if section is None:
        raise KeyError(f"document has no section '{section_name}'")
    if section.body_close is None:
        raise ValueError(f"section '{section_name}' has no body")
    insert_at = section.body_close
    return TextEdit(insert_at, insert_at, payload)
