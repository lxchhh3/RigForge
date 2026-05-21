"""Materials + BlendShapes section-level merge.

Produces a `MergePlan` of byte-level inserts against the TARGET document. Each
`MergeEdit` mirrors the shape of `rigforge.ascii_fbx.edits.TextEdit` so the
integration pass can convert by structural cast — this module deliberately
does not import `edits.py`.

Dedup keys (target wins on collision):
  - Material: short name (e.g. "Cloth")
  - BlendShapeChannel: short name (e.g. "Smile")

When a donor BlendShapeChannel survives dedup, its owning BlendShape Deformer
is also brought across — unless the target already has a BlendShape Deformer
of the same name (in which case the channel is re-rooted onto it implicitly;
re-pointing connections is a separate pass owned by the pipeline integration).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from rigforge.ascii_fbx.sections import SectionView


_OBJECTS_SECTION = "Objects"


@dataclass(frozen=True)
class MergeEdit:
    """A byte splice on the target source. Shape mirrors TextEdit so the
    pipeline integration can convert by ``TextEdit(start, end, replacement)``.
    """
    start: int
    end: int             # exclusive
    replacement: bytes


@dataclass
class MergePlan:
    """A bundle of MergeEdits with provenance for the integration pass + manifest.

    `edits` are byte ops on the TARGET source. `inserted_names` lists the
    donor-side names that survived dedup (for manifest reporting / debugging).
    `skipped_names` lists names dropped because target already owns them.
    """
    edits: list[MergeEdit] = field(default_factory=list)
    inserted_names: list[str] = field(default_factory=list)
    skipped_names: list[str] = field(default_factory=list)


# --- helpers -----------------------------------------------------------------


def _objects_close_offset(view: SectionView) -> int:
    """Byte offset of the `}` that closes the target's Objects section.

    The splice insertion point for all merges is right before this brace.
    """
    section = view.document.root(_OBJECTS_SECTION)
    if section is None:
        raise KeyError("target document has no Objects section")
    if section.body_close is None:
        raise ValueError("target Objects section has no body")
    return section.body_close


def _node_bytes(view: SectionView, node) -> bytes:
    """Slice a donor node's source bytes including the node's own indentation.

    The lexer's `name_span` starts at the identifier; we walk back to the
    previous newline to capture leading tabs/spaces so the spliced node lines
    up with the target's existing indentation style.
    """
    source = view.document.source
    name_start = node.name_span[0]
    i = name_start
    while i > 0 and source[i - 1] in b" \t":
        i -= 1
    return source[i:node.extent_end]


def _payload_with_newline(chunks: Iterable[bytes]) -> bytes:
    """Concatenate node-chunks, ensuring each ends with a newline so the splice
    leaves the target's closing brace on its own line."""
    out: list[bytes] = []
    for c in chunks:
        if not c:
            continue
        out.append(c)
        if not c.endswith(b"\n"):
            out.append(b"\n")
    return b"".join(out)


# --- material merge ----------------------------------------------------------


def merge_materials(*, donor: SectionView, target: SectionView) -> MergePlan:
    """Dedup donor materials against target by name; splice survivors into
    target's Objects section. Target wins on name collision.
    """
    target_names = {m.name for m in target.materials.values()}

    payload_chunks: list[bytes] = []
    inserted: list[str] = []
    skipped: list[str] = []

    # Stable order by donor's material_id so plans are reproducible.
    for mid in sorted(donor.materials):
        mat = donor.materials[mid]
        if mat.name in target_names:
            skipped.append(mat.name)
            continue
        payload_chunks.append(_node_bytes(donor, mat.node_ref))
        inserted.append(mat.name)
        # Track within-donor duplicates so we don't double-insert.
        target_names.add(mat.name)

    if not payload_chunks:
        return MergePlan(edits=[], inserted_names=inserted, skipped_names=skipped)

    insert_at = _objects_close_offset(target)
    payload = _payload_with_newline(payload_chunks)
    return MergePlan(
        edits=[MergeEdit(insert_at, insert_at, payload)],
        inserted_names=inserted,
        skipped_names=skipped,
    )


# --- blendshape merge --------------------------------------------------------


def merge_blendshapes(*, donor: SectionView, target: SectionView) -> MergePlan:
    """Dedup donor BlendShape *channels* against target by name; splice
    survivors plus any owning BlendShape Deformer they depend on.

    Owning-BlendShape rule: each surviving channel's owner is included unless
    the target already exposes a BlendShape Deformer of the same name. Channels
    whose owner is absent in the donor (malformed connection) are still spliced
    so their bytes round-trip — the pipeline's connection-fixup pass deals with
    dangling references.
    """
    target_channel_names = {ch.name for ch in target.blend_shape_channels.values()}
    target_blend_shape_names = {bs.name for bs in target.blend_shapes.values()}

    surviving_channels = []
    inserted: list[str] = []
    skipped: list[str] = []
    for cid in sorted(donor.blend_shape_channels):
        ch = donor.blend_shape_channels[cid]
        if ch.name in target_channel_names:
            skipped.append(ch.name)
            continue
        surviving_channels.append(ch)
        inserted.append(ch.name)
        target_channel_names.add(ch.name)

    # Determine which BlendShape Deformers need to come along. Stable by id.
    owners_to_bring: list[int] = []
    seen_owner_ids: set[int] = set()
    for ch in surviving_channels:
        owner_id = ch.blend_shape_id
        if owner_id is None or owner_id in seen_owner_ids:
            continue
        owner = donor.blend_shapes.get(owner_id)
        if owner is None:
            continue
        seen_owner_ids.add(owner_id)
        if owner.name in target_blend_shape_names:
            # Target already has a BlendShape of this name; channel rerouting is
            # the pipeline's job. Don't duplicate the container.
            continue
        owners_to_bring.append(owner_id)
        target_blend_shape_names.add(owner.name)

    payload_chunks: list[bytes] = []
    # Splice BlendShape owners first so the resulting bytes have parent-before-child
    # order — matches Maya's own export style and is a friendlier diff.
    for owner_id in owners_to_bring:
        payload_chunks.append(_node_bytes(donor, donor.blend_shapes[owner_id].node_ref))
    for ch in surviving_channels:
        payload_chunks.append(_node_bytes(donor, ch.node_ref))

    if not payload_chunks:
        return MergePlan(edits=[], inserted_names=inserted, skipped_names=skipped)

    insert_at = _objects_close_offset(target)
    payload = _payload_with_newline(payload_chunks)
    return MergePlan(
        edits=[MergeEdit(insert_at, insert_at, payload)],
        inserted_names=inserted,
        skipped_names=skipped,
    )
