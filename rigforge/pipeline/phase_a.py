"""Phase A — Donor identification (deterministic only).

Method (PLAN.md):
    Convert clothing to ASCII → extract bone-name set → normalize → Jaccard
    against curated registry fingerprints → ≥0.85 wins, else hard-fail with
    top-3 candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rigforge.ascii_fbx.convert import bin_to_ascii_cached
from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import SectionView, extract
from rigforge.avatars.fingerprint import strip_rig_suffix
from rigforge.avatars.registry import (
    AvatarRegistry,
    DEFAULT_DONOR_THRESHOLD,
    DonorIdentificationError,
    identify_donor,
)


@dataclass
class PhaseAResult:
    donor_id: str
    score: float
    candidates: list[tuple[str, float]]
    ascii_path: Path                # cached intermediate (caller may reuse)
    view: SectionView
    # A garment with no bones (a rigid prop: mesh + blendshapes only). It has
    # nothing to match a donor on, so the target stands in as its donor and
    # Phase B has nothing to classify; Phase C merges its meshes as-is.
    rigid: bool = False
    # A prop with its OWN armature sharing no bone with the target (kevin's
    # Toast). Same as rigid for Phase A/B; Phase C keeps its armature whole.
    self_rigged: bool = False


def is_self_rigged(view: SectionView, target_view: SectionView) -> bool:
    """A garment with bones, none of which is a bone of the target (by name,
    fitting-tool suffix stripped): a prop with its own small armature, merged
    as-is. The one predicate Phase A, Phase C and the inspect API use. A
    garment sharing ANY bone with the target is a garment for an avatar."""
    own = {strip_rig_suffix(b.name) for b in view.limb_bones() if b.name}
    if not own:
        return False
    theirs = {strip_rig_suffix(b.name) for b in target_view.limb_bones() if b.name}
    return not (own & theirs)


def run_phase_a(
    clothing_fbx: Path,
    registry: AvatarRegistry,
    *,
    work_dir: Path,
    ascii_cache_dir: Optional[Path] = None,
    target_id: Optional[str] = None,
    threshold: float = DEFAULT_DONOR_THRESHOLD,
) -> PhaseAResult:
    """Identify the curated donor for `clothing_fbx`.

    Parameters
    ----------
    clothing_fbx
        Path to the clothing FBX (binary or ASCII — conversion is skipped if
        already ASCII).
    registry
        Curated avatar registry (Phase B and C will consume target_avatar from
        the same registry).
    work_dir
        Directory for converter intermediates. Must exist.
    ascii_cache_dir
        Shared bin→ASCII cache directory. MUST be the same directory the API's
        inspect endpoint uses, so the clothing converts exactly once and the
        FE's mesh/bone/channel ids match this view's ids (see
        `bin_to_ascii_cached`). Falls back to `work_dir` when omitted (the
        CLI path, where no separate inspect runs).
    """
    clothing_fbx = Path(clothing_fbx).resolve()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(ascii_cache_dir).resolve() if ascii_cache_dir else work_dir
    # Returns clothing_fbx unchanged when it's already ASCII.
    ascii_path = bin_to_ascii_cached(clothing_fbx, cache_dir)

    doc = parse(ascii_path.read_bytes())
    view = extract(doc)
    if view.is_rigid():
        if target_id is None:
            raise DonorIdentificationError(
                f"{clothing_fbx.name} has no bones (a rigid mesh): name the target "
                f"avatar to merge it onto",
                [],
            )
        return PhaseAResult(
            donor_id=target_id, score=0.0, candidates=[],
            ascii_path=ascii_path, view=view, rigid=True,
        )
    if target_id is not None and is_self_rigged(view, registry.get(target_id).load_ascii_view()):
        return PhaseAResult(
            donor_id=target_id, score=0.0, candidates=[],
            ascii_path=ascii_path, view=view, self_rigged=True,
        )
    # Match on base names: a MochiFitter export's `Hips_MFBase` is maya's `Hips`.
    bone_names = [strip_rig_suffix(b.name) for b in view.limb_bones() if b.name]

    donor_id, score, ranked = identify_donor(
        bone_names, registry, target_id=target_id, threshold=threshold,
    )
    return PhaseAResult(
        donor_id=donor_id,
        score=score,
        candidates=ranked[:3],
        ascii_path=ascii_path,
        view=view,
    )


__all__ = ["run_phase_a", "PhaseAResult", "DonorIdentificationError", "is_self_rigged"]
