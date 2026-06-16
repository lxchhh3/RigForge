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
    bone_names = [b.name for b in view.limb_bones() if b.name]

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


__all__ = ["run_phase_a", "PhaseAResult", "DonorIdentificationError"]
