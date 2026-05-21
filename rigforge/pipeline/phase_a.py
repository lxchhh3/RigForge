"""Phase A — Donor identification (deterministic only).

Method (PLAN.md):
    Convert clothing to ASCII → extract bone-name set → normalize → Jaccard
    against curated registry fingerprints → ≥0.85 wins, else hard-fail with
    top-3 candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rigforge.ascii_fbx.convert import bin_to_ascii
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
    threshold: float = DEFAULT_DONOR_THRESHOLD,
) -> PhaseAResult:
    """Identify the curated donor for `clothing_fbx`.

    Parameters
    ----------
    clothing_fbx
        Path to the clothing FBX (binary or ASCII — ascii_to_bin is skipped if
        already ASCII).
    registry
        Curated avatar registry (Phase B and C will consume target_avatar from
        the same registry).
    work_dir
        Directory for converter intermediates. Must exist.
    """
    clothing_fbx = Path(clothing_fbx).resolve()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    head = clothing_fbx.read_bytes()[:32]
    if head.startswith(b"Kaydara FBX Binary"):
        ascii_path = work_dir / (clothing_fbx.stem + "_ascii.fbx")
        bin_to_ascii(clothing_fbx, ascii_path)
    else:
        ascii_path = clothing_fbx

    doc = parse(ascii_path.read_bytes())
    view = extract(doc)
    bone_names = [b.name for b in view.limb_bones() if b.name]

    donor_id, score, ranked = identify_donor(bone_names, registry, threshold=threshold)
    return PhaseAResult(
        donor_id=donor_id,
        score=score,
        candidates=ranked[:3],
        ascii_path=ascii_path,
        view=view,
    )


__all__ = ["run_phase_a", "PhaseAResult", "DonorIdentificationError"]
