"""Bone-name fingerprint and Jaccard similarity scoring.

Used by Phase A to identify which curated base avatar a clothing FBX was
rigged for. PLAN.md spec:
    normalize names → lowercase, strip separators → set →
    Jaccard against each curated fingerprint →
    highest match >= 0.85 wins, else hard-fail with top-3 candidates.
"""
from __future__ import annotations

import re
from typing import Iterable


_SEPARATORS_RE = re.compile(r"[\s_\-\.]+")


def normalize_bone_name(name: str) -> str:
    """Lowercase, strip separator chars (`_`, `-`, `.`, whitespace), drop common
    DCC-side prefixes (`mixamorig:`, `Armature|`).

    Goal: same logical bone across booth conventions normalizes to the same
    token, so Jaccard intersection is meaningful.
    """
    s = name
    # Strip common rig-namespace prefixes
    for prefix in ("mixamorig:", "Armature|", "Armature:", "Skel:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):]
    s = s.lower()
    s = _SEPARATORS_RE.sub("", s)
    return s


def fingerprint(names: Iterable[str]) -> set[str]:
    """Build a fingerprint set from a list of bone names."""
    return {normalize_bone_name(n) for n in names if n}


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two sets. 0.0 for empty union."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def rank_candidates(
    clothing_fp: set[str],
    candidates: dict[str, set[str]],
) -> list[tuple[str, float]]:
    """Score each candidate by Jaccard, returned sorted high → low.

    Kept for legacy symmetric-set comparison (avatar↔avatar). For
    clothing→avatar donor identification use `rank_clothing_candidates`.
    """
    scored = [(cid, jaccard(clothing_fp, fp)) for cid, fp in candidates.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def rank_clothing_candidates(
    clothing_fp: set[str],
    candidates: dict[str, set[str]],
) -> list[tuple[str, float]]:
    """Score each curated avatar by how well it accounts for the clothing's
    known-avatar bones, with an avatar-uniqueness tiebreaker.

    Jaccard penalizes asymmetric sizes: a clothing FBX with many accessory
    bones (ribbons, chains, hair-clip dynamics) scores low against ANY
    curated avatar because |clothing ∪ avatar| balloons. This scorer
    measures the right thing for clothing→avatar:

        shared_pool      = clothing_fp ∩ union(all_avatar_fps)
        canonical_in_av  = clothing_fp ∩ avatar_fp
        canonical_frac   = |canonical_in_av| / max(1, |shared_pool|)

    Of the clothing's bones that any curated avatar knows about, what
    fraction belongs to THIS avatar? 1.0 means the clothing's recognised
    bones are a strict subset of this avatar.

    When two avatars share most canonical bones (e.g. Hips/Spine across
    avatars), canonical_frac ties. The tiebreaker is avatar-unique bones
    found in clothing (`Hair_L.005` in Moe but not Maya, etc.).
    """
    all_avatar_bones: set[str] = set()
    for fp in candidates.values():
        all_avatar_bones |= fp
    shared_pool = clothing_fp & all_avatar_bones

    scored: list[tuple[str, float, int]] = []
    for cid, av_fp in candidates.items():
        canonical_in_av = clothing_fp & av_fp
        canonical_frac = (
            len(canonical_in_av) / len(shared_pool) if shared_pool else 0.0
        )
        other_bones: set[str] = set()
        for other_cid, other_fp in candidates.items():
            if other_cid != cid:
                other_bones |= other_fp
        avatar_unique = av_fp - other_bones
        unique_in_clothing = clothing_fp & avatar_unique
        scored.append((cid, canonical_frac, len(unique_in_clothing)))

    scored.sort(key=lambda x: (-x[1], -x[2]))
    return [(cid, score) for cid, score, _ in scored]
