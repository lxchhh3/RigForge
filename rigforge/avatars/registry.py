"""Curated avatar registry.

PLAN.md: 4 curated avatars TBD; v1 seed is Maya.fbx (already in canonical
naming). Each avatar carries:
    - fingerprint (bone-name set) — used by Phase A
    - canonical_to_name mapping — used by Phase B (rename target)
    - fbx_path — used by Phase C (loaded ASCII for armature merge)
    - ascii_fbx_path (optional) — preconverted ASCII; if absent, Phase C
      runs bin→ASCII via fbx_env subprocess once and caches the result

Storage: data/curated_avatars.json (committed). Heavy artifacts (cached ASCII
FBX bytes) live in cache/, not in the registry JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

from pydantic import BaseModel, Field

from .fingerprint import (
    fingerprint,
    jaccard,
    normalize_bone_name,
    rank_candidates,
    rank_clothing_candidates,
)

if TYPE_CHECKING:
    from rigforge.ascii_fbx.sections import SectionView


_DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "curated_avatars.json"
)


class CuratedAvatarSpec(BaseModel):
    """JSON schema for one entry in curated_avatars.json."""
    id: str
    display_name: str
    fbx_path: str               # absolute or repo-relative (binary FBX)
    ascii_fbx_path: Optional[str] = None  # preconverted ASCII, optional
    bone_names: list[str]       # full bone-name list (not just fingerprint —
                                # we recompute fingerprint at load so the
                                # normalization rule is the single source of truth)
    canonical_to_name: dict[str, str] = Field(default_factory=dict)


@dataclass
class CuratedAvatar:
    """In-memory view with derived fields.

    Heavy data (parsed SectionView) is lazy via load_ascii_view().
    """
    id: str
    display_name: str
    fbx_path: Path
    bone_names: list[str]
    fingerprint: set[str]
    canonical_to_name: dict[str, str]
    ascii_fbx_path: Optional[Path] = None
    _cached_view: Optional["SectionView"] = field(default=None, repr=False, compare=False)
    _cached_ascii: Optional[bytes] = field(default=None, repr=False, compare=False)

    def load_ascii_bytes(self) -> bytes:
        """Return the avatar's ASCII FBX bytes. Cached after first call.

        Uses ascii_fbx_path if provided; otherwise converts fbx_path via the
        fbx_env subprocess and caches the converted file alongside fbx_path.
        """
        if self._cached_ascii is not None:
            return self._cached_ascii
        if self.ascii_fbx_path is not None and self.ascii_fbx_path.exists():
            self._cached_ascii = self.ascii_fbx_path.read_bytes()
            return self._cached_ascii
        # Fallback: convert on demand. Imported locally to avoid pulling
        # subprocess machinery into registry-load when the path is present.
        from rigforge.ascii_fbx.convert import bin_to_ascii
        derived = self.fbx_path.with_name(self.fbx_path.stem + "_ascii.fbx")
        if not derived.exists():
            bin_to_ascii(self.fbx_path, derived)
        self.ascii_fbx_path = derived
        self._cached_ascii = derived.read_bytes()
        return self._cached_ascii

    def load_ascii_view(self) -> "SectionView":
        """Return a parsed SectionView of this avatar. Cached after first call."""
        if self._cached_view is not None:
            return self._cached_view
        from rigforge.ascii_fbx.lexer import parse as parse_fbx
        from rigforge.ascii_fbx.sections import extract
        raw = self.load_ascii_bytes()
        self._cached_view = extract(parse_fbx(raw))
        return self._cached_view

    def target_bone_id(self, canonical_role: str) -> int:
        """Look up the model_id of this avatar's bone for a canonical role.

        Raises KeyError if the role isn't in canonical_to_name, or if the
        named bone isn't found in the ASCII view.
        """
        if canonical_role not in self.canonical_to_name:
            raise KeyError(
                f"avatar {self.id!r} has no canonical mapping for {canonical_role!r}"
            )
        target_name = self.canonical_to_name[canonical_role]
        view = self.load_ascii_view()
        for b in view.bones.values():
            if b.name == target_name:
                return b.model_id
        raise KeyError(
            f"avatar {self.id!r}: canonical_to_name[{canonical_role!r}]={target_name!r} "
            f"but no bone with that name in the ASCII view"
        )


class RegistryError(RuntimeError):
    """Raised on registry load problems."""


@dataclass
class AvatarRegistry:
    avatars: dict[str, CuratedAvatar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for cid, av in self.avatars.items():
            if cid != av.id:
                raise RegistryError(f"avatar key {cid} != avatar.id {av.id}")

    def get(self, avatar_id: str) -> CuratedAvatar:
        if avatar_id not in self.avatars:
            raise RegistryError(
                f"unknown avatar id '{avatar_id}'. Known: {sorted(self.avatars)}"
            )
        return self.avatars[avatar_id]

    def ids(self) -> list[str]:
        return sorted(self.avatars)

    def fingerprints(self) -> dict[str, set[str]]:
        return {av.id: av.fingerprint for av in self.avatars.values()}

    # --- I/O ---------------------------------------------------------------

    @classmethod
    def load_default(cls) -> "AvatarRegistry":
        return cls.load(_DEFAULT_REGISTRY_PATH)

    @classmethod
    def load(cls, path: Path) -> "AvatarRegistry":
        path = Path(path)
        if not path.exists():
            raise RegistryError(f"registry file missing: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = raw.get("avatars", [])
        avatars: dict[str, CuratedAvatar] = {}
        for entry_dict in entries:
            spec = CuratedAvatarSpec.model_validate(entry_dict)
            fbx_path = Path(spec.fbx_path)
            if not fbx_path.is_absolute():
                fbx_path = (path.parent.parent / spec.fbx_path).resolve()
            ascii_path: Optional[Path] = None
            if spec.ascii_fbx_path:
                ascii_path = Path(spec.ascii_fbx_path)
                if not ascii_path.is_absolute():
                    ascii_path = (path.parent.parent / spec.ascii_fbx_path).resolve()
            av = CuratedAvatar(
                id=spec.id,
                display_name=spec.display_name,
                fbx_path=fbx_path,
                bone_names=list(spec.bone_names),
                fingerprint=fingerprint(spec.bone_names),
                canonical_to_name=dict(spec.canonical_to_name),
                ascii_fbx_path=ascii_path,
            )
            avatars[av.id] = av
        return cls(avatars=avatars)


# ---------------------------------------------------------------------------
# Phase A: donor identification
# ---------------------------------------------------------------------------


DEFAULT_DONOR_THRESHOLD = 0.85


class DonorIdentificationError(RuntimeError):
    """Raised when Phase A cannot pick a donor with high enough confidence.

    Carries the top-3 candidates so the caller can surface them to the user.
    """
    def __init__(self, message: str, top_candidates: list[tuple[str, float]]) -> None:
        super().__init__(message)
        self.top_candidates = top_candidates


def identify_donor(
    clothing_bone_names: Iterable[str],
    registry: AvatarRegistry,
    *,
    target_id: Optional[str] = None,
    threshold: float = DEFAULT_DONOR_THRESHOLD,
) -> tuple[str, float, list[tuple[str, float]]]:
    """Phase A: pick the curated donor that best matches `clothing_bone_names`.

    Returns (donor_id, score, ranked_candidates). Raises DonorIdentificationError
    if the top score < threshold.

    `target_id` (the avatar the clothing is being assembled ONTO) gets first
    refusal: when the target is itself a viable match (score ≥ threshold) we use
    it as the donor so Phase C takes the reliable NAME-BASED merge. The coverage
    scorer favors whichever curated avatar has the larger fingerprint, so a
    target-adapted clothing can lose by a single coincidental bone (e.g. a skirt
    dynamics bone the bigger avatar happens to carry) and get dragged onto the
    cross-avatar (LLM + decision-cache) path it doesn't need. We only fall
    through to the best cross-avatar candidate when the target genuinely doesn't
    fit.
    """
    clothing_fp = fingerprint(clothing_bone_names)
    # Clothing→avatar uses containment + uniqueness; Jaccard would
    # mis-rank when clothing carries many accessory bones the avatar lacks.
    ranked = rank_clothing_candidates(clothing_fp, registry.fingerprints())
    if not ranked:
        raise DonorIdentificationError(
            "no curated avatars in registry", top_candidates=[]
        )
    if target_id is not None:
        target_score = next((s for cid, s in ranked if cid == target_id), None)
        if target_score is not None and target_score >= threshold:
            return (target_id, target_score, ranked)
    top_id, top_score = ranked[0]
    if top_score < threshold:
        raise DonorIdentificationError(
            f"top donor candidate '{top_id}' score {top_score:.3f} < threshold {threshold}",
            top_candidates=ranked[:3],
        )
    return (top_id, top_score, ranked)
