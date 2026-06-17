"""EditPlan — the deterministic output of Phase B step 5.

Per PLAN.md:
    drops   = `verdict=drop`
    renames =
        - canonical bones  → `target_avatar.canonical_to_name[role]`
          (moot under the always-merge Phase C — those bones are stripped and
          the target supplies the name — but kept as the B→C contract);
        - non-canonical ride-along bones → their English translation
          (`Decision.name_en`) so the multilingual team can read JP/KR/CN
          Booth bone names. Absent name_en → the bone keeps its original name.

Phase C consumes the EditPlan to actually rewrite the FBX bytes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from rigforge.ascii_fbx.sections import SectionView
from rigforge.avatars.registry import CuratedAvatar
from rigforge.canonical.decisions import DecisionSet


_WHITESPACE_RE = re.compile(r"\s+")
_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_.\-]+")


def _sanitize_bone_name(name: Optional[str]) -> Optional[str]:
    """Coerce an LLM-proposed English name into a Blender/FBX-safe token:
    trim, collapse whitespace to '_', drop chars outside [A-Za-z0-9_.-], and
    strip leading/trailing separators.

    Returns None when nothing usable survives — e.g. the model left the name
    untranslated (still CJK) so every char is stripped — so the caller keeps
    the bone's original name instead of emitting an empty/garbage rename.
    """
    if not name:
        return None
    s = _WHITESPACE_RE.sub("_", name.strip())
    s = _UNSAFE_CHARS_RE.sub("", s)
    s = s.strip("_.-")
    if not s or not re.search(r"[A-Za-z0-9]", s):
        return None
    return s


@dataclass
class EditPlan:
    drops: list[int] = field(default_factory=list)         # bone model_ids to drop
    renames: dict[int, str] = field(default_factory=dict)  # bone model_id -> new short name
    reparents: dict[int, int] = field(default_factory=dict)  # bone model_id -> new parent bone_id

    def is_noop(self) -> bool:
        return not self.drops and not self.renames and not self.reparents

    @classmethod
    def from_decisions(
        cls,
        decisions: DecisionSet,
        view: SectionView,
        target_avatar: CuratedAvatar,
    ) -> "EditPlan":
        drops: list[int] = []
        renames: dict[int, str] = {}
        reparents: dict[int, int] = {}

        # Role -> first kept bone id. Validators (unique_role) ensure each
        # canonical role appears at most once among kept bones, so the lookup
        # is unambiguous by the time we reach here.
        role_to_bone_id: dict[str, int] = {}
        for d in decisions.bones:
            if d.verdict == "keep":
                role_to_bone_id.setdefault(d.role, d.model_id)

        for d in decisions.bones:
            if d.verdict == "drop":
                drops.append(d.model_id)
                continue
            if d.verdict != "keep":
                continue

            if d.new_parent_role is not None:
                target_bone_id = role_to_bone_id.get(d.new_parent_role)
                if target_bone_id is not None and target_bone_id != d.model_id:
                    reparents[d.model_id] = target_bone_id

            bone = view.bones.get(d.model_id)
            if bone is None:
                continue

            if d.role in target_avatar.canonical_to_name:
                # Canonical humanoid bone. Under the merge-based Phase C these
                # bones get stripped and the target supplies the name, so this
                # rename is moot for the assembled output (see
                # test_phase_c_passthrough_renames_are_irrelevant_*). Kept
                # because it's the B→C contract, drives manifest n_renames, and
                # lands in the rare path where a canonical bone rides along
                # unmatched.
                target_name = target_avatar.canonical_to_name[d.role]
                if bone.name != target_name:
                    renames[d.model_id] = target_name
            else:
                # Non-canonical bone — rides along into the output keeping the
                # clothing's own name. Replace it with the LLM's English
                # translation so the multilingual team can read it (Booth/KR/CN
                # rigs ship JP/KR/CN bone names). Applies to every non-canonical
                # kept bone — structured secondary roles AND the unknown
                # `Secondary.<name>` fallback — because readability is the goal,
                # not the role bucket. Sanitized to a Blender-safe token; when
                # name_en is absent or unusable (left untranslated) we keep the
                # original name — backward compatible with cache/fixtures that
                # predate the field.
                en = _sanitize_bone_name(d.name_en)
                if en is not None and en != bone.name:
                    renames[d.model_id] = en
        return cls(drops=drops, renames=renames, reparents=reparents)
