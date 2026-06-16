"""EditPlan — the deterministic output of Phase B step 5.

Per PLAN.md:
    drops = `verdict=drop`
    renames = `{donor_name → target_avatar.canonical_to_name[role]}`
             for each kept canonical bone

Phase C consumes the EditPlan to actually rewrite the FBX bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rigforge.ascii_fbx.sections import SectionView
from rigforge.avatars.registry import CuratedAvatar
from rigforge.canonical.decisions import DecisionSet
from rigforge.canonical.schema import CanonicalSchema


def _secondary_role_to_name(role: str) -> str:
    """Clean, deterministic bone name for a structured secondary role.

    The role string is already the canonical, manageable label
    (e.g. 'SkirtSide.L.16', 'HairSecondary.C.05'), so we use it verbatim.
    Keeping the output name identical to the role makes the naming scheme
    self-documenting and stable across runs. Centralized here so a future
    convention change (underscores, prefixes) is a one-line edit.
    """
    return role


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
        schema: Optional[CanonicalSchema] = None,
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
                # Canonical humanoid bone. Note: in the merge-based Phase C
                # these bones get stripped and the target supplies the name,
                # so this rename is moot for the assembled output — see
                # test_phase_c_passthrough_renames_are_irrelevant_*. We still
                # compute it: it's the B→C contract, drives the manifest's
                # n_renames, and the rename DOES land in the rare path where a
                # canonical bone rides along unmatched.
                target_name = target_avatar.canonical_to_name[d.role]
                if bone.name != target_name:
                    renames[d.model_id] = target_name
            elif schema is not None and schema.is_secondary(d.role):
                # Structured secondary role (SkirtSide.L.16, HairSecondary.C.05,
                # Accessory.C.01, ...). These bones are NOT in the target's
                # canonical_to_name, so Phase C lets them ride along into the
                # output unstripped — which means this rename is NOT moot: it's
                # where the modder finally gets clean, manageable accessory
                # names. The `Secondary.<bone_name>` fallback role is
                # deliberately not a schema secondary pattern, so it's excluded
                # here and keeps its original name.
                clean = _secondary_role_to_name(d.role)
                if bone.name != clean:
                    renames[d.model_id] = clean
            # else: unrecognized/fallback role — leave the name untouched.
        return cls(drops=drops, renames=renames, reparents=reparents)
