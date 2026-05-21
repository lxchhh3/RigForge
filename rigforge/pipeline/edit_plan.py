"""EditPlan — the deterministic output of Phase B step 5.

Per PLAN.md:
    drops = `verdict=drop`
    renames = `{donor_name → target_avatar.canonical_to_name[role]}`
             for each kept canonical bone

Phase C consumes the EditPlan to actually rewrite the FBX bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rigforge.ascii_fbx.sections import SectionView
from rigforge.avatars.registry import CuratedAvatar
from rigforge.canonical.decisions import DecisionSet


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

            if d.role not in target_avatar.canonical_to_name:
                # Secondary roles (Breast.L.01, HairSecondary.C, ...) aren't in
                # target's canonical_to_name. We leave their names untouched.
                continue
            target_name = target_avatar.canonical_to_name[d.role]
            bone = view.bones.get(d.model_id)
            if bone is None:
                continue
            if bone.name != target_name:
                renames[d.model_id] = target_name
        return cls(drops=drops, renames=renames, reparents=reparents)
