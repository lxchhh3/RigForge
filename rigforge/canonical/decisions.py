"""LLM decision data model — the validated output of Rail L (Phase B step 3).

Shared between the validators (Phase B step 4) and the LLM module (Phase B
step 3, real and mock implementations). Pydantic for shape-validation since
this is the LLM/Python boundary.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Decision(BaseModel):
    """One per bone — what the LLM judged about it."""
    model_id: int
    role: str
    verdict: str           # "keep" | "drop"
    drop_category: Optional[str] = None
    confidence: float = 1.0
    # v2: optional reparent target. When set, the bone is kept but its parent
    # in the assembled rig should be the kept bone holding canonical role
    # `new_parent_role`. Validators treat this as an edit-time override of the
    # as-input parent; Phase C is expected to emit a `reparent_bone_edits`
    # call against the resolved target id.
    new_parent_role: Optional[str] = None
    # v2.2: English translation / normalized name for the bone. The LLM emits
    # this so non-English (JP/KR/CN) Booth bone names become readable for the
    # team. Phase C renames ride-along (non-canonical) bones to it; absent
    # (older cache/fixtures) → the bone keeps its original name.
    name_en: Optional[str] = None


class DecisionSet(BaseModel):
    """The full LLM output for one run."""
    bones: list[Decision]
    llm_model_id: Optional[str] = None    # e.g. "rigforge-lora-v0.1@ollama:deepseek..."
    canonical_schema_version: Optional[str] = None

    def by_id(self) -> dict[int, Decision]:
        return {d.model_id: d for d in self.bones}

    def kept_by_role(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for d in self.bones:
            if d.verdict == "keep":
                out.setdefault(d.role, []).append(d.model_id)
        return out

    def dropped(self) -> list[Decision]:
        return [d for d in self.bones if d.verdict == "drop"]
