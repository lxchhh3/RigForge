"""Phase B — Cross-booth rig mapping (Rail L lives here).

Flow (PLAN.md):
    1. Section extraction  (Rail D)  — done in Phase A; passed in here
    2. Cache lookup        (Rail D)
    3. LLM call            (Rail L)  — single batched classify()
    4. Validation          (Rail D)  — 7 rules; re-prompt once on error
    5. EditPlan synthesis  (Rail D)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from rigforge.ascii_fbx.sections import SectionView
from rigforge.avatars.registry import AvatarRegistry
from rigforge.cache.key import compute_cache_key, compute_cache_key_inputs
from rigforge.cache.store import DecisionCache
from rigforge.canonical.decisions import Decision, DecisionSet
from rigforge.canonical.schema import CanonicalSchema
from rigforge.canonical.validators import Violation, errors, validate_all, warnings
from rigforge.llm.client import LLMClient

from .edit_plan import EditPlan


class PhaseBError(RuntimeError):
    """Raised on unrecoverable Phase B failures (LLM never produces a valid
    classification, even after one re-prompt)."""

    def __init__(self, message: str, violations: list[Violation]) -> None:
        super().__init__(message)
        self.violations = violations


@dataclass
class PhaseBResult:
    decisions: DecisionSet
    edit_plan: EditPlan
    warnings: list[Violation]
    cache_hit: bool
    cache_key: str


def run_phase_b(
    *,
    view: SectionView,
    donor_id: str,
    target_id: str,
    registry: AvatarRegistry,
    schema: CanonicalSchema,
    llm_client: LLMClient,
    cache: Optional[DecisionCache] = None,
    user_drop_bone_ids: Optional[set[int]] = None,
) -> PhaseBResult:
    target_avatar = registry.get(target_id)

    json_records = view.json_records(type_classes=("LimbNode",))
    request = _build_request(
        donor_id=donor_id,
        target_id=target_id,
        schema_version=schema.version,
        bones=json_records,
    )

    cache_inputs = compute_cache_key_inputs(
        canonical_schema_version=schema.version,
        llm_model_id=llm_client.model_id,
        donor_id=donor_id,
        target_id=target_id,
        bones=view.limb_bones(),
    )
    key = compute_cache_key(cache_inputs)

    decisions: Optional[DecisionSet] = None
    cache_hit = False
    if cache is not None:
        decisions = cache.get(key)
        cache_hit = decisions is not None

    if decisions is None:
        decisions = llm_client.classify(request)
        if cache is not None:
            cache.put(key, decisions)

    # User-drop pre-filter: cascade the user's chosen drops to every transitive
    # descendant, then force their verdict to "drop" regardless of LLM output.
    # Runs BEFORE validation so the post-override decision set is what the
    # validators (and downstream EditPlan) actually see.
    if user_drop_bone_ids:
        _apply_user_drops(decisions, view, user_drop_bone_ids)

    violations = validate_all(decisions, view, schema)
    errs = errors(violations)
    if errs:
        # Re-prompt once with the violations appended to the request (PLAN.md
        # Phase B step 4). For MockLLMClient this is a no-op replay — that's
        # fine; we still get a deterministic hard-fail when the fixture is bad.
        reprompt_request = {**request, "previous_violations": [v.message for v in errs]}
        decisions = llm_client.classify(reprompt_request)
        if cache is not None:
            cache.put(key, decisions)
        violations = validate_all(decisions, view, schema)
        errs = errors(violations)
        if errs:
            raise PhaseBError(
                f"Phase B validation failed after re-prompt: "
                f"{len(errs)} error(s) — first: {errs[0].message}",
                violations=errs,
            )

    plan = EditPlan.from_decisions(decisions, view, target_avatar)
    return PhaseBResult(
        decisions=decisions,
        edit_plan=plan,
        warnings=warnings(violations),
        cache_hit=cache_hit,
        cache_key=key,
    )


def _apply_user_drops(
    decisions: DecisionSet,
    view: SectionView,
    user_drop_bone_ids: set[int],
) -> None:
    """Mutate `decisions` in place: force verdict=drop with
    drop_category='user_dropped' on every bone in `user_drop_bone_ids` and
    every transitive descendant.

    Cascading prevents dangling subtrees in the assembled rig — if the user
    drops a chain root, the bones hanging off it must go too.
    """
    forced: set[int] = set()
    stack: list[int] = list(user_drop_bone_ids)
    while stack:
        bid = stack.pop()
        if bid in forced:
            continue
        forced.add(bid)
        bone = view.bones.get(bid)
        if bone is not None:
            stack.extend(bone.children_ids)

    for i, d in enumerate(decisions.bones):
        if d.model_id not in forced:
            continue
        decisions.bones[i] = Decision(
            model_id=d.model_id,
            role="aux",
            verdict="drop",
            drop_category="user_dropped",
            confidence=1.0,
        )


def _build_request(
    *,
    donor_id: str,
    target_id: str,
    schema_version: str,
    bones: list[dict],
) -> dict[str, Any]:
    return {
        "donor_id": donor_id,
        "target_id": target_id,
        "canonical_schema_version": schema_version,
        "bones": bones,
    }
