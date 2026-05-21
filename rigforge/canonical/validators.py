"""Phase B validators — the safety net between the LLM and the FBX edit stage.

Seven rules from PLAN.md, severity error|warning. Any error blocks edits; the
orchestrator re-prompts once with the violation text and hard-fails on the
second failure.

Each rule is a pure function:
    def rule_X(decisions, view, schema) -> list[Violation]

`validate_all` runs them in order and concatenates results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from rigforge.ascii_fbx.sections import BoneRecord, SectionView
from rigforge.canonical.decisions import Decision, DecisionSet
from rigforge.canonical.schema import CanonicalSchema


# Threshold for drop_safety in v1: count-based (PLAN.md mentions 0.05 weight
# in the v2 weight-value interpretation; for v1 we use cluster_weight_count
# as a cheap proxy because parsing the per-vertex weight array is expensive).
# Note: drop_safety gates on SUBTREE weight, not per-bone weight — dropping a
# zero-weight ancestor (e.g. `Head Phones_Root`, `Chain_1.006`) orphans its
# weighted descendants from the rig hierarchy. Threshold allows a few stray
# binds from export noise.
_DROP_SAFETY_MAX_SUBTREE_WEIGHTS = 10
# Lateral pair X-mirror tolerance (PLAN: ±5%)
_LR_MIRROR_REL_TOL = 0.05
_LR_MIRROR_ABS_FLOOR = 1e-4  # absolute tolerance for near-zero coordinates


@dataclass
class Violation:
    severity: str           # "error" | "warning"
    rule: str
    message: str
    bone_ids: list[int] = field(default_factory=list)


def _bones_by_role(decisions: DecisionSet, role: str) -> list[int]:
    return [d.model_id for d in decisions.bones
            if d.verdict == "keep" and d.role == role]


# ---------------------------------------------------------------------------
# Rule 1: unique_role
# ---------------------------------------------------------------------------


def rule_unique_role(
    decisions: DecisionSet,
    view: SectionView,
    schema: CanonicalSchema,
) -> list[Violation]:
    """Each canonical (non-secondary) role assigned exactly once with verdict=keep.

    Also: "unknown" is an error (the LLM refused to classify).
    """
    out: list[Violation] = []
    seen: dict[str, list[int]] = {}
    for d in decisions.bones:
        if d.role == "unknown":
            out.append(Violation(
                severity="error",
                rule="unique_role",
                message=f"bone {d.model_id} got role='unknown' — LLM must classify or use 'aux'",
                bone_ids=[d.model_id],
            ))
            continue
        if d.verdict != "keep":
            continue
        # Only canonical roles need uniqueness. Secondary roles can repeat.
        if not schema.is_canonical(d.role):
            continue
        seen.setdefault(d.role, []).append(d.model_id)
    for role, ids in seen.items():
        if len(ids) > 1:
            out.append(Violation(
                severity="error",
                rule="unique_role",
                message=f"canonical role '{role}' assigned to {len(ids)} kept bones: {ids}",
                bone_ids=ids,
            ))
    return out


# ---------------------------------------------------------------------------
# Rule 2: required_roles_present
# ---------------------------------------------------------------------------


def rule_required_roles_present(
    decisions: DecisionSet,
    view: SectionView,
    schema: CanonicalSchema,
) -> list[Violation]:
    """Every required role has at least one kept bone."""
    out: list[Violation] = []
    kept = decisions.kept_by_role()
    for r in schema.required_roles:
        if r not in kept or not kept[r]:
            out.append(Violation(
                severity="error",
                rule="required_roles_present",
                message=f"required role '{r}' has no kept bone",
            ))
    return out


# ---------------------------------------------------------------------------
# Rule 3: l_r_pairing — X-mirror within 5%
# ---------------------------------------------------------------------------


def _is_x_mirror(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """X of `a` should be ~ -X of `b`; Y/Z roughly equal within 5%."""
    scale = max(abs(a[0]), abs(b[0]), _LR_MIRROR_ABS_FLOOR)
    x_match = abs(a[0] + b[0]) <= scale * _LR_MIRROR_REL_TOL + _LR_MIRROR_ABS_FLOOR
    y_match = math.isclose(a[1], b[1],
                           rel_tol=_LR_MIRROR_REL_TOL, abs_tol=_LR_MIRROR_ABS_FLOOR)
    z_match = math.isclose(a[2], b[2],
                           rel_tol=_LR_MIRROR_REL_TOL, abs_tol=_LR_MIRROR_ABS_FLOOR)
    return x_match and y_match and z_match


def rule_l_r_pairing(
    decisions: DecisionSet,
    view: SectionView,
    schema: CanonicalSchema,
) -> list[Violation]:
    """For each kept .L role, its .R peer must exist and mirror in X (±5%).

    Note: translations are Lcl Translations (parent-relative). For Y/Z to match
    the peer should share the same parent — which is implicit in canonical
    pairs (Shoulder.L and Shoulder.R both parent under Chest, etc.).
    """
    out: list[Violation] = []
    kept = decisions.kept_by_role()
    for role, ids in kept.items():
        if not role.endswith(".L"):
            continue
        peer = schema.lateral_pair_of(role) or role[:-2] + ".R"
        if peer not in kept or not kept[peer]:
            out.append(Violation(
                severity="error",
                rule="l_r_pairing",
                message=f"{role} is kept but its peer {peer} is missing",
                bone_ids=ids,
            ))
            continue
        # Compare first bone of each (uniqueness rule will catch duplicates)
        l_bone = view.bones.get(ids[0])
        r_bone = view.bones.get(kept[peer][0])
        if l_bone is None or r_bone is None:
            continue
        if not _is_x_mirror(l_bone.translation_xyz, r_bone.translation_xyz):
            out.append(Violation(
                severity="error",
                rule="l_r_pairing",
                message=(
                    f"{role}/{peer} translations don't mirror in X "
                    f"(L={l_bone.translation_xyz}, R={r_bone.translation_xyz})"
                ),
                bone_ids=[l_bone.model_id, r_bone.model_id],
            ))
    return out


# ---------------------------------------------------------------------------
# Rule 4: monotonic_spine_y
# ---------------------------------------------------------------------------


def rule_monotonic_spine_y(
    decisions: DecisionSet,
    view: SectionView,
    schema: CanonicalSchema,
) -> list[Violation]:
    """Spine chain ys must be monotonically increasing (Hips < Spine < ... < Head).

    Y is in *world* space if we summed translations along the chain. Booth FBX
    Lcl Translations are parent-relative — but spine segments are nearly
    aligned along +Y from Hips, so each successive Y delta should be positive
    (i.e. each child's Lcl Y > 0 when measured from its parent).

    Implementation: check each consecutive kept pair in spine_chain. The
    child's Lcl Translation Y must be > 0 (it's offset upward from its parent).
    """
    out: list[Violation] = []
    kept = decisions.kept_by_role()
    chain = schema.spine_chain
    for i in range(1, len(chain)):
        child_role = chain[i]
        if child_role not in kept or not kept[child_role]:
            continue  # absent roles handled by required_roles_present
        child_bone = view.bones.get(kept[child_role][0])
        if child_bone is None:
            continue
        if child_bone.translation_xyz[1] <= 0:
            out.append(Violation(
                severity="error",
                rule="monotonic_spine_y",
                message=(
                    f"{child_role}'s Lcl Y is {child_bone.translation_xyz[1]:.4f}, "
                    f"expected > 0 (spine should ascend)"
                ),
                bone_ids=[child_bone.model_id],
            ))
    return out


# ---------------------------------------------------------------------------
# Rule 5: hierarchy_consistency
# ---------------------------------------------------------------------------


def _is_subsequence(short: list[str], long: list[str]) -> bool:
    """True if `short` appears in `long` in the same order (intermediates may
    be missing). e.g. [A,C,E] is a subseq of [A,B,C,D,E]."""
    i = 0
    for x in long:
        if i < len(short) and short[i] == x:
            i += 1
    return i == len(short)


def rule_hierarchy_consistency(
    decisions: DecisionSet,
    view: SectionView,
    schema: CanonicalSchema,
) -> list[Violation]:
    """Each kept canonical bone's actual ancestor chain must:
       (a) be a subsequence of the schema's expected ancestor chain (same
           order — wrong-side errors like Hand.L under UpperArm.R get caught), and
       (b) contain every *required* canonical ancestor (so e.g. Hand.L can't
           parent straight to Spine, bypassing UpperArm.L and Chest).

    Optional intermediates (Shoulder.L, LowerArm.L) may be absent without
    triggering this rule — they aren't in `schema.required_roles`.

    v2: a Decision can carry `new_parent_role` to express "this bone is kept,
    but its parent in the assembled rig should be the kept bone holding role
    X." When set, the chain walk redirects through that bone (Rail D will
    later emit the matching `reparent_bone_edits`). An unresolvable
    `new_parent_role` (no kept bone has that role) is itself an error — the
    reparent has nowhere to land.
    """
    out: list[Violation] = []
    by_id = decisions.by_id()
    required_set = set(schema.required_roles)

    # Build role -> first kept bone id index for reparent target resolution.
    role_to_bone_id: dict[str, int] = {}
    for d in decisions.bones:
        if d.verdict == "keep" and schema.is_canonical(d.role):
            role_to_bone_id.setdefault(d.role, d.model_id)

    for d in decisions.bones:
        if d.verdict != "keep" or not schema.is_canonical(d.role):
            continue
        bone = view.bones.get(d.model_id)
        if bone is None:
            continue

        # Resolve the effective parent for the walk. A `new_parent_role`
        # overrides the as-input `bone.parent_id` and points at whichever
        # kept bone holds that role.
        if d.new_parent_role is not None:
            target_bone_id = role_to_bone_id.get(d.new_parent_role)
            if target_bone_id is None:
                out.append(Violation(
                    severity="error",
                    rule="hierarchy_consistency",
                    message=(
                        f"bone {d.model_id} (role={d.role}) requests reparent "
                        f"to role '{d.new_parent_role}' but no kept bone "
                        f"holds that role"
                    ),
                    bone_ids=[d.model_id],
                ))
                continue
            start = view.bones.get(target_bone_id)
        else:
            start = view.bones.get(bone.parent_id) if bone.parent_id else None

        # Walk up the effective parent chain, applying reparent overrides at
        # every node we cross. Cycle-guard with a visited set.
        actual_chain: list[str] = [d.role]
        visited: set[int] = {d.model_id}
        cur = start
        while cur is not None and cur.model_id not in visited:
            visited.add(cur.model_id)
            cd = by_id.get(cur.model_id)
            if cd and cd.verdict == "keep" and schema.is_canonical(cd.role):
                actual_chain.append(cd.role)
            if cd is not None and cd.new_parent_role is not None:
                nxt_id = role_to_bone_id.get(cd.new_parent_role)
                cur = view.bones.get(nxt_id) if nxt_id is not None else None
            else:
                cur = view.bones.get(cur.parent_id) if cur.parent_id else None

        expected = schema.ancestor_chain(d.role)

        if not _is_subsequence(actual_chain, expected):
            out.append(Violation(
                severity="error",
                rule="hierarchy_consistency",
                message=(
                    f"bone {d.model_id} (role={d.role}) actual canonical "
                    f"ancestors {actual_chain} not a subsequence of "
                    f"expected {expected} (wrong order or wrong side)"
                ),
                bone_ids=[d.model_id],
            ))
            continue

        required_in_expected = [r for r in expected if r in required_set and r != d.role]
        actual_set = set(actual_chain)
        missing = [r for r in required_in_expected if r not in actual_set]
        if missing:
            out.append(Violation(
                severity="error",
                rule="hierarchy_consistency",
                message=(
                    f"bone {d.model_id} (role={d.role}) actual chain "
                    f"{actual_chain} is missing required ancestors {missing}"
                ),
                bone_ids=[d.model_id],
            ))
    return out


# ---------------------------------------------------------------------------
# Rule 6: drop_safety
# ---------------------------------------------------------------------------


def _subtree_weight(
    bone: BoneRecord,
    view: SectionView,
    memo: dict[int, int],
) -> int:
    """Sum of cluster_weight_count over `bone` and all transitive descendants.
    Memoized; safe against cycles (which shouldn't exist in valid FBX, but the
    extractor never validates DAG-ness)."""
    if bone.model_id in memo:
        return memo[bone.model_id]
    memo[bone.model_id] = -1  # cycle sentinel — recursion returns 0 if hit
    total = bone.cluster_weight_count
    for cid in bone.children_ids:
        child = view.bones.get(cid)
        if child is None:
            continue
        sub = _subtree_weight(child, view, memo)
        if sub > 0:
            total += sub
    memo[bone.model_id] = total
    return total


def rule_drop_safety(
    decisions: DecisionSet,
    view: SectionView,
    schema: CanonicalSchema,
) -> list[Violation]:
    """Dropping a bone with weighted descendants orphans those descendants from
    the rig hierarchy — even if the bone itself has zero weights. Gate on
    SUBTREE weight (this bone + all transitive children).

    PLAN.md cites a 0.05 weight-value threshold (v2 / weight-value-based).
    v1 uses cluster_weight_count (cheap; no per-vertex parsing) with threshold
    `_DROP_SAFETY_MAX_SUBTREE_WEIGHTS`.
    """
    out: list[Violation] = []
    memo: dict[int, int] = {}
    for d in decisions.bones:
        if d.verdict != "drop":
            continue
        if d.drop_category == "user_dropped":
            # Explicit user choice — trust it. The compose-flow pre-filter is
            # responsible for cascading the drop to the subtree, so there are
            # no orphaned weighted descendants for this case.
            continue
        bone = view.bones.get(d.model_id)
        if bone is None:
            continue
        sub = _subtree_weight(bone, view, memo)
        if sub > _DROP_SAFETY_MAX_SUBTREE_WEIGHTS:
            own = bone.cluster_weight_count
            descendant = sub - own
            out.append(Violation(
                severity="error",
                rule="drop_safety",
                message=(
                    f"bone {d.model_id} ({bone.name}) marked drop but its subtree "
                    f"carries {sub} cluster weight bindings "
                    f"(self={own}, descendants={descendant}; > {_DROP_SAFETY_MAX_SUBTREE_WEIGHTS}); "
                    f"dropping would orphan weighted descendants"
                ),
                bone_ids=[d.model_id],
            ))
    return out


# ---------------------------------------------------------------------------
# Rule 7: finger_count_sanity (warn-only in v2; checks L/R parity)
# ---------------------------------------------------------------------------


def rule_finger_count_sanity(
    decisions: DecisionSet,
    view: SectionView,
    schema: CanonicalSchema,
) -> list[Violation]:
    """For each canonical finger role kept on the left side, its right-side
    peer should also be kept (and vice versa). Mismatch is a warning — fingers
    are optional in v2, so a half-rigged hand is suspicious but not fatal.
    Non-canonical finger-like names are ignored here (rule_unique_role handles
    `unknown`/junk role values)."""
    out: list[Violation] = []
    kept = decisions.kept_by_role()
    seen_stems: set[str] = set()
    for role, ids in kept.items():
        rdef = schema.roles.get(role)
        if rdef is None or rdef.category != "finger":
            continue
        if role.endswith(".L"):
            stem = role[:-2]
        elif role.endswith(".R"):
            stem = role[:-2]
        else:
            continue
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        l_role = stem + ".L"
        r_role = stem + ".R"
        l_kept = bool(kept.get(l_role))
        r_kept = bool(kept.get(r_role))
        if l_kept != r_kept:
            missing = r_role if l_kept else l_role
            present = l_role if l_kept else r_role
            out.append(Violation(
                severity="warning",
                rule="finger_count_sanity",
                message=(
                    f"finger {present} kept but its peer {missing} is missing "
                    f"(hands should mirror)"
                ),
                bone_ids=list(kept.get(present, [])),
            ))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


RULES: list[tuple[str, Callable]] = [
    ("unique_role",              rule_unique_role),
    ("required_roles_present",   rule_required_roles_present),
    ("l_r_pairing",              rule_l_r_pairing),
    ("monotonic_spine_y",        rule_monotonic_spine_y),
    ("hierarchy_consistency",    rule_hierarchy_consistency),
    ("drop_safety",              rule_drop_safety),
    ("finger_count_sanity",      rule_finger_count_sanity),
]


def validate_all(
    decisions: DecisionSet,
    view: SectionView,
    schema: CanonicalSchema,
) -> list[Violation]:
    """Run every rule in order. Returns all violations (callers split by severity)."""
    out: list[Violation] = []
    for _name, fn in RULES:
        out.extend(fn(decisions, view, schema))
    return out


def errors(violations: list[Violation]) -> list[Violation]:
    return [v for v in violations if v.severity == "error"]


def warnings(violations: list[Violation]) -> list[Violation]:
    return [v for v in violations if v.severity == "warning"]
