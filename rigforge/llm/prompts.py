"""Phase B classification prompt builder.

The base DS V4 Flash has no canonical-schema knowledge baked in (no LoRA in v1
— see PLAN.md Open Question #1 and the conversation log around the RunPod
swap-out). So the schema is embedded *in every prompt*. When the fine-tune
lands, this prompt simplifies dramatically (the model already knows the
roles); the call site doesn't change.

Output contract — model MUST return ONLY JSON of shape:
    {"bones": [
        {"model_id": <int>, "role": <str>, "verdict": "keep"|"drop",
         "drop_category": <str|null>, "confidence": <float 0..1>},
        ...
    ]}
"""
from __future__ import annotations

import json
from typing import Any

from rigforge.canonical.schema import CanonicalSchema


def build_messages(
    *,
    request: dict[str, Any],
    schema: CanonicalSchema,
    previous_violations: list[str] | None = None,
) -> list[dict[str, str]]:
    """Return Ollama-style chat messages: [{role, content}, ...].

    `previous_violations`, if non-empty, triggers the re-prompt-once shape:
    the orchestrator passes Phase B validator error strings here so the model
    can correct itself. (The pipeline.phase_b.run_phase_b function adds
    `previous_violations` to its request payload — we surface them through.)
    """
    system = _system_message(schema)
    user = _user_message(request)
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if previous_violations:
        msgs.append({
            "role": "user",
            "content": _violation_repair_message(previous_violations),
        })
    return msgs


def _system_message(schema: CanonicalSchema) -> str:
    canonical_roles = sorted(schema.role_names())
    required_roles = list(schema.required_roles)
    drop_categories = list(schema.drop_categories)
    # Show one example per prefix (covers all secondary role types without
    # blowing up the prompt) — pick `.C` if available, else the first suffix.
    secondary_examples = []
    for p in schema.secondary_role_patterns:
        sfx = ".C" if ".C" in p.suffixes else p.suffixes[0]
        secondary_examples.append(f"{p.prefix}{sfx}.NN")
    lines = [
        "You are a bone classifier for a character-rig retargeting pipeline.",
        "",
        "TASK: For each input bone, assign a canonical ROLE and a VERDICT.",
        "",
        f"CANONICAL ROLES (v{schema.version}):",
        "  " + ", ".join(canonical_roles),
        "",
        "REQUIRED ROLES (the assembled rig must have at least one of each):",
        "  " + ", ".join(required_roles),
        "",
        "SECONDARY-CHAIN ROLE EXAMPLES (numbered, can repeat):",
        "  " + ", ".join(secondary_examples),
        "  (e.g. Breast.L.01, HairSecondary.C.05 — index is a small integer).",
        "",
        "SPECIAL ROLES:",
        '  - "aux"     : non-functional bone (IK target, twist corrective, dummy null)',
        '             — pair with verdict="drop"',
        '  - "unknown" : ONLY if classification is impossible. Triggers hard-fail.',
        "",
        'VERDICT must be "keep" or "drop". Junk bones get verdict="drop".',
        "",
        f"DROP_CATEGORY (when verdict=drop): one of {drop_categories} or null.",
        "",
        "OPTIONAL FIELD — new_parent_role (use only when needed):",
        "  If a bone is correctly identified but parented WRONG in the input",
        "  (e.g. Hand.L's parent in input is Spine, but canonically should be",
        "  LowerArm.L), set new_parent_role to the canonical role its parent",
        "  SHOULD hold. The pipeline will reparent it at edit time. Omit the",
        "  field for bones whose input parent already matches the canonical",
        "  chain. The target role must itself be assigned to another kept bone.",
        "",
        "RULES:",
        "  1. Each canonical role appears AT MOST ONCE across kept bones.",
        "  2. .L bones must have matching .R peers (and vice versa).",
        "  3. Spine chain ascends Hips → Spine → Chest → Neck → Head.",
        "  4. Children of canonical bones must follow the canonical parent chain",
        "     (use new_parent_role to fix mismatches, don't drop a real bone).",
        "  5. Drop only low-influence bones (cluster_weight_count near 0).",
        "  6. The TERMINAL arm bone — the deepest one whose only children are",
        "     fingers or aux/accessory bones (no further forearm) — is Hand.L/.R,",
        "     NOT LowerArm.L/.R. In Maya-style chains 'shoulder→arm→elbow→wrist',",
        "     the 'wrist' bone is Hand and the 'elbow' bone is LowerArm. The",
        "     last named segment (wrist / hand / palm / paw) is always Hand.",
        "  7. Optional canonical roles (e.g. UpperChest) may be left unfilled.",
        "     Only assign them when a real rig bone genuinely sits between the",
        "     canonical parent and child (UpperChest must lie between Chest and",
        "     Neck on the spine). Do NOT assign an optional spine/limb role to",
        "     an accessory or attachment bone (jacket roots, ribbon roots, hair",
        "     anchors, decorative chains) just because it sits nearby.",
        "  8. STRUCTURAL-PARENT bones — those whose CHILDREN carry skin weights",
        "     but the bone itself has cluster_weight_count near 0 and doesn't",
        "     fit any canonical role (ribbon roots, jacket roots, decorative",
        "     chain anchors, accessory holders) — MUST be KEPT (verdict=keep),",
        "     labeled with the Accessory.C.NN / Accessory.L.NN / Accessory.R.NN",
        "     secondary role. Dropping such a bone would orphan its weighted",
        "     descendants from the rig hierarchy. Pick the suffix by position:",
        "     centered → .C, left/right side → .L/.R. NN starts at 01 and",
        "     increments per Accessory bone kept.",
        "",
        "OUTPUT FORMAT — STRICT JSON, NOTHING ELSE. No prose, no markdown fences.",
        "",
        '{"bones": [',
        '  {"model_id": <int>, "role": "<role>", "verdict": "<keep|drop>",',
        '   "drop_category": "<aux|twist|ik_target|null_locator|duplicate|null>",',
        '   "confidence": <float 0..1>,',
        '   "new_parent_role": "<canonical role or omit>"},',
        "  ...one per input bone...",
        "]}",
        "",
        "Output ONE decision per input bone, in the same order as the input list.",
    ]
    return "\n".join(lines)


def _user_message(request: dict[str, Any]) -> str:
    donor = request.get("donor_id", "?")
    target = request.get("target_id", "?")
    bones = request.get("bones", [])
    payload = {
        "donor_id": donor,
        "target_id": target,
        "n_bones": len(bones),
        "bones": bones,
    }
    return (
        f"Classify these bones. The clothing was rigged for donor='{donor}', "
        f"to be retargeted onto target='{target}'.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _violation_repair_message(violations: list[str]) -> str:
    lines = [
        "Your previous response failed validation. Fix these errors and reply with the corrected JSON:",
        "",
    ]
    for v in violations:
        lines.append(f"  - {v}")
    lines.append("")
    lines.append("COMMON REPAIRS:")
    lines.append(
        "  - unique_role / 'canonical role X assigned to N bones': pick the ONE"
    )
    lines.append(
        "    bone that best fits role X (usually the most central / least"
    )
    lines.append(
        "    indexed) and REASSIGN the others to a more accurate canonical role"
    )
    lines.append(
        "    (often a finger like Thumb1.L, a twist like LowerArm.Twist.L) or"
    )
    lines.append(
        "    to aux/drop. Do NOT just shuffle which bones share role X."
    )
    lines.append(
        "  - hierarchy_consistency / 'missing required ancestors': the offending"
    )
    lines.append(
        "    bone is almost certainly mis-labeled — its real-rig position can't"
    )
    lines.append(
        "    support that canonical role. CHANGE its role to aux (verdict=drop)"
    )
    lines.append(
        "    or to a secondary/decorative role that matches its actual parent."
    )
    lines.append(
        "    Do NOT invent a missing ancestor by relabeling another bone."
    )
    lines.append(
        "  - drop_safety / 'subtree carries N cluster weight bindings': the"
    )
    lines.append(
        "    bone's children carry skin weights even though the bone itself"
    )
    lines.append(
        "    doesn't. KEEP it (verdict=keep) and assign Accessory.C.NN (or"
    )
    lines.append(
        "    .L.NN / .R.NN). Dropping it orphans its weighted descendants."
    )
    lines.append("")
    lines.append("Return ONLY the corrected JSON object — same shape as before.")
    return "\n".join(lines)
