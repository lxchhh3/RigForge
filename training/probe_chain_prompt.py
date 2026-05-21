"""A/B probe: baseline prompt vs enhanced prompt on classic_chic chain bones.

Goal: empirically measure whether prompt engineering alone can rescue the
three chain bones DS V4 Flash dropped (Chain_1.006, Chain_1.009,
PB_beret_chain.001) — all of which have cluster_weight_count=0 but are
load-bearing ancestors of weighted descendants OR labeled chain-family
members.

This is intentionally a small focused probe (14 bones, two real-LLM calls)
so we can answer the "is the prompt at ceiling?" question in ~2 min, not
~12 min.

The bones are the entire `Chain_*` + `PB_beret_chain*` family from classic_chic.

Run:
    PYTHONPATH=. python training/probe_chain_prompt.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.canonical.schema import CanonicalSchema
from rigforge.llm.config import load_config
from rigforge.llm.ollama import OllamaLLMClient, StreamProgress
from rigforge.llm.parser import parse_decision_set
from rigforge.llm.prompts import build_messages


CLOTHING = Path("D:/2files/models/vrc/ASCII_models/clothing/ClassicChic_Maya/FBX_Maya_ascii.fbx")
CHAIN_IDS = {
    2106730200992, 2106730188992, 2106730204992, 2106730206992,
    2106730192992, 2106730208992, 2106730210992, 2106730214992,
    2106730391856, 2106730405856, 2106730395856, 2106730407856,
    2106730669648, 2106730667648,
}
# Three the baseline dropped — these are the rescue targets.
TARGET_DROPPED = {2106730206992, 2106730407856, 2106730667648}
NAME_BY_ID = {
    2106730200992: "PB_Belly_Chain_root",
    2106730188992: "Chain_1",
    2106730204992: "Chain_1.001",
    2106730206992: "Chain_1.006",
    2106730192992: "Chain_1.002",
    2106730208992: "Chain_1.003",
    2106730210992: "Chain_1.004",
    2106730214992: "Chain_1.005",
    2106730391856: "Chain_2",
    2106730405856: "Chain_1.007",
    2106730395856: "Chain_1.008",
    2106730407856: "Chain_1.009",
    2106730669648: "PB_beret_chain",
    2106730667648: "PB_beret_chain.001",
}


def _enhanced_system_message(schema: CanonicalSchema) -> str:
    """Drop `duplicate` from drop categories; add chain-family + subtree-weight rules."""
    canonical = sorted(schema.role_names())
    required = list(schema.required_roles)
    secondary_examples = [
        f"{p.prefix}{sfx}.NN"
        for p in schema.secondary_role_patterns
        for sfx in p.suffixes
    ][:8]
    # NOTE: dropped 'duplicate' from this list.
    drop_categories = ["aux", "twist", "ik_target", "null_locator"]
    return "\n".join([
        "You are a bone classifier for a character-rig retargeting pipeline.",
        "",
        "TASK: For each input bone, assign a canonical ROLE and a VERDICT.",
        "",
        f"CANONICAL ROLES (v{schema.version}):",
        "  " + ", ".join(canonical),
        "",
        "REQUIRED ROLES (the assembled rig must have at least one of each):",
        "  " + ", ".join(required),
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
        "RULES:",
        "  1. Each canonical role appears AT MOST ONCE across kept bones.",
        "  2. .L bones must have matching .R peers (and vice versa).",
        "  3. Spine chain ascends Hips → Spine → Chest → Neck → Head.",
        "  4. Children of canonical bones must follow the canonical parent chain.",
        "  5. DROP RULES — ALL must hold to drop a bone:",
        "     a. The bone itself has cluster_weight_count near 0, AND",
        "     b. The bone has no children that are themselves kept (dropping a",
        "        parent disconnects its kept children from the rig hierarchy), AND",
        "     c. The bone is not a named member of a numbered family that has",
        "        other members kept (see rule 6).",
        "  6. NUMBERED-FAMILY RULE: Blender exports use the suffix convention",
        "     `Name`, `Name.001`, `Name.002`, ... `Name.NNN` for distinct bones",
        "     in the same family (NOT for duplicates). Examples: `Chain_1`,",
        "     `Chain_1.001`, `Chain_1.006`, `PB_beret_chain`, `PB_beret_chain.001`.",
        "     - These suffixes do NOT mean duplicate. They are distinct bones.",
        "     - The numeric suffix does NOT indicate sequence order — follow the",
        "       parent chain to determine link order, not the .NNN suffix.",
        "     - If you keep any member of a numbered family, you MUST keep all",
        "       members of that family unless they meet every condition in rule 5.",
        "  7. CHAIN-LINK / SECONDARY BONES: long flexible chains (hair, skirt,",
        "     ribbon, belt-chain, beret-chain) are usually weighted only at the",
        "     visible tip of each link or sparsely along the segment. A chain",
        "     link with cluster_weight_count=0 between two weighted siblings is",
        "     a structural connector — ALWAYS keep it.",
        "",
        "OUTPUT FORMAT — STRICT JSON, NOTHING ELSE. No prose, no markdown fences.",
        "",
        '{"bones": [',
        '  {"model_id": <int>, "role": "<role>", "verdict": "<keep|drop>",',
        f'   "drop_category": "<{"|".join(drop_categories)}|null>",',
        '   "confidence": <float 0..1>},',
        "  ...one per input bone...",
        "]}",
        "",
        "Output ONE decision per input bone, in the same order as the input list.",
    ])


def _call(client: OllamaLLMClient, messages: list[dict], label: str) -> dict:
    """One Ollama round trip with the given messages; return parsed JSON."""
    print(f"\n--- {label} ---", flush=True)
    t0 = time.time()
    body = client._build_body(messages)
    content = client._post_stream(body)
    elapsed = time.time() - t0
    ds = parse_decision_set(content)
    print(f"  done in {elapsed:.1f}s, {len(ds.bones)} decisions", flush=True)
    return {d.model_id: d for d in ds.bones}


def main() -> int:
    print(f"loading {CLOTHING.name}...", flush=True)
    doc = parse(CLOTHING.read_bytes())
    view = extract(doc)
    bones = [b.to_json_record() for b in view.limb_bones() if b.model_id in CHAIN_IDS]
    print(f"  extracted {len(bones)} chain bones", flush=True)

    cfg = load_config()
    schema = CanonicalSchema.load_default()

    def cb(p: StreamProgress):
        if p.done or int(p.elapsed_seconds) % 3 == 0:
            print(f"  [stream] chunks={p.chunks} bytes={p.content_bytes} "
                  f"t={p.elapsed_seconds:.1f}s done={p.done}", flush=True)

    client = OllamaLLMClient(cfg, schema=schema, progress_callback=cb)

    request = {"donor_id": "maya", "target_id": "maya",
               "canonical_schema_version": schema.version, "bones": bones}

    # Arm A: baseline prompt (the one in production)
    baseline_msgs = build_messages(request=request, schema=schema)
    baseline = _call(client, baseline_msgs, "BASELINE prompt")

    # Arm B: enhanced prompt
    enhanced_sys = _enhanced_system_message(schema)
    user_text = baseline_msgs[1]["content"]   # reuse the user message
    enhanced_msgs = [
        {"role": "system", "content": enhanced_sys},
        {"role": "user", "content": user_text},
    ]
    enhanced = _call(client, enhanced_msgs, "ENHANCED prompt")

    # Verdict table
    print("\n\n" + "=" * 92)
    print(f"{'bone':<22} {'wgt':>5} {'baseline':<26} {'enhanced':<26} {'rescued?'}")
    print("-" * 92)
    bone_recs = {b['model_id']: b for b in bones}
    rescued = []
    regressed = []
    for bid in CHAIN_IDS:
        name = NAME_BY_ID[bid]
        wgt = bone_recs[bid]['cluster_weight_count']
        b = baseline.get(bid)
        e = enhanced.get(bid)
        b_str = f"{b.verdict} ({b.drop_category})" if b else "—"
        e_str = f"{e.verdict} ({e.drop_category})" if e else "—"
        flag = ""
        if bid in TARGET_DROPPED:
            if e and e.verdict == "keep":
                flag = "RESCUED"
                rescued.append(name)
            else:
                flag = "still dropped"
        elif b and e and b.verdict == "keep" and e.verdict == "drop":
            flag = "REGRESSED"
            regressed.append(name)
        print(f"{name:<22} {wgt:>5} {b_str:<26} {e_str:<26} {flag}")
    print("=" * 92)
    print(f"\nrescued ({len(rescued)}/{len(TARGET_DROPPED)}): {rescued}")
    if regressed:
        print(f"REGRESSED ({len(regressed)}): {regressed}")
    out = Path("data/training/_probe/chain_prompt.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "baseline": {k: {"verdict": v.verdict, "drop_category": v.drop_category,
                          "role": v.role, "confidence": v.confidence}
                      for k, v in baseline.items()},
        "enhanced": {k: {"verdict": v.verdict, "drop_category": v.drop_category,
                          "role": v.role, "confidence": v.confidence}
                      for k, v in enhanced.items()},
        "rescued": rescued,
        "regressed": regressed,
    }, indent=2))
    print(f"\nfull result: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
