"""A/B probe on full PicodraTech rig: JSON without world_xyz vs JSON with it.

Prompt is held constant (current production prompt — see prompts.py).
The ONLY difference between arms is the per-bone JSON shape:
  - Arm A (baseline):  7 fields per bone (the pre-v1.1 shape)
  - Arm B (enriched):  11 fields — adds world_xyz, height_pct, lateral, front_back

Goal: measure whether spatial signal alone (with no prompt explanation of
what world_xyz means) changes the LLM's verdict and role assignments on
PicodraTech — the clothing where name-ambiguous skirt panels (OuterA/B/C/D)
need spatial disambiguation that names cannot provide.

Specifically check:
  - Outer* skirt panels: does enriched arm assign SkirtFront/SkirtBack/SkirtSide
    correctly per Z-axis sign?
  - Overall drop count delta + any high-subtree-weight drops
  - Role assignments for required canonical bones (Hips, Spine, etc.)

Run:
    PYTHONPATH=. python training/probe_world_picodratech.py
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


CLOTHING = Path("D:/2files/models/vrc/ASCII_models/clothing/Maya_PicodraTech/PicodraTech_MAYA_ascii.fbx")
OUT_DIR = Path("data/training/_probe/picodratech")


_BASELINE_FIELDS = {
    "model_id", "name", "parent_name", "translation_xyz", "child_names",
    "has_skin_cluster", "cluster_weight_count",
}


def _strip_to_baseline(rec: dict) -> dict:
    """Drop v1.1-only fields so the LLM sees the pre-world_xyz JSON shape."""
    return {k: v for k, v in rec.items() if k in _BASELINE_FIELDS}


def _call(client: OllamaLLMClient, messages: list[dict], label: str) -> dict:
    print(f"\n--- {label} ---", flush=True)
    t0 = time.time()
    body = client._build_body(messages)
    content = client._post_stream(body)
    elapsed = time.time() - t0
    ds = parse_decision_set(content)
    print(f"  done in {elapsed:.1f}s, {len(ds.bones)} decisions", flush=True)
    return {d.model_id: d for d in ds.bones}


def _compute_subtree_weight(view) -> dict[int, int]:
    """Subtree skin weight per bone (sum over self + all descendants)."""
    bones_by_id = {b.model_id: b for b in view.limb_bones()}
    memo: dict[int, int] = {}
    def walk(bid):
        if bid in memo:
            return memo[bid]
        b = bones_by_id.get(bid)
        if b is None:
            return 0
        total = b.cluster_weight_count
        for cid in b.children_ids:
            if cid in bones_by_id:
                total += walk(cid)
        memo[bid] = total
        return total
    for bid in bones_by_id:
        walk(bid)
    return memo


def _classify_summary(decisions: dict, view, subtree_weight: dict[int, int]) -> dict:
    bones_by_id = {b.model_id: b for b in view.limb_bones()}
    keep = drop = 0
    drops_with_subtree_weight = []
    role_assignments = {}
    for bid, d in decisions.items():
        if d.verdict == "keep":
            keep += 1
        else:
            drop += 1
            sw = subtree_weight.get(bid, 0)
            if sw > 0:
                b = bones_by_id.get(bid)
                drops_with_subtree_weight.append({
                    "bone": b.name if b else f"id={bid}",
                    "self_weight": b.cluster_weight_count if b else None,
                    "subtree_weight": sw,
                    "role": d.role,
                    "drop_category": d.drop_category,
                })
        role_assignments.setdefault(d.role, []).append(bid)
    return {
        "keep": keep, "drop": drop,
        "drops_with_subtree_weight": drops_with_subtree_weight,
        "role_count": {role: len(ids) for role, ids in role_assignments.items()},
    }


def _outer_family_table(decisions_a: dict, decisions_b: dict, view) -> str:
    """Print verdict + role for every Outer* bone in both arms."""
    lines = []
    lines.append(f"{'bone':<22} {'world_xyz (x,y,z)':<28} "
                  f"{'baseline role':<22} {'enriched role':<22}")
    lines.append("-" * 96)
    outer = sorted(
        [b for b in view.limb_bones() if b.name.startswith("Outer")],
        key=lambda b: b.name,
    )
    for b in outer:
        a = decisions_a.get(b.model_id)
        e = decisions_b.get(b.model_id)
        wx, wy, wz = b.world_xyz
        a_str = f"{a.role}({a.verdict[0]})" if a else "—"
        e_str = f"{e.role}({e.verdict[0]})" if e else "—"
        lines.append(f"{b.name:<22} ({wx:+.3f},{wy:+.3f},{wz:+.3f})  "
                     f"{a_str:<22} {e_str:<22}")
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"loading {CLOTHING.name}...", flush=True)
    doc = parse(CLOTHING.read_bytes())
    view = extract(doc)
    bones_enriched = [b.to_json_record() for b in view.limb_bones()]
    bones_baseline = [_strip_to_baseline(r) for r in bones_enriched]
    subtree_weight = _compute_subtree_weight(view)
    print(f"  extracted {len(bones_enriched)} bones, "
          f"enriched fields per bone: {len(bones_enriched[0])}", flush=True)

    cfg = load_config()
    schema = CanonicalSchema.load_default()
    last_emit = [0.0]
    def cb(p: StreamProgress):
        now = time.time()
        if p.done or (now - last_emit[0]) > 5.0:
            last_emit[0] = now
            print(f"  [stream] chunks={p.chunks} bytes={p.content_bytes} "
                  f"t={p.elapsed_seconds:.1f}s done={p.done}", flush=True)
    client = OllamaLLMClient(cfg, schema=schema, progress_callback=cb)

    request_base = {"donor_id": "maya", "target_id": "maya",
                    "canonical_schema_version": schema.version}

    # ARM A: baseline 7-field JSON
    msgs_a = build_messages(
        request={**request_base, "bones": bones_baseline}, schema=schema)
    decisions_a = _call(client, msgs_a, "BASELINE JSON (7 fields)")

    # ARM B: enriched 11-field JSON (same prompt)
    msgs_b = build_messages(
        request={**request_base, "bones": bones_enriched}, schema=schema)
    decisions_b = _call(client, msgs_b, "ENRICHED JSON (+ world_xyz, +bucketed)")

    sum_a = _classify_summary(decisions_a, view, subtree_weight)
    sum_b = _classify_summary(decisions_b, view, subtree_weight)

    print("\n\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"keep/drop  baseline={sum_a['keep']}/{sum_a['drop']}   "
          f"enriched={sum_b['keep']}/{sum_b['drop']}")
    print(f"drops with subtree_weight > 0:  baseline={len(sum_a['drops_with_subtree_weight'])}  "
          f"enriched={len(sum_b['drops_with_subtree_weight'])}")
    print()
    print(_outer_family_table(decisions_a, decisions_b, view))

    print("\n\nBASELINE drops that have load-bearing subtree (the real failure metric):")
    for d in sum_a["drops_with_subtree_weight"][:30]:
        print(f"  {d['bone']:<30} subtree_w={d['subtree_weight']:>5} "
              f"role={d['role']!r} cat={d['drop_category']!r}")
    print("\nENRICHED drops that have load-bearing subtree:")
    for d in sum_b["drops_with_subtree_weight"][:30]:
        print(f"  {d['bone']:<30} subtree_w={d['subtree_weight']:>5} "
              f"role={d['role']!r} cat={d['drop_category']!r}")

    out = OUT_DIR / "probe_result.json"
    out.write_text(json.dumps({
        "baseline_summary": sum_a,
        "enriched_summary": sum_b,
        "baseline_decisions": {k: {"verdict": v.verdict, "role": v.role,
                                    "drop_category": v.drop_category,
                                    "confidence": v.confidence}
                                for k, v in decisions_a.items()},
        "enriched_decisions": {k: {"verdict": v.verdict, "role": v.role,
                                    "drop_category": v.drop_category,
                                    "confidence": v.confidence}
                                for k, v in decisions_b.items()},
    }, indent=2))
    print(f"\nfull result: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
