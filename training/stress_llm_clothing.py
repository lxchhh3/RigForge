"""Stress-test DS V4 Flash on real clothing → Maya across N samples.

Goal: validate the LLM's clothing-classification capability on a meaningful
sample so we can decide whether DS V4 Flash zero-shot is good enough for v1,
or whether we need DS V4 Pro / LoRA / stricter prompt rails.

What's checked per clothing:
  - Phase A: donor score against the curated registry
  - Phase B: real Ollama call, validator pass/fail, re-prompt count
  - EditPlan: drops + renames count
  - Drop sanity: any dropped bone whose cluster_weight_count > 100 is a
    RED FLAG (model removed a real skinned bone)
  - Phase C: pass-through (donor==target=maya)
  - Output FBX written + structurally valid

Outputs:
  - data/training/_stress/<slug>.manifest.json per run
  - data/training/_stress/_summary.json with the aggregated table
  - stdout summary at the end

Usage:
    PYTHONPATH=. python training/stress_llm_clothing.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from rigforge.avatars.registry import AvatarRegistry
from rigforge.canonical.schema import CanonicalSchema
from rigforge.llm.config import load_config
from rigforge.llm.ollama import OllamaLLMClient
from rigforge.manifest import build_manifest, write_manifest
from rigforge.pipeline.orchestrator import PipelineError, assemble


def _print(*args, **kwargs):
    """Always-flushed print so progress streams live to redirected stdout."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


REPO_ROOT = Path(__file__).resolve().parent.parent
CLOTHING_ROOT = Path("D:/2files/models/vrc/ASCII_models/clothing")


# (slug, dir_name, ascii_filename) — using ASCII so the bin↔ASCII id space
# stays stable across the runtime view + cached manifests.
CLOTHINGS = [
    ("classic_chic",   "ClassicChic_Maya",     "FBX_Maya_ascii.fbx"),
    ("azure_virtue",   "Azure_VirtueMaya",     "AzureVirtue_Maya_FBX_ascii.fbx"),
    ("rabbit_gear",    "Maya_RabbitGear",      "Maya_RabbitGear_ascii.fbx"),
    ("picodra_tech",   "Maya_PicodraTech",     "PicodraTech_MAYA_ascii.fbx"),
    ("school_uniform", "Maya_SchoolUniform",   "Maya_SchoolUniform_ascii.fbx"),
]


@dataclass
class RunReport:
    slug: str
    clothing_path: str
    ok: bool
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    donor_id: Optional[str] = None
    donor_score: Optional[float] = None
    total_limb_bones: int = 0
    decisions_count: int = 0
    keep_count: int = 0
    drop_count: int = 0
    renames_count: int = 0
    drops_count: int = 0
    cache_hit: bool = False
    warnings: int = 0
    high_weight_drops: list[dict] = field(default_factory=list)
    role_distribution: dict[str, int] = field(default_factory=dict)
    output_fbx: Optional[str] = None
    manifest_path: Optional[str] = None


def _drop_quality_flags(run, view, decisions) -> list[dict]:
    """Identify drops whose cluster_weight_count > 100 — these are real
    skinned bones the model is over-pruning."""
    bones_by_id = {b.model_id: b for b in view.limb_bones()}
    flags = []
    for bid in run.edit_plan.drops:
        b = bones_by_id.get(bid)
        if b is None:
            continue
        if b.cluster_weight_count > 100:
            decision = next((d for d in decisions.bones if d.model_id == bid), None)
            flags.append({
                "model_id": bid,
                "name": b.name,
                "cluster_weight_count": b.cluster_weight_count,
                "decision_role": decision.role if decision else None,
                "decision_category": decision.drop_category if decision else None,
            })
    return flags


def _role_distribution(decisions) -> dict[str, int]:
    dist: dict[str, int] = {}
    for d in decisions.bones:
        key = d.role if not d.role.startswith("Secondary.") else "Secondary.*"
        dist[key] = dist.get(key, 0) + 1
    return dist


def run_one(
    slug: str,
    ascii_path: Path,
    work_root: Path,
    registry: AvatarRegistry,
    schema: CanonicalSchema,
    client: OllamaLLMClient,
) -> RunReport:
    rep = RunReport(slug=slug, clothing_path=str(ascii_path), ok=False)
    out_fbx = work_root / f"{slug}__maya.fbx"
    manifest_path = work_root / f"{slug}.manifest.json"

    _print(f"\n{'='*72}\n[{slug}]  {ascii_path.name}\n{'='*72}")
    t0 = time.time()
    try:
        run = assemble(
            clothing_fbx=ascii_path,
            target_id="maya",
            out_fbx=out_fbx,
            registry=registry,
            schema=schema,
            llm_client=client,
            cache=None,
            work_dir=work_root / f"{slug}_work",
        )
    except PipelineError as e:
        rep.error = f"PipelineError: {e}"
        rep.elapsed_seconds = time.time() - t0
        _print(f"  FAIL: {rep.error}")
        return rep
    except Exception as e:
        rep.error = f"{type(e).__name__}: {e}"
        rep.elapsed_seconds = time.time() - t0
        _print(f"  FAIL: {rep.error}")
        traceback.print_exc()
        return rep

    rep.ok = True
    rep.elapsed_seconds = time.time() - t0
    rep.donor_id = run.donor_id
    rep.donor_score = run.score
    view = run.phase_a.view
    rep.total_limb_bones = len(view.limb_bones())
    decisions = run.phase_b.decisions
    rep.decisions_count = len(decisions.bones)
    rep.keep_count = sum(1 for d in decisions.bones if d.verdict == "keep")
    rep.drop_count = sum(1 for d in decisions.bones if d.verdict == "drop")
    rep.renames_count = len(run.edit_plan.renames)
    rep.drops_count = len(run.edit_plan.drops)
    rep.cache_hit = run.cache_hit
    rep.warnings = len(run.warnings)
    rep.high_weight_drops = _drop_quality_flags(run, view, decisions)
    rep.role_distribution = _role_distribution(decisions)
    rep.output_fbx = str(out_fbx)

    write_manifest(run, manifest_path)
    rep.manifest_path = str(manifest_path)

    _print(f"  elapsed: {rep.elapsed_seconds:.1f}s")
    _print(f"  donor:   {rep.donor_id}  score={rep.donor_score:.3f}")
    _print(f"  bones:   {rep.total_limb_bones} limbs in clothing")
    _print(f"  edits:   {rep.renames_count} renames, {rep.drops_count} drops")
    _print(f"  verdicts: keep={rep.keep_count}  drop={rep.drop_count}")
    _print(f"  warnings: {rep.warnings}")
    if rep.high_weight_drops:
        _print(f"  RED FLAG: {len(rep.high_weight_drops)} drops with weight>100:")
        for d in rep.high_weight_drops[:10]:
            _print(f"    - {d['name']!r:<35} wc={d['cluster_weight_count']:>5}  "
                  f"role={d['decision_role']!r:<22} cat={d['decision_category']}")
    else:
        _print(f"  drop sanity: OK (no drops with weight > 100)")
    return rep


def _make_progress_callback():
    """Throttled heartbeat: emit a line at most every 3s and on done."""
    state = {"last_emit": 0.0}

    def cb(p):
        now = time.time()
        if not p.done and (now - state["last_emit"]) < 3.0:
            return
        state["last_emit"] = now
        marker = "DONE" if p.done else "live"
        _print(f"  [llm:{marker}] chunks={p.chunks}  bytes={p.content_bytes}  "
               f"elapsed={p.elapsed_seconds:.1f}s")
    return cb


def main() -> int:
    # v1.1 run: world_xyz in JSON + subtree-weight drop_safety.
    # Earlier v1.0 outputs live in `_stress/` and stay untouched for A/B
    # comparison in REPORT.md.
    work = REPO_ROOT / "data" / "training" / "_stress_v1_1"
    work.mkdir(parents=True, exist_ok=True)

    config = load_config()
    schema = CanonicalSchema.load_default()
    registry = AvatarRegistry.load_default()
    client = OllamaLLMClient(config, schema=schema,
                              progress_callback=_make_progress_callback())
    _print(f"LLM:     model={config.model}  timeout={config.timeout_seconds}s")
    _print(f"Target:  maya  ({len(registry.get('maya').canonical_to_name)} canonical mappings)")
    _print(f"Clothings: {len(CLOTHINGS)}")

    reports: list[RunReport] = []
    for slug, dir_name, ascii_name in CLOTHINGS:
        ascii_path = CLOTHING_ROOT / dir_name / ascii_name
        if not ascii_path.exists():
            rep = RunReport(slug=slug, clothing_path=str(ascii_path), ok=False,
                            error=f"file not found: {ascii_path}")
            reports.append(rep)
            _print(f"\n[{slug}]  SKIP - file not found: {ascii_path}")
            continue
        rep = run_one(slug, ascii_path, work, registry, schema, client)
        reports.append(rep)
        # Persist incrementally so a crash doesn't lose prior results.
        summary_path = work / "_summary.json"
        summary_path.write_text(json.dumps(
            [asdict(r) for r in reports], indent=2, ensure_ascii=False))

    # Final report
    _print(f"\n\n{'='*72}\nSUMMARY\n{'='*72}")
    _print(f"{'slug':<16} {'ok':<4} {'bones':<6} {'keep':<5} {'drop':<5} "
          f"{'flags':<6} {'time':<6}")
    _print("-" * 72)
    for r in reports:
        status = "OK" if r.ok else "FAIL"
        flags = len(r.high_weight_drops)
        _print(f"{r.slug:<16} {status:<4} {r.total_limb_bones:<6} "
              f"{r.keep_count:<5} {r.drop_count:<5} {flags:<6} "
              f"{r.elapsed_seconds:>5.1f}s")
        if not r.ok:
            _print(f"  └ {r.error}")
        elif r.high_weight_drops:
            _print(f"  └ {len(r.high_weight_drops)} suspicious drops "
                  f"(see manifest: {r.manifest_path})")

    summary_path = work / "_summary.json"
    summary_path.write_text(json.dumps(
        [asdict(r) for r in reports], indent=2, ensure_ascii=False))
    _print(f"\nfull summary: {summary_path}")

    failures = sum(1 for r in reports if not r.ok)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
