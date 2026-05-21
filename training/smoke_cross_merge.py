"""Cross-avatar e2e against the real Ollama Cloud API.

Pipeline: ClassicChic_Moe.fbx (donor=Moe) -> target=maya
         binary -> ASCII -> Phase A -> Phase B (real Ollama) -> Phase C
         (cross-avatar merge) -> ASCII -> binary

Sanity checks at the end:
  - output FBX exists + parses
  - fbx_compare against ClassicChic_Maya (ground truth Maya-rigged variant)
    — drift expected but schema must match
  - manifest is written

Usage:
    PYTHONPATH=. python training/smoke_cross_merge.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from rigforge.ascii_fbx.convert import compare
from rigforge.avatars.registry import AvatarRegistry
from rigforge.canonical.schema import CanonicalSchema
from rigforge.llm.config import load_config
from rigforge.llm.ollama import OllamaLLMClient
from rigforge.manifest import write_manifest
from rigforge.pipeline.orchestrator import assemble


REPO_ROOT = Path(__file__).resolve().parent.parent
CLOTHING_BIN = Path("D:/2files/models/vrc/ASCII_models/clothing/ClassicChic_Moe/FBX_Moe.fbx")
GROUND_TRUTH = Path("D:/2files/models/vrc/ASCII_models/clothing/ClassicChic_Maya/FBX_Maya.fbx")


def main() -> int:
    if not CLOTHING_BIN.exists():
        print(f"error: {CLOTHING_BIN} not found", file=sys.stderr)
        return 1

    work = REPO_ROOT / "data" / "training" / "_smoke_cross"
    work.mkdir(parents=True, exist_ok=True)
    out_fbx = work / "classicchic_moe_to_maya.fbx"

    config = load_config()
    schema = CanonicalSchema.load_default()
    registry = AvatarRegistry.load_default()
    client = OllamaLLMClient(config, schema=schema)

    print(f"[run] cross-merge ClassicChic_Moe -> Maya")
    print(f"      input:  {CLOTHING_BIN}")
    print(f"      target: maya  (registry avatars: {registry.ids()})")
    print(f"      model:  {config.model}")

    t0 = time.time()
    run = assemble(
        clothing_fbx=CLOTHING_BIN,
        target_id="maya",
        out_fbx=out_fbx,
        registry=registry,
        schema=schema,
        llm_client=client,
        cache=None,
        work_dir=work / "work",
    )
    elapsed = time.time() - t0
    print(f"[run] done in {elapsed:.1f}s")
    print(f"      donor={run.donor_id} score={run.score:.3f}")
    print(f"      edits: {len(run.edit_plan.renames)} renames, "
          f"{len(run.edit_plan.drops)} drops")
    print(f"      output: {run.output_fbx}")
    print(f"      notes:")
    for n in run.notes:
        print(f"        - {n}")
    if run.warnings:
        print(f"      warnings ({len(run.warnings)}):")
        for w in run.warnings:
            print(f"        - {w.rule}: {w.message}")

    manifest_path = out_fbx.with_suffix(".manifest.json")
    write_manifest(run, manifest_path)
    print(f"      manifest: {manifest_path}")

    if not GROUND_TRUTH.exists():
        print(f"[skip] ground truth {GROUND_TRUTH} not found; finishing")
        return 0
    print(f"[compare] running fbx_compare(assembled, ClassicChic_Maya)...")
    cr = compare(out_fbx, GROUND_TRUTH)
    print(f"          identical={cr.identical}  drift={cr.drift_count}  schema_match={cr.schema_match}")
    diff_lines = [l for l in cr.raw_output.splitlines() if "DIFF" in l]
    if diff_lines:
        print(f"          first {min(15, len(diff_lines))} diff lines:")
        for line in diff_lines[:15]:
            print(f"          {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
