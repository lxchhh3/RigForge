"""End-to-end smoke against the REAL Ollama Cloud API.

Builds synth clothing → orchestrator.assemble() with OllamaLLMClient →
binary FBX → fbx_compare against original Maya.fbx.

Cost: one full Ollama call (~10-15K tokens out) plus two fbx_env subprocess
hops. Use after smoke_ollama.py to confirm the full pipeline closes.

Usage:
    PYTHONPATH=. python training/smoke_full_pipeline.py
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
from training.synth_clothing import build_synth_clothing


REPO_ROOT = Path(__file__).resolve().parent.parent
MAYA_TOOLCHAIN = Path("D:/2files/models/vrc/Maya/Maya_Ver1.02.2")
MAYA_ASCII = MAYA_TOOLCHAIN / "Maya_ascii.fbx"
MAYA_BIN = MAYA_TOOLCHAIN / "Maya.fbx"


def main() -> int:
    work = REPO_ROOT / "data" / "training" / "_smoke_full"
    work.mkdir(parents=True, exist_ok=True)
    synth_ascii = work / "synth_clothing_ascii.fbx"
    synth_decisions = work / "synth_decisions.json"

    if not synth_ascii.exists():
        build_synth_clothing(
            source_ascii=MAYA_ASCII,
            out_ascii=synth_ascii,
            out_decisions=synth_decisions,
        )
        print(f"[setup] built synth clothing at {synth_ascii}")
    else:
        print(f"[setup] reusing existing synth clothing {synth_ascii}")

    config = load_config()
    schema = CanonicalSchema.load_default()
    registry = AvatarRegistry.load_default()
    client = OllamaLLMClient(config, schema=schema)

    out_fbx = work / "assembled.fbx"
    print(f"[run] calling assemble() with model={config.model}...")
    t0 = time.time()
    run = assemble(
        clothing_fbx=synth_ascii,
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
    if run.warnings:
        print(f"      warnings ({len(run.warnings)}):")
        for w in run.warnings:
            print(f"        - {w.rule}: {w.message}")

    manifest_path = out_fbx.with_suffix(".manifest.json")
    write_manifest(run, manifest_path)
    print(f"      manifest: {manifest_path}")

    print(f"[compare] running fbx_compare(assembled, Maya.fbx)...")
    cr = compare(out_fbx, MAYA_BIN)
    print(f"          identical={cr.identical}  drift={cr.drift_count}  schema_match={cr.schema_match}")
    if not cr.identical:
        # show only the diff lines, not the full report
        for line in cr.raw_output.splitlines():
            if "DIFF" in line:
                print(f"          {line}")
        return 1
    print("[ok] full pipeline closes — synth clothing → real Ollama → binary FBX → structurally identical to Maya.fbx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
