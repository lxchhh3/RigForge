"""Re-run the 3 clothings that failed v1.1 stress, now with v2.0 schema +
new prompt rules (terminal-arm-is-Hand, no-optional-role-overreach,
stronger violation-repair hints).

Expected outcomes:
  - azure_virtue   PASS  (v2 finger roles absorb the 15-bone collapse)
  - classic_chic   PASS  (new wrist rule -> Left wrist = Hand.L)
  - school_uniform PASS  (new optional-role rule -> Jakit_Root != UpperChest)

Outputs to data/training/_verify_v2/ so the v1.1 baseline stays intact.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from rigforge.avatars.registry import AvatarRegistry
from rigforge.canonical.schema import CanonicalSchema
from rigforge.llm.config import load_config
from rigforge.llm.ollama import OllamaLLMClient

# Reuse run_one and helpers from the main stress runner
from training.stress_llm_clothing import (
    CLOTHING_ROOT,
    REPO_ROOT,
    RunReport,
    _make_progress_callback,
    _print,
    run_one,
)


CLOTHINGS = [
    ("classic_chic",   "ClassicChic_Maya",     "FBX_Maya_ascii.fbx"),
    ("azure_virtue",   "Azure_VirtueMaya",     "AzureVirtue_Maya_FBX_ascii.fbx"),
    ("school_uniform", "Maya_SchoolUniform",   "Maya_SchoolUniform_ascii.fbx"),
]


def main() -> int:
    work = REPO_ROOT / "data" / "training" / "_verify_v21"
    work.mkdir(parents=True, exist_ok=True)

    config = load_config()
    schema = CanonicalSchema.load_default()
    registry = AvatarRegistry.load_default()
    client = OllamaLLMClient(config, schema=schema,
                              progress_callback=_make_progress_callback())
    _print(f"LLM:    model={config.model}  schema_version={schema.version}")
    _print(f"Target: maya")
    _print(f"Re-running {len(CLOTHINGS)} previously-failing clothings...")

    reports: list[RunReport] = []
    for slug, dir_name, ascii_name in CLOTHINGS:
        ascii_path = CLOTHING_ROOT / dir_name / ascii_name
        rep = run_one(slug, ascii_path, work, registry, schema, client)
        reports.append(rep)
        summary_path = work / "_summary.json"
        summary_path.write_text(json.dumps(
            [asdict(r) for r in reports], indent=2, ensure_ascii=False))

    _print(f"\n\n{'='*72}\nVERIFY SUMMARY\n{'='*72}")
    _print(f"{'slug':<16} {'ok':<4} {'bones':<6} {'keep':<5} {'drop':<5} {'time':<6}")
    _print("-" * 72)
    for r in reports:
        status = "OK" if r.ok else "FAIL"
        _print(f"{r.slug:<16} {status:<4} {r.total_limb_bones:<6} "
              f"{r.keep_count:<5} {r.drop_count:<5} {r.elapsed_seconds:>5.1f}s")
        if not r.ok:
            _print(f"  -> {r.error}")

    failures = sum(1 for r in reports if not r.ok)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
