"""Run report emitter — distills a PipelineRun into a JSON manifest.

The manifest accompanies every output FBX and gives reviewers (or the
collaborator who'll feed corrections back into training data) a stable record
of what the LLM said, what was edited, and what validators flagged.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rigforge.pipeline.orchestrator import PipelineRun


def build_manifest(run: PipelineRun) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "donor_id": run.donor_id,
            "target_id": run.target_id,
            "donor_score": run.score,
            "candidates": [{"id": c[0], "score": c[1]} for c in run.candidates],
        },
        "output_fbx": str(run.output_fbx),
        "cache": {
            "key": run.cache_key,
            "hit": run.cache_hit,
        },
        "edit_plan": {
            "drops": list(run.edit_plan.drops),
            "renames": dict(run.edit_plan.renames),
            "n_drops": len(run.edit_plan.drops),
            "n_renames": len(run.edit_plan.renames),
        },
        "warnings": [
            {"rule": v.rule, "message": v.message, "bone_ids": list(v.bone_ids)}
            for v in run.warnings
        ],
        "decisions_summary": {
            "kept": sum(1 for d in run.phase_b.decisions.bones if d.verdict == "keep"),
            "dropped": sum(1 for d in run.phase_b.decisions.bones if d.verdict == "drop"),
            "llm_model_id": run.phase_b.decisions.llm_model_id,
        },
        "notes": list(run.notes),
    }


def write_manifest(run: PipelineRun, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_manifest(run), indent=2), encoding="utf-8")
    return path
