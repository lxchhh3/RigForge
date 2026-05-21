"""LLMClient — abstract Rail L interface.

Per PLAN.md, Phase B step 3 is a single batched LLM call. This module defines
the protocol every concrete client (mock, Ollama, etc.) implements. Keeping
this thin lets the pipeline depend on the interface and lets us defer the
prompt-engineering + real Ollama wiring until the LoRA weights ship.

Input shape (the `LLMRequest`)
------------------------------
{
    "donor_id": "vroid_default",
    "target_id": "maya",
    "canonical_schema_version": "1.0",
    "bones": [
        {"model_id": ..., "name": ..., "parent_name": ..., "translation_xyz": [...],
         "child_names": [...], "has_skin_cluster": bool, "cluster_weight_count": int},
        ...
    ]
}

Output shape — validated into `DecisionSet`
-------------------------------------------
{
    "bones": [
        {"model_id": ..., "role": "Hips" | "aux" | "unknown" | "Breast.L.01" | ...,
         "verdict": "keep" | "drop",
         "drop_category": "aux" | "twist" | ... | null,
         "confidence": 0.0..1.0},
        ...
    ]
}

The validator (canonical/validators.py) operates on `DecisionSet`. Errors
bubble back to the orchestrator, which re-prompts once.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from rigforge.canonical.decisions import DecisionSet


@runtime_checkable
class LLMClient(Protocol):
    """Rail L contract: one synchronous classify() call per pipeline run."""

    @property
    def model_id(self) -> str:
        """Stable identifier (vendor + version) — folded into the cache key."""
        ...

    def classify(self, request: dict[str, Any]) -> DecisionSet:
        """Send a batched-bone request, return validated decisions.

        Implementations are responsible for:
          - JSON serialization / RPC details
          - Schema validation of the response (use DecisionSet.model_validate)
          - Raising on transport / parse failures
        """
        ...


class LLMError(RuntimeError):
    """Transport or response-parse failure. Validator-level errors don't raise
    here — they surface as Violations from canonical/validators.py."""
