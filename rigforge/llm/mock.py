"""MockLLMClient — fixture-driven Rail L stand-in.

Loads a DecisionSet from a JSON file (or in-memory dict) and returns it on
every classify() call. Used for:
  - unit tests of validators / pipeline (deterministic input)
  - end-to-end smoke tests before the LoRA-fine-tuned client is ready
  - regression fixtures: capture a real-model run once, replay it forever

Drop-in compatible with `LLMClient`: the orchestrator never knows it isn't
the real thing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from rigforge.canonical.decisions import DecisionSet

from .client import LLMClient, LLMError


class MockLLMClient:
    """Returns canned decisions, ignoring the request payload.

    For multi-fixture scenarios use `MockLLMClient.from_dispatch` (selects a
    fixture by donor_id+target_id).
    """

    def __init__(
        self,
        decisions: Union[DecisionSet, dict, Path, str],
        *,
        model_id: str = "mock-llm@fixture",
    ) -> None:
        self._model_id = model_id
        self._decisions = _coerce_decisions(decisions)

    @property
    def model_id(self) -> str:
        return self._model_id

    def classify(self, request: dict[str, Any]) -> DecisionSet:
        # Defensive: verify the fixture covers every bone in the request.
        request_ids = {b["model_id"] for b in request.get("bones", [])}
        fixture_ids = {d.model_id for d in self._decisions.bones}
        missing = request_ids - fixture_ids
        if missing:
            raise LLMError(
                f"mock fixture missing decisions for bone ids {sorted(missing)[:10]}"
                f" (request had {len(request_ids)} bones, fixture has {len(fixture_ids)})"
            )
        return self._decisions

    @classmethod
    def from_dispatch(
        cls,
        dispatch: dict[tuple[str, str], Union[DecisionSet, dict, Path, str]],
        *,
        model_id: str = "mock-llm@dispatch",
    ) -> "DispatchMockClient":
        return DispatchMockClient(dispatch, model_id=model_id)


class DispatchMockClient:
    """Picks the right fixture based on (donor_id, target_id) in the request."""

    def __init__(
        self,
        dispatch: dict[tuple[str, str], Union[DecisionSet, dict, Path, str]],
        *,
        model_id: str = "mock-llm@dispatch",
    ) -> None:
        self._model_id = model_id
        self._dispatch: dict[tuple[str, str], DecisionSet] = {
            key: _coerce_decisions(val) for key, val in dispatch.items()
        }

    @property
    def model_id(self) -> str:
        return self._model_id

    def classify(self, request: dict[str, Any]) -> DecisionSet:
        donor = request.get("donor_id")
        target = request.get("target_id")
        key = (donor, target)
        if key not in self._dispatch:
            raise LLMError(
                f"no mock fixture for (donor={donor!r}, target={target!r}); "
                f"known: {sorted(self._dispatch)}"
            )
        return self._dispatch[key]


def _coerce_decisions(value: Union[DecisionSet, dict, Path, str]) -> DecisionSet:
    if isinstance(value, DecisionSet):
        return value
    if isinstance(value, dict):
        return DecisionSet.model_validate(value)
    if isinstance(value, (str, Path)):
        path = Path(value)
        data = json.loads(path.read_text(encoding="utf-8"))
        return DecisionSet.model_validate(data)
    raise TypeError(f"unsupported decisions type: {type(value).__name__}")
