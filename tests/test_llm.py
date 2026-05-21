"""Tests for the LLM interface + MockLLMClient.

The real OllamaLLMClient is intentionally not exercised here — its `classify()`
raises until the LoRA fine-tune lands (see rigforge/llm/ollama.py docstring).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.canonical.decisions import Decision, DecisionSet
from rigforge.llm.client import LLMClient, LLMError
from rigforge.llm.mock import DispatchMockClient, MockLLMClient
from rigforge.llm.ollama import OllamaLLMClient


def _ds(*decisions: Decision) -> DecisionSet:
    return DecisionSet(bones=list(decisions))


# --- Interface conformance --------------------------------------------------


def test_mock_client_implements_protocol():
    mock = MockLLMClient(_ds(Decision(model_id=1, role="Hips", verdict="keep")))
    assert isinstance(mock, LLMClient)


def test_ollama_client_implements_protocol():
    from rigforge.llm.config import OllamaConfig
    cfg = OllamaConfig(model="test-model", api_key="dummy-protocol-only")
    real = OllamaLLMClient(cfg)
    assert isinstance(real, LLMClient)
    assert real.model_id.startswith("ollama:")


# --- MockLLMClient ----------------------------------------------------------


def test_mock_classify_returns_fixture():
    decisions = _ds(
        Decision(model_id=1, role="Hips", verdict="keep"),
        Decision(model_id=2, role="Spine", verdict="keep"),
    )
    mock = MockLLMClient(decisions)
    result = mock.classify({
        "donor_id": "x", "target_id": "y",
        "bones": [{"model_id": 1}, {"model_id": 2}],
    })
    assert result == decisions
    assert mock.model_id == "mock-llm@fixture"


def test_mock_raises_on_missing_bones():
    """If the fixture is missing decisions for any requested bone, mock must
    raise — better than silently passing partial data to the validator."""
    mock = MockLLMClient(_ds(Decision(model_id=1, role="Hips", verdict="keep")))
    with pytest.raises(LLMError, match="missing decisions"):
        mock.classify({
            "donor_id": "x", "target_id": "y",
            "bones": [{"model_id": 1}, {"model_id": 999}],
        })


def test_mock_from_dict():
    fixture = {
        "bones": [
            {"model_id": 1, "role": "Hips", "verdict": "keep"},
            {"model_id": 2, "role": "aux", "verdict": "drop"},
        ],
    }
    mock = MockLLMClient(fixture)
    result = mock.classify({"bones": [{"model_id": 1}, {"model_id": 2}]})
    assert len(result.bones) == 2


def test_mock_from_json_path(tmp_path: Path):
    fixture = tmp_path / "decisions.json"
    fixture.write_text(
        '{"bones": [{"model_id": 1, "role": "Hips", "verdict": "keep"}]}',
        encoding="utf-8",
    )
    mock = MockLLMClient(fixture)
    result = mock.classify({"bones": [{"model_id": 1}]})
    assert result.bones[0].role == "Hips"


def test_mock_custom_model_id():
    mock = MockLLMClient(_ds(), model_id="rigforge-test@v0")
    assert mock.model_id == "rigforge-test@v0"


# --- DispatchMockClient -----------------------------------------------------


def test_dispatch_picks_by_donor_target():
    fixture_a = _ds(Decision(model_id=1, role="Hips", verdict="keep"))
    fixture_b = _ds(Decision(model_id=99, role="aux", verdict="drop"))
    mock = MockLLMClient.from_dispatch({
        ("donor_a", "target_x"): fixture_a,
        ("donor_b", "target_x"): fixture_b,
    })
    result = mock.classify({
        "donor_id": "donor_b", "target_id": "target_x",
        "bones": [{"model_id": 99}],
    })
    assert result == fixture_b


def test_dispatch_raises_on_unknown_pair():
    mock = MockLLMClient.from_dispatch({
        ("a", "b"): _ds(),
    })
    with pytest.raises(LLMError, match="no mock fixture"):
        mock.classify({"donor_id": "x", "target_id": "y", "bones": []})
