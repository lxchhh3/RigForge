"""Tests for the LLM JSON extractor."""
from __future__ import annotations

import json

import pytest

from rigforge.llm.parser import ParseError, parse_decision_set


_VALID_PAYLOAD = {
    "bones": [
        {"model_id": 1, "role": "Hips", "verdict": "keep",
         "drop_category": None, "confidence": 0.95},
        {"model_id": 2, "role": "aux", "verdict": "drop",
         "drop_category": "aux", "confidence": 0.8},
    ]
}


def test_parses_strict_json():
    raw = json.dumps(_VALID_PAYLOAD)
    ds = parse_decision_set(raw)
    assert len(ds.bones) == 2
    assert ds.bones[0].role == "Hips"


def test_parses_fenced_markdown_block():
    raw = "Sure, here you go:\n\n```json\n" + json.dumps(_VALID_PAYLOAD) + "\n```\n"
    ds = parse_decision_set(raw)
    assert ds.bones[1].verdict == "drop"


def test_parses_trailing_prose_via_brace_slice():
    raw = json.dumps(_VALID_PAYLOAD) + "\n\nLet me know if you need adjustments."
    ds = parse_decision_set(raw)
    assert len(ds.bones) == 2


def test_rejects_empty_input():
    with pytest.raises(ParseError, match="empty"):
        parse_decision_set("")


def test_rejects_invalid_shape():
    bad = json.dumps({"wrong_key": "no bones field"})
    with pytest.raises(ParseError):
        parse_decision_set(bad)


def test_rejects_unparseable_garbage():
    with pytest.raises(ParseError):
        parse_decision_set("not json at all, just words")


def test_handles_extra_fields_gracefully():
    """Pydantic ignores unknown fields by default; verdicts still parse."""
    payload = {
        "bones": [
            {"model_id": 1, "role": "Hips", "verdict": "keep",
             "drop_category": None, "confidence": 0.9, "extra_field": "ignored"},
        ],
        "explanation": "I was very thoughtful about this",
    }
    ds = parse_decision_set(json.dumps(payload))
    assert ds.bones[0].model_id == 1
