"""Robust JSON extraction from LLM responses.

Ollama's `format: "json"` mode is supposed to guarantee well-formed JSON, but
zero-shot base models occasionally wrap output in markdown fences or trailing
prose. This parser tries strict parse first, then a couple of forgiving
fallbacks, then validates against `DecisionSet`.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from rigforge.canonical.decisions import DecisionSet


class ParseError(RuntimeError):
    """Raised when an LLM response can't be coerced into a DecisionSet."""


_FENCED_BLOCK_RE = re.compile(
    r"```(?:json|JSON)?\s*(?P<body>\{.*?\})\s*```", re.DOTALL
)


def parse_decision_set(raw: str) -> DecisionSet:
    """Try strict, then fenced-block, then balanced-brace extraction.

    Validates against DecisionSet. Raises ParseError with a helpful message
    on persistent failure.
    """
    if not raw or not raw.strip():
        raise ParseError("empty LLM response")

    candidates = list(_yield_candidates(raw))
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            last_error = e
            continue
        try:
            return DecisionSet.model_validate(data)
        except ValidationError as e:
            last_error = e
            continue

    head = raw[:200].replace("\n", " ")
    raise ParseError(
        f"could not parse a DecisionSet from LLM output (tried {len(candidates)} "
        f"candidate slices). First 200 chars: {head!r}. Last parse error: {last_error}"
    )


def _yield_candidates(raw: str):
    """Yield text slices likely to be the JSON object, in order of confidence."""
    s = raw.strip()
    # 1. The whole string (Ollama format=json should give us this directly).
    yield s

    # 2. Fenced markdown blocks ```json ... ```
    for m in _FENCED_BLOCK_RE.finditer(s):
        yield m.group("body").strip()

    # 3. Substring from first `{` to last `}` — generous balanced fallback.
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        yield s[first : last + 1]
