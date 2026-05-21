"""Anthropic Messages API LLM client.

Drop-in replacement for OllamaLLMClient — same LLMClient protocol, same
prompt builder, same DecisionSet parser. Uses Anthropic's Messages API at
https://api.anthropic.com/v1/messages directly via stdlib urllib so the
project doesn't pick up an `anthropic` package dependency.

This client exists for fast iteration when a hosted Ollama call is slow
or unavailable. The cache-key still includes `model_id`, so switching
between OllamaLLMClient and AnthropicLLMClient invalidates Phase B cache
entries automatically (no stale-decision risk).

API key handling: never echoed, never logged, never put in error messages.
Read from disk once at construction; held in memory thereafter.
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from rigforge.canonical.decisions import DecisionSet
from rigforge.canonical.schema import CanonicalSchema

from .client import LLMError
from .parser import ParseError, parse_decision_set
from .prompts import build_messages


_ANTHROPIC_VERSION = "2023-06-01"
_API_URL = "https://api.anthropic.com/v1/messages"


@dataclass(frozen=True)
class AnthropicConfig:
    """Connection config. `api_key` is the bare secret; `model` is the
    Anthropic model id (e.g. 'claude-opus-4-7'). Defaults chosen to match
    the user's current setup."""
    api_key: str
    model: str = "claude-opus-4-7"
    max_tokens: int = 32000
    timeout_seconds: float = 600.0

    def model_tag_id(self) -> str:
        # Same shape as OllamaConfig.model_tag_id() — used in cache keys.
        return f"anthropic@{self.model}"


def load_anthropic_key(path: Optional[Path] = None) -> str:
    """Read the API key from a file. Default location matches the user's
    'API_KEY_MAKE_VERY_SURE_NO_LEAKAGE.md' workflow file (one key, optionally
    followed by trailing whitespace/newline)."""
    if path is None:
        path = Path("API_KEY_MAKE_VERY_SURE_NO_LEAKAGE.md")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise LLMError(f"empty API key file at {path}")
    # Strip Markdown fencing if present (the file is .md after all)
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith("```"):
            return s
    raise LLMError(f"no API key found in {path}")


class AnthropicLLMClient:
    """Synchronous Anthropic Messages API client.

    Construction
    ------------
    >>> client = AnthropicLLMClient.from_key_file()
    >>> # or with an explicit key + model:
    >>> client = AnthropicLLMClient(AnthropicConfig(api_key=KEY, model="..."))

    `schema` defaults to CanonicalSchema.load_default() — same prompt embed
    as OllamaLLMClient.
    """

    def __init__(
        self,
        config: AnthropicConfig,
        schema: Optional[CanonicalSchema] = None,
        *,
        temperature: float = 0.0,
    ) -> None:
        self._config = config
        self._schema = schema or CanonicalSchema.load_default()
        self._temperature = temperature

    @classmethod
    def from_key_file(
        cls,
        path: Optional[Path] = None,
        model: Optional[str] = None,
        schema: Optional[CanonicalSchema] = None,
    ) -> "AnthropicLLMClient":
        key = load_anthropic_key(path)
        # Use the dataclass default if `model` not specified, so callers don't
        # have to know what it is.
        cfg = (AnthropicConfig(api_key=key) if model is None
               else AnthropicConfig(api_key=key, model=model))
        return cls(config=cfg, schema=schema)

    @property
    def model_id(self) -> str:
        return self._config.model_tag_id()

    def classify(self, request: dict[str, Any]) -> DecisionSet:
        previous_violations = request.get("previous_violations") or None
        messages = build_messages(
            request=request,
            schema=self._schema,
            previous_violations=previous_violations,
        )
        # Anthropic's Messages API puts the system prompt as a top-level
        # field, not as a message with role="system". Split it out.
        system_text = ""
        user_messages: list[dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                user_messages.append({"role": m["role"], "content": m["content"]})

        body = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "system": system_text,
            "messages": user_messages,
        }
        # `temperature` is deprecated for newer Anthropic models (4.x). Only
        # include it when explicitly non-default — older models still accept it.
        if self._temperature != 0.0:
            body["temperature"] = self._temperature
        content = self._post(body)
        try:
            return parse_decision_set(content)
        except ParseError as e:
            raise LLMError(
                f"could not parse Anthropic response into DecisionSet: {e}"
            ) from e

    # --- internals ---------------------------------------------------------

    def _post(self, body: dict[str, Any]) -> str:
        data = json.dumps(body).encode("utf-8")
        # x-api-key header is the documented auth method. Never log it.
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._config.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        req = urllib.request.Request(_API_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            # Sanitize error body — strip x-api-key echoes if any.
            err_body = _scrub_secret(err_body, self._config.api_key)
            raise LLMError(
                f"Anthropic HTTP {e.code} {e.reason}: {err_body[:500]}"
            ) from None
        except (urllib.error.URLError, socket.timeout) as e:
            raise LLMError(f"Anthropic transport failure: {e}") from None
        return _extract_text(payload)


def _extract_text(payload: dict[str, Any]) -> str:
    """Concatenate every text block in the response. Anthropic returns
    `{content: [{type: 'text', text: '...'}, ...]}` — usually one block."""
    blocks = payload.get("content") or []
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text")
            if isinstance(t, str):
                parts.append(t)
    if not parts:
        raise LLMError(
            f"Anthropic response had no text blocks; stop_reason="
            f"{payload.get('stop_reason')!r}"
        )
    return "".join(parts)


def _scrub_secret(text: str, secret: str) -> str:
    """Defensive: redact the API key if it ever shows up in an error body
    so the LLMError that bubbles to the user can never leak it."""
    if not secret:
        return text
    return text.replace(secret, "<REDACTED-API-KEY>")
