"""Ollama (Cloud) LLM client.

POSTs to {host}/api/chat with the bone-classification prompt. Auth via
`Authorization: Bearer <api_key>` from `rigforge/llm/config.py`. Uses stdlib
urllib so the project keeps zero HTTP deps.

Streams the response (`stream: true`) so callers get a live progress signal
instead of staring at silence for the 2–7 minutes a 200-bone classification
takes. A `progress_callback` fires per chunk with running counts; the default
is silent. Token content is accumulated and parsed as one JSON document at
the end via the existing `parse_decision_set`.

The orchestrator's re-prompt-once flow (Phase B step 4) appends
`previous_violations` to its request dict; this client picks that up and
passes it to the prompt builder.

WHEN THE LORA LANDS: drop the schema embedding in the prompt and update
`OllamaConfig.model` to the new tag. No client-side code changes needed.
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from rigforge.canonical.decisions import DecisionSet
from rigforge.canonical.schema import CanonicalSchema

from .client import LLMError
from .config import OllamaConfig, load_config
from .parser import ParseError, parse_decision_set
from .prompts import build_messages


@dataclass(frozen=True)
class StreamProgress:
    """Cumulative stats fired to the progress callback as chunks arrive."""
    chunks: int
    content_bytes: int
    elapsed_seconds: float
    done: bool


ProgressCallback = Callable[[StreamProgress], None]


class OllamaLLMClient:
    """Synchronous Ollama Cloud / local chat client, streaming.

    Construction
    ------------
    >>> client = OllamaLLMClient.from_config_file()
    >>> # or with an explicit config + schema:
    >>> client = OllamaLLMClient(config=cfg, schema=schema)
    >>> # with a live progress callback:
    >>> client = OllamaLLMClient(cfg, progress_callback=lambda p: print(p))

    `schema` defaults to CanonicalSchema.load_default() — the prompt builder
    embeds the schema text into every call, so the model knows what roles to
    emit even without LoRA.
    """

    def __init__(
        self,
        config: OllamaConfig,
        schema: Optional[CanonicalSchema] = None,
        *,
        temperature: float = 0.0,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self._config = config
        self._schema = schema or CanonicalSchema.load_default()
        self._temperature = temperature
        self._progress_callback = progress_callback

    @classmethod
    def from_config_file(
        cls,
        path: Optional[Path] = None,
        schema: Optional[CanonicalSchema] = None,
        **kwargs: Any,
    ) -> "OllamaLLMClient":
        return cls(config=load_config(path), schema=schema, **kwargs)

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
        body = self._build_body(messages)
        content = self._post_stream(body)
        try:
            return parse_decision_set(content)
        except ParseError as e:
            raise LLMError(
                f"could not parse Ollama response into DecisionSet: {e}"
            ) from e

    # --- internals ---------------------------------------------------------

    def _build_body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model": self._config.model,
            "messages": messages,
            "stream": True,
            "format": self._config.request_format,
            "options": {"temperature": self._temperature},
        }

    def _post_stream(self, body: dict[str, Any]) -> str:
        """POST and stream the NDJSON response, accumulating message content."""
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **self._config.auth_header(),
        }
        req = urllib.request.Request(
            self._config.chat_url(),
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout_seconds) as resp:
                return self._consume_stream(resp)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise LLMError(
                f"Ollama HTTP {e.code} {e.reason} from {self._config.chat_url()}: {err_body}"
            ) from e
        except (urllib.error.URLError, socket.timeout) as e:
            raise LLMError(
                f"Ollama transport failure to {self._config.chat_url()}: {e}"
            ) from e

    def _consume_stream(self, resp: Any) -> str:
        """Read NDJSON chunks from `resp`. Accumulate message.content into a
        single string. Fire the progress callback once per chunk.
        """
        chunks: list[str] = []
        bytes_seen = 0
        chunk_count = 0
        t0 = time.time()
        done = False
        last_error: Optional[str] = None

        # urllib's HTTPResponse is iterable line-by-line. Each line is a
        # complete JSON object in NDJSON streaming mode.
        for line in resp:
            if not line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Some servers wrap stream lines in `data: ` SSE prefixes —
                # support that gracefully.
                if line.startswith(b"data: "):
                    try:
                        obj = json.loads(line[6:])
                    except json.JSONDecodeError:
                        last_error = f"non-JSON stream line: {line[:120]!r}"
                        continue
                else:
                    last_error = f"non-JSON stream line: {line[:120]!r}"
                    continue
            if "error" in obj:
                raise LLMError(f"Ollama stream error: {obj['error']}")
            msg = obj.get("message") or {}
            piece = msg.get("content", "")
            if piece:
                chunks.append(piece)
                bytes_seen += len(piece)
            chunk_count += 1
            done = bool(obj.get("done"))
            if self._progress_callback is not None:
                self._progress_callback(StreamProgress(
                    chunks=chunk_count,
                    content_bytes=bytes_seen,
                    elapsed_seconds=time.time() - t0,
                    done=done,
                ))
            if done:
                break

        if not chunks:
            detail = f" ({last_error})" if last_error else ""
            raise LLMError(
                f"Ollama stream finished without any content"
                f" after {chunk_count} chunk(s){detail}"
            )
        return "".join(chunks)
