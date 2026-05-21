"""Tests for the real OllamaLLMClient — HTTP layer mocked.

These DO NOT hit ollama.com. They mock `urllib.request.urlopen` so we exercise
prompt construction, auth header, body shape, response parsing, and error
mapping without network or API key requirements.

For a real-API smoke run, see training/smoke_ollama.py (manual, not pytest).
"""
from __future__ import annotations

import io
import json
import urllib.error
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from rigforge.canonical.schema import CanonicalSchema
from rigforge.llm.client import LLMError
from rigforge.llm.config import ConfigError, OllamaConfig, load_config
from rigforge.llm.ollama import OllamaLLMClient


def _make_cfg(api_key: str = "test-key-not-real") -> OllamaConfig:
    return OllamaConfig(
        host="https://ollama.example",
        model="ds-v4-flash",
        api_key=api_key,
        timeout_seconds=5.0,
    )


def _good_response(decisions: dict) -> bytes:
    """Bytes payload that simulates a successful Ollama /api/chat streamed
    response. Streaming mode emits NDJSON — one JSON object per line. For a
    single-chunk fixture we emit two lines: the full content, then a done
    marker."""
    content = json.dumps(decisions)
    line1 = json.dumps({
        "model": "ds-v4-flash",
        "message": {"role": "assistant", "content": content},
        "done": False,
    }) + "\n"
    line2 = json.dumps({
        "model": "ds-v4-flash",
        "message": {"role": "assistant", "content": ""},
        "done": True,
    }) + "\n"
    return (line1 + line2).encode("utf-8")


def _multi_chunk_response(content: str, n_chunks: int = 4) -> bytes:
    """NDJSON stream that splits `content` across n_chunks token messages,
    plus a final done marker."""
    parts = []
    chunk_size = max(1, len(content) // n_chunks)
    for i in range(0, len(content), chunk_size):
        parts.append(json.dumps({
            "model": "ds-v4-flash",
            "message": {"role": "assistant", "content": content[i:i + chunk_size]},
            "done": False,
        }) + "\n")
    parts.append(json.dumps({
        "model": "ds-v4-flash",
        "message": {"role": "assistant", "content": ""},
        "done": True,
    }) + "\n")
    return "".join(parts).encode("utf-8")


class _FakeResponse:
    """Iterable response — yields one NDJSON line per `__next__` call so the
    streaming reader gets multi-chunk behaviour even for short fixtures."""
    def __init__(self, body: bytes) -> None:
        self._buf = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return self

    def __next__(self):
        line = self._buf.readline()
        if not line:
            raise StopIteration
        return line

    def read(self) -> bytes:
        return self._buf.read()


@contextmanager
def _capture_request(response_body: bytes):
    """Patch urlopen to return `response_body` and capture the outgoing Request."""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8")) if req.data else None
        captured["timeout"] = timeout
        return _FakeResponse(response_body)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        yield captured


# --- config ---------------------------------------------------------------


def test_config_load_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.json")


def test_config_load_placeholder_key_raises(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({
        "host": "https://ollama.com",
        "model": "x",
        "api_key": "REPLACE_WITH_YOUR_OLLAMA_API_KEY",
    }))
    with pytest.raises(ConfigError, match="placeholder"):
        load_config(cfg)


def test_config_load_valid(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({
        "host": "https://ollama.com",
        "model": "x",
        "api_key": "real-key",
    }))
    parsed = load_config(cfg)
    assert parsed.model == "x"
    assert parsed.chat_url() == "https://ollama.com/api/chat"


# --- client wiring --------------------------------------------------------


def test_classify_posts_to_chat_url_with_bearer_auth():
    schema = CanonicalSchema.load_default()
    client = OllamaLLMClient(_make_cfg("abc123"), schema=schema)
    payload = {
        "bones": [
            {"model_id": 1, "role": "Hips", "verdict": "keep",
             "drop_category": None, "confidence": 0.9},
        ]
    }
    request = {"donor_id": "d", "target_id": "t",
               "bones": [{"model_id": 1, "name": "Hips", "parent_name": None,
                          "translation_xyz": [0, 0.9, 0], "child_names": [],
                          "has_skin_cluster": True, "cluster_weight_count": 100}]}

    with _capture_request(_good_response(payload)) as captured:
        ds = client.classify(request)

    assert captured["url"] == "https://ollama.example/api/chat"
    # Headers are normalized to title-case by urllib; check both
    auth = captured["headers"].get("Authorization") or captured["headers"].get("authorization")
    assert auth == "Bearer abc123"
    body = captured["body"]
    assert body["model"] == "ds-v4-flash"
    assert body["stream"] is True
    assert body["format"] == "json"
    assert body["options"]["temperature"] == 0.0
    # Two messages: system + user
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"
    # DecisionSet round-trip
    assert len(ds.bones) == 1
    assert ds.bones[0].role == "Hips"


def test_classify_passes_violations_as_third_message():
    """Re-prompt scenario from phase_b: violations get appended as user msg."""
    schema = CanonicalSchema.load_default()
    client = OllamaLLMClient(_make_cfg(), schema=schema)
    response = _good_response({"bones": []})
    request = {
        "donor_id": "d", "target_id": "t",
        "bones": [],
        "previous_violations": ["unique_role: Hips assigned twice"],
    }
    with _capture_request(response) as captured:
        client.classify(request)
    msgs = captured["body"]["messages"]
    assert len(msgs) == 3
    assert "Hips assigned twice" in msgs[2]["content"]


def test_classify_http_error_maps_to_llm_error():
    schema = CanonicalSchema.load_default()
    client = OllamaLLMClient(_make_cfg(), schema=schema)

    def raise_http(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad key")
        )
    with patch("urllib.request.urlopen", side_effect=raise_http):
        with pytest.raises(LLMError, match="401"):
            client.classify({"bones": [], "donor_id": "d", "target_id": "t"})


def test_classify_transport_failure_maps_to_llm_error():
    schema = CanonicalSchema.load_default()
    client = OllamaLLMClient(_make_cfg(), schema=schema)

    def raise_url(req, timeout=None):
        raise urllib.error.URLError("connection refused")
    with patch("urllib.request.urlopen", side_effect=raise_url):
        with pytest.raises(LLMError, match="transport"):
            client.classify({"bones": [], "donor_id": "d", "target_id": "t"})


def test_classify_non_json_stream_raises():
    """Stream lines that aren't JSON yield zero content → LLMError."""
    schema = CanonicalSchema.load_default()
    client = OllamaLLMClient(_make_cfg(), schema=schema)
    with _capture_request(b"not json\nmore not json\n"):
        with pytest.raises(LLMError, match="without any content"):
            client.classify({"bones": [], "donor_id": "d", "target_id": "t"})


def test_classify_stream_done_with_no_content_raises():
    """Stream that closes without ever emitting message.content → LLMError."""
    schema = CanonicalSchema.load_default()
    client = OllamaLLMClient(_make_cfg(), schema=schema)
    bad_body = (json.dumps({"done": True}) + "\n").encode("utf-8")
    with _capture_request(bad_body):
        with pytest.raises(LLMError, match="without any content"):
            client.classify({"bones": [], "donor_id": "d", "target_id": "t"})


def test_classify_stream_error_chunk_raises():
    """Ollama can emit `{\"error\": \"...\"}` mid-stream — surface it loudly."""
    schema = CanonicalSchema.load_default()
    client = OllamaLLMClient(_make_cfg(), schema=schema)
    body = (json.dumps({"error": "model not found"}) + "\n").encode("utf-8")
    with _capture_request(body):
        with pytest.raises(LLMError, match="stream error"):
            client.classify({"bones": [], "donor_id": "d", "target_id": "t"})


def test_classify_unparseable_content_raises():
    """Stream produced content but it isn't a valid DecisionSet JSON."""
    schema = CanonicalSchema.load_default()
    client = OllamaLLMClient(_make_cfg(), schema=schema)
    body = _multi_chunk_response("just words, no json")
    with _capture_request(body):
        with pytest.raises(LLMError, match="could not parse"):
            client.classify({"bones": [], "donor_id": "d", "target_id": "t"})


def test_classify_accumulates_multi_chunk_stream():
    """Streamed JSON delivered across many chunks parses cleanly once joined."""
    schema = CanonicalSchema.load_default()
    client = OllamaLLMClient(_make_cfg(), schema=schema)
    decisions = {"bones": [{"model_id": 1, "role": "Hips", "verdict": "keep",
                             "drop_category": None, "confidence": 0.9}]}
    body = _multi_chunk_response(json.dumps(decisions), n_chunks=8)
    with _capture_request(body):
        ds = client.classify({"bones": [{"model_id": 1, "name": "Hips",
                                          "parent_name": None,
                                          "translation_xyz": [0, 0, 0],
                                          "child_names": [],
                                          "has_skin_cluster": True,
                                          "cluster_weight_count": 100}],
                              "donor_id": "d", "target_id": "t"})
    assert ds.bones[0].role == "Hips"


def test_progress_callback_fires_for_each_chunk():
    schema = CanonicalSchema.load_default()
    events = []
    client = OllamaLLMClient(_make_cfg(), schema=schema,
                             progress_callback=lambda p: events.append(p))
    decisions = {"bones": []}
    body = _multi_chunk_response(json.dumps(decisions), n_chunks=5)
    with _capture_request(body):
        client.classify({"bones": [], "donor_id": "d", "target_id": "t"})
    # At least one chunk per content piece + the final done marker
    assert len(events) >= 2
    assert events[-1].done is True
    # Monotonically non-decreasing bytes/chunk count
    assert events[0].chunks <= events[-1].chunks


def test_model_id_is_stable():
    client = OllamaLLMClient(_make_cfg(), schema=CanonicalSchema.load_default())
    assert client.model_id == "ollama:ds-v4-flash@https://ollama.example"
