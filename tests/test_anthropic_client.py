"""Tests for the Anthropic LLM client config.

These pin the *default* model so the dev launcher (`launch.bat`, which uses
bare `--llm-anthropic` with no model override) gets the model we expect.
"""
from __future__ import annotations

from rigforge.llm.anthropic import AnthropicConfig, AnthropicLLMClient


def test_default_model_is_sonnet():
    """Default Anthropic model is Sonnet — chosen for fast output when the
    primary (Ollama) provider is unavailable."""
    assert AnthropicConfig(api_key="x").model == "claude-sonnet-4-6"


def test_model_tag_id_reflects_default_sonnet():
    cfg = AnthropicConfig(api_key="x")
    assert cfg.model_tag_id() == "anthropic@claude-sonnet-4-6"


def test_client_model_id_defaults_to_sonnet():
    client = AnthropicLLMClient(AnthropicConfig(api_key="x"))
    assert client.model_id == "anthropic@claude-sonnet-4-6"


def test_explicit_model_still_overridable():
    """Changing the default must not break explicit model selection (e.g.
    Opus for a higher-quality pass)."""
    cfg = AnthropicConfig(api_key="x", model="claude-opus-4-8")
    assert cfg.model_tag_id() == "anthropic@claude-opus-4-8"
