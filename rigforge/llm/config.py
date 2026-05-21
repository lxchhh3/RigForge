"""Ollama client configuration.

Lives in `data/ollama.json` (gitignored) so the API key never enters source
control. A committed `data/ollama.example.json` shows the schema.

Loader picks the file in this order:
  1. explicit path argument
  2. RIGFORGE_OLLAMA_CONFIG environment variable
  3. {repo_root}/data/ollama.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "ollama.json"
ENV_CONFIG_PATH = "RIGFORGE_OLLAMA_CONFIG"


class OllamaConfig(BaseModel):
    """Schema of data/ollama.json."""
    host: str = "https://ollama.com"
    model: str
    api_key: str
    timeout_seconds: float = 120.0
    request_format: str = "json"   # passed to Ollama as the `format` field

    def chat_url(self) -> str:
        host = self.host.rstrip("/")
        return f"{host}/api/chat"

    def auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def model_tag_id(self) -> str:
        """Stable identifier folded into the cache key — host + tag, no key."""
        return f"ollama:{self.model}@{self.host}"


class ConfigError(RuntimeError):
    pass


def load_config(path: Optional[Path] = None) -> OllamaConfig:
    """Load and validate the Ollama config. Raises ConfigError on missing
    files or missing required fields (api_key in particular)."""
    chosen = _resolve_path(path)
    if not chosen.exists():
        raise ConfigError(
            f"Ollama config not found at {chosen}. "
            f"Copy data/ollama.example.json to data/ollama.json and fill in your api_key."
        )
    try:
        raw = json.loads(chosen.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"invalid JSON in {chosen}: {e}") from e
    try:
        cfg = OllamaConfig.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"config shape mismatch in {chosen}: {e}") from e
    if cfg.api_key.startswith("REPLACE_") or not cfg.api_key.strip():
        raise ConfigError(
            f"api_key in {chosen} is the placeholder — fill in a real Ollama API key."
        )
    return cfg


def _resolve_path(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(ENV_CONFIG_PATH)
    if env:
        return Path(env)
    return _DEFAULT_CONFIG_PATH
