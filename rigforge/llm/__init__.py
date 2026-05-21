"""LLM module — Rail L abstraction + concrete clients."""

from .client import LLMClient, LLMError
from .config import ConfigError, OllamaConfig, load_config
from .mock import DispatchMockClient, MockLLMClient
from .ollama import OllamaLLMClient

__all__ = [
    "LLMClient",
    "LLMError",
    "MockLLMClient",
    "DispatchMockClient",
    "OllamaLLMClient",
    "OllamaConfig",
    "ConfigError",
    "load_config",
]
