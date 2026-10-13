"""agents/llm_protocol.py -- the interface every LLM client in this project provides.

The only implementation is agents.kaggle_client.KaggleTransformersClient (local
transformers model, no server, no external API). Used for type hints only.
"""
from typing import Optional, Protocol


class LLMClient(Protocol):
    temperature: float

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str: ...
    def generate_json(self, prompt: str, system: Optional[str] = None) -> dict: ...
    def usage_stats(self) -> dict: ...
    def reset_usage(self) -> None: ...
