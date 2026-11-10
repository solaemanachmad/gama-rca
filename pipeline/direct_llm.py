"""Compatibility wrapper: Direct LLM baseline has moved to baselines/direct_llm.py."""
from baselines.direct_llm import direct_llm, DIRECT_SYSTEM_PROMPT, FINAL_SCHEMA_HINT

__all__ = ["direct_llm", "DIRECT_SYSTEM_PROMPT", "FINAL_SCHEMA_HINT"]
