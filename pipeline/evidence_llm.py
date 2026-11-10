"""Compatibility wrapper: Evidence->LLM baseline has moved to baselines/evidence_llm.py."""
from baselines.evidence_llm import evidence_llm, build_prompt, EVIDENCE_SYSTEM

__all__ = ["evidence_llm", "build_prompt", "EVIDENCE_SYSTEM"]
