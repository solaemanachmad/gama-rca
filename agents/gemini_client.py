"""
gemini_client.py
==================
Drop-in replacement for llm_client.LLMClient, backed by the Gemini API
instead of local Ollama. Same generate()/generate_json()/usage_stats()
interface, so agents.py and pipeline.py need ZERO changes -- just swap
which client class gets instantiated (see USAGE below).

Why: local Ollama inference on CPU is slow per-call (seconds per agent call,
x5 agents x N cases), and ties up memory alongside the embedding model.
Gemini API offloads inference to Google's infra -- much faster wall-clock
per call, at the cost of: (1) needing an API key + internet, (2) real
per-token cost (check current pricing), (3) losing the "fully local /
reproducible offline" property Ollama gave you for the paper's methodology
section. Mention this trade-off explicitly if you switch for final results.

Setup:
    pip install google-genai --break-system-packages
    export GEMINI_API_KEY=your_key_here

USAGE (in pipeline.py / baselines.py, wherever LLMClient() is constructed):
    from gemini_client import GeminiClient
    llm = GeminiClient(model_name="gemini-3.1-flash-lite")
    # everywhere else stays identical: llm.generate(...), llm.generate_json(...)
"""

import json
import os
from typing import Optional

from google import genai
from google.genai import types

import config


class GeminiClient:
    def __init__(self, model_name: str = "gemini-3.1-flash-lite",
                 api_key: Optional[str] = None,
                 temperature: float = config.LLM_TEMPERATURE,
                 max_tokens: int = config.LLM_MAX_TOKENS):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found. Set the GEMINI_API_KEY environment "
                "variable, or pass api_key= explicitly."
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_tokens_used = 0
        self.total_calls = 0

    def generate(self, prompt: str, system: Optional[str] = None,
                 json_mode: bool = False) -> str:
        gen_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=system or None,
            response_mime_type="application/json" if json_mode else "text/plain",
            # minimal thinking: RCA agent prompts are extraction/classification-
            # shaped (pick entity, pick fault type), not open-ended reasoning --
            # "high" thinking mainly adds latency/cost here for little accuracy
            # gain. Bump to "high" specifically for the Coordinator if you find
            # it under-reasoning on the causal chain.
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        )

        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt, config=gen_config,
        )

        self.total_calls += 1
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self.total_tokens_used += (getattr(usage, "prompt_token_count", 0) or 0)
            self.total_tokens_used += (getattr(usage, "candidates_token_count", 0) or 0)

        return (response.text or "").strip()

    def generate_json(self, prompt: str, system: Optional[str] = None) -> dict:
        raw = self.generate(prompt, system=system, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(raw[start:end + 1])
                except json.JSONDecodeError:
                    pass
            return {"_parse_error": True, "_raw": raw}

    def usage_stats(self) -> dict:
        return {"total_calls": self.total_calls, "total_tokens": self.total_tokens_used}

    def reset_usage(self) -> None:
        self.total_calls = 0
        self.total_tokens_used = 0