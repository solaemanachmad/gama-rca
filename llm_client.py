"""
llm_client.py
==============
Thin, swappable wrapper around a local LLM served by Ollama. Every agent and
the coordinator call `LLMClient.generate(...)` — swapping models means
changing config.LLM_MODEL_NAME only, no code changes elsewhere.

On Kaggle, start the Ollama server in a background cell first:
    !curl -fsSL https://ollama.com/install.sh | sh
    import subprocess, time
    subprocess.Popen(["ollama", "serve"])
    time.sleep(5)
    !ollama pull qwen2.5:7b
"""

import json
from typing import Optional
import requests

import config


class LLMClient:
    def __init__(self, model_name: str = config.LLM_MODEL_NAME,
                 host: str = config.OLLAMA_HOST,
                 temperature: float = config.LLM_TEMPERATURE,
                 max_tokens: int = config.LLM_MAX_TOKENS):
        self.model_name = model_name
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_tokens_used = 0     # cumulative, for cost/efficiency metrics
        self.total_calls = 0

    def generate(self, prompt: str, system: Optional[str] = None,
                 json_mode: bool = False) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=300)
        if resp.status_code == 404:
            raise RuntimeError(
                f"Ollama returned 404 for model '{self.model_name}' — it's very likely "
                f"NOT PULLED on this server yet. Run `ollama list` to see available "
                f"models, then either `ollama pull {self.model_name}` or set "
                f"config.LLM_MODEL_NAME to a model you've already pulled. "
                f"Raw response: {resp.text}"
            )
        resp.raise_for_status()
        data = resp.json()

        self.total_calls += 1
        self.total_tokens_used += data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

        return data.get("response", "").strip()

    def generate_json(self, prompt: str, system: Optional[str] = None) -> dict:
        raw = self.generate(prompt, system=system, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # best-effort recovery: extract the first {...} block
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
        """Call this at the start of each pipeline/baseline run when the
        LLMClient instance is shared across multiple systems/cases (as
        run_experiment.py does for efficiency) -- otherwise total_calls/
        total_tokens reported per system silently accumulate across every
        prior call made with this instance, making cross-system cost
        comparison meaningless."""
        self.total_calls = 0
        self.total_tokens_used = 0
