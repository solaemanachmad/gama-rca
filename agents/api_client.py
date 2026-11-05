"""OpenAI-compatible chat API client (v5 cross-model study: one large API model, PREREGISTRATION_V5.md F4).

Same interface as agents/kaggle_client.KaggleTransformersClient (generate / generate_json / usage_stats / reset_usage), so every
prompt, parser and stage is unchanged. Selected with LLM_BACKEND=api; provider and model from the environment:

  API_BASE_URL          e.g. https://api.groq.com/openai/v1  |  https://api.cerebras.ai/v1
  API_MODEL             e.g. openai/gpt-oss-120b (Groq)       |  gpt-oss-120b (Cerebras)
  API_KEY               the key (set by scripts/kaggle_job.py from a Kaggle Secret; never logged, never written to a file)
  API_REASONING_EFFORT  low | medium | high (reasoning models; fixed before the run), empty = not sent
  API_MIN_INTERVAL_S    minimum seconds between requests (free tiers: requests per minute), default 0
  LLM_TEMPERATURE / LLM_SEED / LLM_MAX_TOKENS as for the local model (reasoning tokens count against LLM_MAX_TOKENS)

Data: the provider's data controls must be set to no retention / no training before the run (paper, data ethics). Requests are
retried on 429 / 5xx / network errors with the provider's retry-after (else exponential back-off); any other error raises, and the
calling stage records it as a failed call (as a parse failure of the local model would be).
"""
from __future__ import annotations

import os
import time
from typing import Optional

import config

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class ApiChatClient:
    def __init__(self, base_url: str = None, model: str = None, api_key: str = None, temperature: float = config.LLM_TEMPERATURE,
                 max_tokens: int = config.LLM_MAX_TOKENS, reasoning_effort: str = None, min_interval_s: float = None,
                 max_retries: int = 10, timeout_s: float = 300.0, session=None):
        self.base_url = (base_url or os.environ.get("API_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("API_MODEL", "")
        self._key = api_key or os.environ.get("API_KEY", "")
        assert self.base_url and self.model, "API_BASE_URL and API_MODEL must be set for LLM_BACKEND=api"
        assert self._key, "API_KEY is empty (Kaggle: add the provider key as a Secret; scripts/kaggle_job.py passes it)"
        self.temperature, self.max_tokens = temperature, max_tokens
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else os.environ.get("API_REASONING_EFFORT", "")
        self.min_interval_s = float(min_interval_s if min_interval_s is not None else os.environ.get("API_MIN_INTERVAL_S", "0"))
        self.max_retries, self.timeout_s = max_retries, timeout_s
        if session is None:
            import requests
            session = requests.Session()
        self._s = session
        self._last = 0.0
        self.tokenizer = None
        self.reset_usage()

    def __repr__(self):                                     # never shows the key
        return f"ApiChatClient({self.base_url}, {self.model}, effort={self.reasoning_effort or '-'})"

    def _post(self, body: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self._s.post(url, json=body, headers=headers, timeout=self.timeout_s)
            except Exception as e:                          # network error: retry
                if attempt == self.max_retries:
                    raise RuntimeError(f"API request failed after {attempt + 1} attempts: {type(e).__name__}") from None
                self.retries += 1
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                self.retries += 1
                if r.status_code == 429:
                    self.rate_limited += 1
                try:
                    ra = float(r.headers.get("retry-after", ""))
                except ValueError:
                    ra = delay
                time.sleep(min(max(ra, 1.0), 300.0))
                delay = min(delay * 2, 120)
                continue
            msg = (r.text or "")[:300]
            raise RuntimeError(f"API error {r.status_code}: {msg}")
        raise RuntimeError("API: retries exhausted")

    def generate(self, prompt: str, system: Optional[str] = None, json_mode: bool = False) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": messages, "temperature": self.temperature, "seed": config.LLM_SEED,
                "max_completion_tokens": self.max_tokens}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        out = self._post(body)
        ch = (out.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        u = out.get("usage") or {}
        pin, pout = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        rt = int(((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
        self.total_calls += 1
        self.total_tokens_used += pin + pout
        self.input_tokens += pin
        self.output_tokens += pout
        self.reasoning_tokens += rt
        self.max_input_tokens = max(self.max_input_tokens, pin)
        if ch.get("finish_reason") == "length":
            self.thinking_truncated += 1                   # the token budget ended before (or inside) the answer
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        self.thinking_chars += len(reasoning)
        return (msg.get("content") or "").strip()

    def generate_json(self, prompt: str, system: Optional[str] = None) -> dict:
        raw = self.generate(prompt, system=system, json_mode=True)
        from agents.json_utils import parse_json_lenient
        return parse_json_lenient(raw)

    def score_options(self, prompt, labels, system=None):
        return None                                         # no logits through the API (callers fall back, as for the processor path)

    def usage_stats(self) -> dict:
        return {"total_calls": self.total_calls, "total_tokens": self.total_tokens_used, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "reasoning_tokens": self.reasoning_tokens, "max_input_tokens": self.max_input_tokens,
                "truncated_prompts": 0, "thinking_chars": self.thinking_chars, "thinking_truncated": self.thinking_truncated,
                "api_retries": self.retries, "api_rate_limited": self.rate_limited}

    def reset_usage(self) -> None:
        self.total_calls = self.total_tokens_used = self.input_tokens = self.output_tokens = 0
        self.reasoning_tokens = self.max_input_tokens = self.thinking_chars = self.thinking_truncated = 0
        self.retries = self.rate_limited = 0
