"""
kaggle_client.py
==================
LLM client backed by a model pulled from Kaggle Models (kagglehub) and run
directly via transformers -- no Ollama server, no external API. Meant for
running INSIDE a Kaggle Notebook with a free GPU attached, where kagglehub
is pre-authenticated automatically. Same generate()/generate_json()/
usage_stats() interface as LLMClient and GeminiClient, so agents.py /
pipeline.py need zero changes -- just pick this backend via config.

Uses AutoProcessor (not AutoTokenizer) -- required for Gemma 4's "Unified"
architecture (encoder-free multimodal: text/audio/image/video share one
model), per the official model card. A plain AutoTokenizer flow will not
work correctly with this architecture even for text-only use.

Setup (inside a Kaggle Notebook):
    pip install -U kagglehub transformers torch accelerate --quiet
    # kagglehub is auto-authenticated inside Kaggle Notebooks.
    # Running locally instead of on Kaggle: set KAGGLE_USERNAME / KAGGLE_KEY
    # (from kaggle.com -> Account -> Create New Token, downloads kaggle.json)

Model access: some Kaggle-hosted models (e.g. Gemma) require accepting the
license on the model's Kaggle page once (logged in) before kagglehub can
download it -- you'll get a clear permission error the first time if you
haven't.

USAGE:
    # config.py / .env:
    LLM_BACKEND=kaggle
    KAGGLE_MODEL_HANDLE=google/gemma-4/transformers/gemma-4-12b-it

    # PREFERRED on Kaggle: attach the model via the notebook's "Add Input" >
    # Models panel instead of downloading it -- this mounts it locally under
    # /kaggle/input/models/... with no network dependency, which is far more
    # reliable than kagglehub.model_download() repeatedly hitting the network.
    # Once attached, point directly at the mounted path:
    KAGGLE_MODEL_LOCAL_PATH=/kaggle/input/models/google/gemma-4/transformers/gemma-4-12b-it/2
"""

import json
import os
from typing import Optional

import config


class KaggleTransformersClient:
    def __init__(self, model_handle: str = None, local_path: str = None,
                 temperature: float = config.LLM_TEMPERATURE,
                 max_tokens: int = config.LLM_MAX_TOKENS,
                 enable_thinking: bool = False):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        local_path = local_path or config.KAGGLE_MODEL_LOCAL_PATH
        if local_path:
            # Model already mounted locally (Add Input > Models in the
            # notebook UI) -- use it directly, no network call at all.
            if not os.path.isdir(local_path):
                raise RuntimeError(
                    f"KAGGLE_MODEL_LOCAL_PATH={local_path!r} does not exist. "
                    f"Check the exact mounted path shown in the notebook's Input "
                    f"panel (right sidebar) -- it usually ends in a version "
                    f"number, e.g. '.../gemma-4-12b-it/2'."
                )
            model_path = local_path
        else:
            # Falls back to downloading via kagglehub (needs internet + the
            # model license accepted on kaggle.com once).
            import kagglehub
            self.model_handle = model_handle or config.KAGGLE_MODEL_HANDLE
            model_path = kagglehub.model_download(self.model_handle)

        # AutoProcessor, not AutoTokenizer -- Gemma 4's "Unified" encoder-free
        # multimodal architecture requires it even for text-only prompts.
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        # enable_thinking=False by default: our downstream code parses raw
        # JSON out of the response (generate_json below) -- chain-of-thought
        # text mixed into the output would break that parsing. Flip this on
        # only if you also update generate_json() to call
        # self.processor.parse_response() and extract the final answer
        # segment from it first.
        self.enable_thinking = enable_thinking
        self.total_tokens_used = 0
        self.total_calls = 0

    def generate(self, prompt: str, system: Optional[str] = None,
                 json_mode: bool = False) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=max(self.temperature, 1e-4),  # 0 breaks some sampling configs
            do_sample=self.temperature > 0,
        )

        self.total_calls += 1
        new_tokens = outputs[0][input_len:]
        self.total_tokens_used += input_len + len(new_tokens)

        response = self.processor.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

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