"""
kaggle_client.py
==================
LLM client backed by a model pulled either from Kaggle Models (kagglehub) or
directly from the Hugging Face Hub, run via transformers -- no Ollama
server, no external API. Meant for running INSIDE a Kaggle Notebook with a
free GPU attached. Same generate()/generate_json()/usage_stats() interface
as LLMClient and GeminiClient, so agents.py / pipeline.py need zero changes
-- just pick this backend via config.

Two model families are supported, auto-detected from the handle/path:
  1. Gemma 4's "Unified" architecture (encoder-free multimodal: text/audio/
     image/video share one model) -- requires AutoProcessor, not
     AutoTokenizer, per the official model card, even for text-only use.
  2. Everything else (Qwen2.5, Llama-3.1, Phi, Mistral, etc.) -- standard
     AutoTokenizer + chat template flow.

Handle formats:
  - Kaggle-catalog handle (org/model/framework/variant, e.g.
    "google/gemma-4/transformers/gemma-4-12b-it") -- downloaded via
    kagglehub, requires accepting the model's license on its Kaggle page
    once (logged in) before the first download.
  - Plain Hugging Face Hub repo id (org/model, e.g.
    "Qwen/Qwen2.5-7B-Instruct") -- downloaded directly by transformers, no
    kagglehub involved. Preferred default here for reproducibility: no
    Kaggle-specific license click needed for open weights, and the exact
    same handle works identically outside Kaggle (local machine, other
    notebook platforms), which matters for other people re-running this
    code.

Setup (inside a Kaggle Notebook):
    pip install -U kagglehub transformers torch accelerate --quiet
    # bitsandbytes only needed if COORDINATOR_USE_4BIT=1:
    pip install -U bitsandbytes --quiet
    # kagglehub is auto-authenticated inside Kaggle Notebooks; a plain HF
    # repo id needs no auth at all for open (non-gated) models.

PREFERRED on Kaggle for a Kaggle-catalog handle: attach the model via the
notebook's "Add Input" > Models panel instead of downloading it -- mounts
it locally under /kaggle/input/models/... with no network dependency, far
more reliable than kagglehub.model_download() repeatedly hitting the
network. Once attached, point directly at the mounted path:
    KAGGLE_MODEL_LOCAL_PATH=/kaggle/input/models/google/gemma-4/transformers/gemma-4-12b-it/2

USAGE (config.py / .env):
    LLM_BACKEND=kaggle
    KAGGLE_MODEL_HANDLE=Qwen/Qwen2.5-7B-Instruct   # default -- see config/__init__.py
"""

import json
import os
from typing import Optional

import config


def _is_gemma4_unified(identifier: Optional[str]) -> bool:
    identifier = (identifier or "").lower()
    return "gemma-4" in identifier or "gemma4" in identifier


class KaggleTransformersClient:
    def __init__(self, model_handle: str = None, local_path: str = None,
                 temperature: float = config.LLM_TEMPERATURE,
                 max_tokens: int = config.LLM_MAX_TOKENS,
                 enable_thinking: bool = False,
                 use_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM

        model_handle = model_handle or config.KAGGLE_MODEL_HANDLE
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
            arch_hint = local_path
        elif model_handle.count("/") >= 3:
            # Kaggle-catalog handle format: org/model/framework/variant.
            import kagglehub
            model_path = kagglehub.model_download(model_handle)
            arch_hint = model_handle
        else:
            # Plain Hugging Face Hub repo id -- transformers downloads and
            # caches it directly (needs internet enabled on the notebook;
            # HF_TOKEN env var only required for gated models).
            model_path = model_handle
            arch_hint = model_handle

        self._is_gemma4_unified = _is_gemma4_unified(arch_hint)

        quant_kwargs = {}
        if use_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        if self._is_gemma4_unified:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.tokenizer = None
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.processor = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", **quant_kwargs
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        # enable_thinking=False by default: our downstream code parses raw
        # JSON out of the response (generate_json below) -- chain-of-thought
        # text mixed into the output would break that parsing. Only used on
        # the Gemma-4-Unified (AutoProcessor) path; standard AutoTokenizer
        # chat templates (Qwen2.5, Llama-3.1, etc.) don't take this kwarg.
        self.enable_thinking = enable_thinking
        self.total_tokens_used = 0
        self.total_calls = 0

    def generate(self, prompt: str, system: Optional[str] = None,
                 json_mode: bool = False) -> str:
        import torch
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if self.processor is not None:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        else:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        # Guard against GPU OOM on very long prompts (attention memory grows with
        # the square of the length with the default SDPA path; a ~16k-token
        # Coordinator prompt needed a 13 GiB allocation on a 15 GB T4). Keep the
        # head (instructions) and tail (task + schema), drop the middle.
        max_in = int(os.environ.get("MAX_INPUT_TOKENS", "9000"))
        if input_len > max_in and self.tokenizer is not None:
            ids = inputs["input_ids"][0]
            head, tail = int(max_in * 0.55), max_in - int(max_in * 0.55)
            ids = torch.cat([ids[:head], ids[-tail:]]).unsqueeze(0)
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            self.truncated_prompts = getattr(self, "truncated_prompts", 0) + 1
            input_len = ids.shape[-1]

        gen_kwargs = dict(
            max_new_tokens=self.max_tokens,
            temperature=max(self.temperature, 1e-4),  # 0 breaks some sampling configs
            do_sample=self.temperature > 0,
        )
        try:
            outputs = self.model.generate(**inputs, **gen_kwargs)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            ids = inputs["input_ids"][0]
            keep = max(2000, ids.shape[-1] // 2)
            ids = torch.cat([ids[:keep // 2], ids[-(keep - keep // 2):]]).unsqueeze(0)
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            self.oom_retries = getattr(self, "oom_retries", 0) + 1
            input_len = ids.shape[-1]
            outputs = self.model.generate(**inputs, **gen_kwargs)

        self.total_calls += 1
        new_tokens = outputs[0][input_len:]
        self.total_tokens_used += input_len + len(new_tokens)

        if self.processor is not None:
            response = self.processor.decode(new_tokens, skip_special_tokens=True)
        else:
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
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
