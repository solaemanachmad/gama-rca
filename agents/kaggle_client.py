"""
kaggle_client.py
==================
LLM client backed by a model pulled either from Kaggle Models (kagglehub) or
directly from the Hugging Face Hub, run via transformers -- no Ollama
server, no external API. Meant for running INSIDE a Kaggle Notebook with a
free GPU attached. Same generate()/generate_json()/usage_stats() interface
as agents.llm_protocol.LLMClient.

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
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, config.KAGGLE_DTYPE))

        if self._is_gemma4_unified:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.tokenizer = None
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.processor = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=getattr(torch, config.KAGGLE_DTYPE), device_map="auto", **quant_kwargs
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
                enable_thinking=self.enable_thinking,      # Qwen3 templates default to thinking ON; other templates ignore this kwarg
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
            do_sample=self.temperature > 0,
        )
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        else:
            # Greedy: drop the sampling knobs a model's generation_config may carry.
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None
        # Seed before every call. Greedy ignores it. When sampling (a deliberate study, e.g.
        # self-consistency) the call counter is added so repeated samples differ yet the whole
        # run stays reproducible.
        import torch as _torch
        _seed = config.LLM_SEED + (getattr(self, "total_calls", 0) if self.temperature > 0 else 0)
        _torch.manual_seed(_seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(_seed)
        outputs = None
        chunk = int(os.environ.get("PREFILL_CHUNK", "1024"))
        if (os.environ.get("CHUNKED_PREFILL", "0") == "1" and self.tokenizer is not None
                and inputs["input_ids"].shape[-1] > chunk + 1):
            try:
                outputs = self._generate_chunked(inputs, gen_kwargs, chunk)
                self.chunked_prefills = getattr(self, "chunked_prefills", 0) + 1
            except Exception:                      # any failure -> original path below
                torch.cuda.empty_cache()
                self.chunked_fallbacks = getattr(self, "chunked_fallbacks", 0) + 1
                outputs = None
        try:
            if outputs is None:
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

    def _generate_chunked(self, inputs, gen_kwargs, chunk: int):
        """Prefill the prompt in chunks into a KV cache (attention memory ~ chunk x length instead of length^2),
        then generate from the cached prefix. Same model, same tokens, same greedy decoding."""
        import torch
        from transformers import DynamicCache
        ids = inputs["input_ids"]
        n = ids.shape[-1]
        cache = DynamicCache()
        with torch.no_grad():
            pos = 0
            while pos < n - 1:
                end = min(pos + chunk, n - 1)
                try:
                    self.model(input_ids=ids[:, pos:end], past_key_values=cache, use_cache=True, logits_to_keep=1)
                except TypeError:
                    self.model(input_ids=ids[:, pos:end], past_key_values=cache, use_cache=True)
                pos = end
            return self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       past_key_values=cache, **gen_kwargs)

    def score_options(self, prompt: str, labels, system: Optional[str] = None):
        """Constrained choice: log-probability of each single-token label as the FIRST token of the answer
        (log-softmax restricted to `labels`). One forward pass, no sampling. Returns {label: logprob} or None
        if a label is not a single token / the processor path is in use."""
        import torch
        if self.tokenizer is None:
            return None
        ids = []
        for lab in labels:
            t = self.tokenizer.encode(lab, add_special_tokens=False)
            if len(t) != 1:
                return None
            ids.append(t[0])
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        max_in = int(os.environ.get("MAX_INPUT_TOKENS", "9000"))
        if inputs["input_ids"].shape[-1] > max_in:
            self.truncated_prompts = getattr(self, "truncated_prompts", 0) + 1
            head, tail = int(max_in * 0.55), max_in - int(max_in * 0.55)
            t_ids = torch.cat([inputs["input_ids"][0][:head], inputs["input_ids"][0][-tail:]]).unsqueeze(0)
            inputs = {"input_ids": t_ids, "attention_mask": torch.ones_like(t_ids)}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float()
        sel = torch.log_softmax(logits[ids], dim=-1).tolist()
        self.total_calls += 1
        self.total_tokens_used += int(inputs["input_ids"].shape[-1])
        return dict(zip(labels, sel))

    def generate_json(self, prompt: str, system: Optional[str] = None) -> dict:
        raw = self.generate(prompt, system=system, json_mode=True)
        from agents.json_utils import parse_json_lenient
        return parse_json_lenient(raw)

    def usage_stats(self) -> dict:
        return {"total_calls": self.total_calls, "total_tokens": self.total_tokens_used,
                "truncated_prompts": getattr(self, "truncated_prompts", 0),
                "oom_retries": getattr(self, "oom_retries", 0),
                "chunked_prefills": getattr(self, "chunked_prefills", 0),
                "chunked_fallbacks": getattr(self, "chunked_fallbacks", 0)}

    def reset_usage(self) -> None:
        self.total_calls = 0
        self.total_tokens_used = 0
        self.truncated_prompts = 0
        self.oom_retries = 0
        self.chunked_prefills = 0
        self.chunked_fallbacks = 0
