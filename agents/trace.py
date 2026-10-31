"""Run traces (v5): every LLM call and every case's decision, as JSON, for audit and W&B.

Files, next to the results CSV (<name>.csv -> <name>_trace/):
  llm_calls.jsonl     one line per LLM call: case, system, stage (sensor / typing / narrator / evidence_llm / ...), the system
                      message and prompt EXACTLY as sent, the raw output, prompt and output token counts, latency, truncation
  cases/<case>__<system>.json   the decision trail of one case: calibrated observations that hold, top hypotheses, propagation pi,
                      onsets, text probes, typing shortlist and choice, call path, chain, grounding, and (added by main.py AFTER the
                      diagnosis, for analysis only) the prediction and the scores. Nothing in a trace is ever fed back to a model.
JSON Lines for the calls (append-only, survives a crashed session, streams into pandas / W&B); one JSON file per case for the trail."""
from __future__ import annotations

import json
import os
import time

STAGES = (("You classify failure messages", "sensor"), ("You are a site-reliability engineer classifying the fault mechanism", "typing"),
          ("You are a site-reliability engineer. The diagnosis has already been decided", "narrator"),
          ("You are a site-reliability engineer performing root cause analysis of a microservice incident from summarised", "evidence_llm"))


def stage_of(system: str) -> str:
    for prefix, name in STAGES:
        if (system or "").startswith(prefix):
            return name
    return "other"


def _default(o):
    try:
        return o.isoformat()
    except Exception:
        return str(o)


def install_llm_trace(llm, path: str) -> None:
    """Wrap llm.generate (used by generate_json) so every call is appended to `path`. The caller sets llm.trace_ctx per case."""
    if llm is None or getattr(llm, "_trace_installed", False):
        return
    orig = llm.generate
    os.makedirs(os.path.dirname(path), exist_ok=True)
    llm.trace_ctx = {}

    def generate(prompt, system=None, json_mode=False):
        t0 = time.time()
        tok0, calls0 = getattr(llm, "total_tokens_used", 0), getattr(llm, "total_calls", 0)
        trunc0 = getattr(llm, "truncated_prompts", 0)
        err, raw = None, None
        try:
            raw = orig(prompt, system=system, json_mode=json_mode)
            return raw
        except Exception as e:
            err = repr(e)
            raise
        finally:
            rec = {**getattr(llm, "trace_ctx", {}), "stage": stage_of(system), "ts": round(t0, 3),
                   "latency_s": round(time.time() - t0, 2), "system_message": system, "prompt": prompt, "output": raw,
                   "prompt_chars": len(prompt or ""), "tokens_total_delta": getattr(llm, "total_tokens_used", 0) - tok0,
                   "calls_delta": getattr(llm, "total_calls", 0) - calls0,
                   "truncated": getattr(llm, "truncated_prompts", 0) > trunc0,
                   "system_merged": bool(getattr(getattr(llm, "tokenizer", None), "system_merged", False)), "error": err}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=_default) + "\n")

    llm.generate = generate
    llm._trace_installed = True


def write_case_trace(trace_dir: str, case_id: str, system: str, trail: dict, outcome: dict) -> str:
    os.makedirs(os.path.join(trace_dir, "cases"), exist_ok=True)
    p = os.path.join(trace_dir, "cases", f"{case_id}__{system}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"case_id": case_id, "system": system, "trail": trail or {}, "outcome_for_analysis_only": outcome}, f,
                  ensure_ascii=False, indent=1, default=_default)
    return p


def log_to_wandb(wandb_run, trace_dir: str, name: str) -> None:
    """Upload the trace folder as a W&B artifact and the LLM calls as a browsable table."""
    import wandb
    calls = os.path.join(trace_dir, "llm_calls.jsonl")
    if os.path.exists(calls):
        rows = [json.loads(l) for l in open(calls, encoding="utf-8") if l.strip()]
        cols = ["case_id", "system", "stage", "latency_s", "tokens_total_delta", "prompt_chars", "truncated", "system_message", "prompt",
                "output", "error"]
        wandb_run.log({"llm_calls": wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])})
    art = wandb.Artifact(name=f"trace-{name}".replace("/", "_")[:120], type="trace")
    art.add_dir(trace_dir)
    wandb_run.log_artifact(art)
