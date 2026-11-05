"""Golden outputs of gama/ (tests/test_gama.py). The golden file was generated at commit fc4759d, when gama/ was proven identical to
the frozen development code (tag v5-frozen) on these synthetic scenarios and on all 103 RCA100 cases (PREREGISTRATION_V5.md).
StubLLM / snapshot / diff come from scripts/equivalence.py (archived with the development code)."""
from __future__ import annotations

import hashlib
import json
import re

IGNORE_STATS = {"total_pipeline_time_s", "gama_modules"}


class StubLLM:
    """Deterministic stand-in for the local LLM: answers from a hash of the prompt, records every call."""

    def __init__(self):
        self.calls = []
        self.reset_usage()

    def reset_usage(self):
        self.total_calls, self.total_tokens_used = 0, 0

    def usage_stats(self):
        return {"total_calls": self.total_calls, "total_tokens": self.total_tokens_used}

    def generate_json(self, prompt, system=None):
        self.calls.append((system or "", prompt))
        self.total_calls += 1
        self.total_tokens_used += len(prompt) // 4
        h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        msgs = re.findall(r"^- (.+)$", prompt, flags=re.M)
        if "Which ONE category" in prompt:
            from pipeline.belief_diagnosis import CATS
            return {"category": CATS[h % len(CATS)], "quote": (msgs[0][:30] if msgs else "")}
        if '"fault_type"' in prompt:
            cands = re.findall(r"^- (F\d+-\w+) \(statistical", prompt, flags=re.M)
            lines = [l for l in msgs if not re.match(r"F\d+-", l)]
            return {"fault_type": cands[h % len(cands)] if cands else "", "evidence_line": lines[h % len(lines)] if lines else ""}
        if "reasoning_chain" in prompt:
            return {"reasoning_chain": ["cause: stub", "impact: stub"]} if h % 3 else {}
        return {}


def snapshot(r, llm=None):
    st = {k: v for k, v in r.retrieval_stats.items() if k not in IGNORE_STATS and k != "_trace"}
    return {"entity": r.predicted_entity_ids, "type": r.predicted_fault_type, "confidence": repr(r.confidence),
            "chain": r.reasoning_chain, "evidence": [str(getattr(e.observation, "text", e.observation)) for e in r.evidence_items],
            "stats": json.loads(json.dumps(st, default=str, sort_keys=True)),
            "trail": json.loads(json.dumps(r.retrieval_stats.get("_trace"), default=str, sort_keys=True)),
            "llm_calls": llm.calls if llm is not None else None}


def diff(a, b, path=""):
    if type(a) != type(b):
        return [f"{path}: {str(a)[:80]!r} != {str(b)[:80]!r}"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'legacy' if k in a else 'gama'}")
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {str(a)[:80]!r} != {str(b)[:80]!r}"]


