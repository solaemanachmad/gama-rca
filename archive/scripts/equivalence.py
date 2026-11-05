"""Equivalence of the refactored method (gama/) with the frozen development code (pipeline/belief_rca.run_case + belief_v4, tag
v5-frozen) on real cases. Compares, per case and per module configuration, everything the run returns: entity, type, confidence,
chain, evidence items, every stats column (except wall time) and the decision trail. No labels are read.

  python scripts/equivalence.py --data <RCA100 root> --case_ids t001 t002 ... --configs core,loo [--out equivalence.csv]

configs: core (GAMA_CORE) | loo (the five leave-one-module-out arms + no chain) | base (M0 only) | llm (GAMA_LLM with a
deterministic stub LLM that records every prompt: prompts and answers must match too). Usually run via
scripts/kaggle_job.py --check ... (same environment as the scored runs)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

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


def configs(names):
    from gama.config import Modules, CORE
    out = []
    for n in names:
        if n == "core":
            out.append(("core", Modules.parse("core")))
        elif n == "base":
            out.append(("base", Modules.parse("base")))
        elif n == "llm":
            out.append(("llm", Modules.parse("full")))
        elif n == "loo":
            out += [(f"core,-{x}", Modules.parse(f"core,-{x}")) for x in CORE]
        else:
            out.append((n, Modules.parse(n)))
    return out


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


def run_pair(make_case, m):
    """make_case() -> a fresh Case. Returns (legacy snapshot, gama snapshot)."""
    import config
    from pipeline.belief_diagnosis import PROBE_BUDGET
    from pipeline.belief_rca import run_case as legacy_run
    from gama.run import run_case as gama_run
    config.ALERT_T0_MODE = "window"
    os.environ["V4_PARTS"] = ",".join(m.legacy_parts())
    l1 = StubLLM() if m.llm else None
    r1 = legacy_run(make_case(), l1, PROBE_BUDGET if l1 else 0, narrate=l1 is not None, v4=True)
    os.environ.pop("V4_PARTS", None)                  # gama must not depend on it
    l2 = StubLLM() if m.llm else None
    r2 = gama_run(make_case(), l2, m)
    return snapshot(r1, l1), snapshot(r2, l2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--case_ids", nargs="+", required=True)
    ap.add_argument("--configs", default="core")
    ap.add_argument("--out", default="equivalence.csv")
    a = ap.parse_args()
    import config
    from data.loader import Case
    cases_dir = os.path.join(a.data, "cases") if os.path.isdir(os.path.join(a.data, "cases")) else config.CASES_DIR
    rows, n_bad = [], 0
    for cid in a.case_ids:
        for name, m in configs(a.configs.split(",")):
            t0 = time.time()
            try:
                s1, s2 = run_pair(lambda: Case(cid, cases_dir=cases_dir), m)
                d = diff(s1, s2)
                err = ""
            except Exception:
                import traceback
                d, err = ["exception"], traceback.format_exc()[-600:]
            n_bad += bool(d)
            rows.append({"case_id": cid, "config": name, "identical": not d, "n_diffs": len(d), "diffs": " | ".join(d[:8]),
                         "error": err, "seconds": round(time.time() - t0, 1)})
            print(f"{cid} {name:14s} {'IDENTICAL' if not d else 'DIFFERENT'} ({len(d)}) {rows[-1]['seconds']} s "
                  + (d[0] if d else ""), flush=True)
            if err:
                print(err, flush=True)
    import csv
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows) - n_bad}/{len(rows)} identical -> {a.out}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
