"""GAMA-RCA on AIOps2025, zero-shot (PREREGISTRATION_AIOPS2025.md). Reads input.json and the telemetry; never opens ground truth.

    python scripts/aiops2025_run.py --data /kaggle/input/aiops2025 --out /kaggle/working --shard 0/4            # diagnose
    python scripts/aiops2025_run.py --data ... --check --cases first:5                                         # label-free checks

Writes <name>.jsonl (one answer per case, the benchmark's submission format: uuid, component, reason, time, reasoning_trace),
<name>.csv (decision statistics per case), <name>.manifest.json. --check writes <name>_check.csv: adapter coverage and, per entity
kind and signal, how many (entity, signal) pairs are testable in the window and how many hold (false-alarm view).
"""
import os
import sys

os.environ.setdefault("DATASET_PROFILE", "aiops2025")       # must precede every import of the core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse          # noqa: E402
import json              # noqa: E402
import platform          # noqa: E402
import subprocess        # noqa: E402
import time              # noqa: E402
import traceback         # noqa: E402
from collections import Counter   # noqa: E402

import pandas as pd      # noqa: E402

from data.dataset_profile import PROFILE   # noqa: E402
from data.aiops2025_profile import phrase  # noqa: E402
from adapters.aiops2025 import Release, AIOpsCase   # noqa: E402
from gama.config import Modules            # noqa: E402
from gama.belief import modality           # noqa: E402

ACTION = {"metrics": "LoadMetrics", "traces": "TraceAnalysis", "logs": "LogSearch", "events": "LoadEvents"}
MAX_STEPS = 5                               # A6: short reasoning trace (Efficiency)


def answer(case, r) -> dict:
    tr = r.retrieval_stats.get("_trace") or {}
    ev = [e for e in (tr.get("evidence_by_bayes_factor") or []) if e.get("holds")][:MAX_STEPS]
    comp = r.predicted_entity_ids[0] if r.predicted_entity_ids else ""
    reason = phrase(r.predicted_fault_type) + (f" at {comp}" if comp else "")
    if ev:
        reason += "; " + "; ".join(f"{e['entity']} {e['signal']}" for e in ev[:3])
    steps = [{"step": i + 1, "action": f"{ACTION[modality(e['signal'])]}({e['entity']})", "observation": str(e["ev"])[:160]}
             for i, e in enumerate(ev)]
    return {"uuid": case.case_id, "component": comp, "reason": reason, "time": case.alert.raw["alert_window"]["start"],
            "reasoning_trace": steps}


def check(case, m) -> list:
    """Label-free: testable / holding continuous signals per (entity kind, signal) in the window."""
    from gama import evidence
    from gama.chain import callee_map
    from pipeline.fault_v4 import calibrate_v4
    import config
    config.ALERT_T0_MODE = "window"                     # as in gama/run.run_case: evidence window = the case window
    db = evidence.signal_db(case, case.alert.alert_timestamp, m)
    if "services" not in db:
        return [{"case": case.case_id, "error": db.get("error")}]
    calibrate_v4(db, peer=True, robust_null=False, ops_level=False, selfnorm=True, ratio=False, callees=callee_map(case.topology))
    kind = {e: ("node" if e in db["nodes"] else ("pod" if e in db.get("memberships", {}) else "service"))
            for e in list(db["services"]) + list(db["nodes"])}
    tot, hold, ent = Counter(), Counter(), Counter(kind.values())
    for e, sigs in db["_cal"].items():
        for s, r in sigs.items():
            if r.get("p") is not None:
                tot[(kind.get(e, "?"), s)] += 1
                hold[(kind.get(e, "?"), s)] += r["p"] <= 0.05
    return [{"case": case.case_id, "kind": k, "signal": s, "entities": ent[k], "testable": n, "holds": hold[(k, s)]}
            for (k, s), n in sorted(tot.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input/aiops2025")
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "/kaggle/working"))
    ap.add_argument("--cases", default="all", help="all | first:N | uuid,uuid,...")
    ap.add_argument("--shard", default=None, help="k/n: every n-th case starting at k")
    ap.add_argument("--modules", default="core")
    ap.add_argument("--check", action="store_true", help="label-free adapter / signal checks only (no diagnosis)")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    assert PROFILE == "aiops2025", "DATASET_PROFILE must be aiops2025 before the core is imported"
    os.makedirs(a.out, exist_ok=True)
    rel = Release(a.data)
    specs = rel.cases()
    windows = [(s["start"], s["end"]) for s in specs]
    if a.cases.startswith("first:"):
        specs = specs[:int(a.cases.split(":")[1])]
    elif a.cases != "all":
        want = set(a.cases.split(","))
        specs = [s for s in specs if s["uuid"] in want]
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        specs = [s for i, s in enumerate(specs) if i % n == k]
    m = Modules.parse(a.modules)
    commit = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).stdout.strip()
    name = a.name or ("AIOPS_CHECK" if a.check else f"AIOPS_GAMA_{m.spec().replace(',', '_').replace('-', 'no')}") + \
        (f"_shard{a.shard.replace('/', 'of')}" if a.shard else "")
    print(f"commit {commit} | profile {PROFILE} | modules {m.spec()} | {len(specs)} cases | output {name}", flush=True)
    manifest = {"commit": commit, "profile": PROFILE, "modules": m.spec(), "cases": [s["uuid"] for s in specs], "shard": a.shard,
                "check": a.check, "python": platform.python_version(), "cpu_count": os.cpu_count(),
                "history_mask": os.environ.get("AIOPS_HISTORY_MASK", "none"), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    rows, checks, t_all = [], [], time.time()
    jl = open(os.path.join(a.out, f"{name}.jsonl"), "w", encoding="utf-8") if not a.check else None
    for i, s in enumerate(specs):
        t0 = time.time()
        try:
            case = AIOpsCase(rel, s, all_windows=windows)
            cov = case.coverage()
            if a.check:
                checks += check(case, m)
                rows.append({"case_id": s["uuid"], **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in cov.items()},
                             "wall_s": round(time.time() - t0, 1)})
                print(f"[{i + 1}/{len(specs)}] {s['uuid']} {cov}", flush=True)
                continue
            from gama.run import run_case
            r = run_case(case, None, m)
            ans = answer(case, r)
            jl.write(json.dumps(ans, ensure_ascii=False) + "\n")
            jl.flush()
            st = {k: v for k, v in r.retrieval_stats.items() if not k.startswith("_") and not isinstance(v, (dict, list))}
            rows.append({"case_id": s["uuid"], "component": ans["component"], "fault_type": r.predicted_fault_type,
                         "confidence": round(r.confidence, 4), "reason": ans["reason"], "n_steps": len(ans["reasoning_trace"]),
                         **{k: v for k, v in cov.items() if not isinstance(v, dict)}, **st, "wall_s": round(time.time() - t0, 1)})
            print(f"[{i + 1}/{len(specs)} {(time.time() - t_all) / 60:5.1f} min] {s['uuid']} -> {ans['component']} "
                  f"{r.predicted_fault_type} p={r.confidence:.2f} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            traceback.print_exc()
            rows.append({"case_id": s["uuid"], "error": repr(e)[:300], "wall_s": round(time.time() - t0, 1)})
            if jl:
                jl.write(json.dumps({"uuid": s["uuid"], "component": "", "reason": "", "time": "", "reasoning_trace": []}) + "\n")
            print(f"[{i + 1}/{len(specs)}] {s['uuid']} [error] {e!r}", flush=True)
    if jl:
        jl.close()
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
    if checks:
        c = pd.DataFrame(checks)
        c.to_csv(os.path.join(a.out, f"{name}_check.csv"), index=False)
        if "signal" in c.columns:
            g = c.groupby(["kind", "signal"])[["entities", "testable", "holds"]].sum()
            g["testable_share"] = (g["testable"] / g["entities"]).round(3)
            g["hold_rate"] = (g["holds"] / g["testable"].clip(lower=1)).round(3)
            print(g.to_string())
    manifest.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), wall_s=round(time.time() - t_all, 1),
                    n_errors=sum(1 for r in rows if r.get("error")))
    json.dump(manifest, open(os.path.join(a.out, f"{name}.manifest.json"), "w"), indent=1)
    subprocess.run(f"cd {a.out} && rm -f {name}_all.zip && zip -q {name}_all.zip {name}.* {name}_check.csv 2>/dev/null", shell=True)
    print("download:", os.path.join(a.out, f"{name}_all.zip"))


if __name__ == "__main__":
    main()
