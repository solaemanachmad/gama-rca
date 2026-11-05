"""
retype.py -- re-run ONLY the fault-typing step of the FP arms on the saved state of the reference arm (GPU budget).

Why this is equivalent to a full run: the FP arms differ from the reference arm A1rT_fixed_ABS_KP only AFTER the chain-first Coordinator
(same specialists, same Coordinator prompt, deterministic decoding). Entity, chain and RP are therefore identical; only the fault type
(and the FI-based scores) can change. `compare` checks this on cases that were also run in full.

    # 0) reference arm run with state saving (Cell 5 sets SAVE_TYPING_STATE_DIR=/kaggle/working/state_<name>)
    # 1) CPU: signal databases for both scorings (loads each case once; no model)
    python scripts/retype.py prepare --ref REF.csv --state STATE_DIR
    # 2) per arm: rule arms on CPU, show/shortlist arms on GPU (one LLM call per case)
    python scripts/retype.py run --ref REF.csv --state STATE_DIR --arm A1rT_fixed_ABS_KP_FPSL --out FPSL.csv
    # 3) equivalence check against a full run of the same arm on the same cases
    python scripts/retype.py compare FPSL_retyped.csv FPSL_full.csv

Reads answer_key/ only to SCORE the new fault type (evaluation, as main.py does); the typing step itself never sees it.
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

REFERENCE_ARM = "A1rT_fixed_ABS_KP"
ARMS = {   # arm -> (FAULT_FINGERPRINT, FP_SCORING); every other flag equals the reference arm (see scripts/kaggle_run.py ARMS)
    "A1rT_fixed_ABS_KP_FPSL": ("shortlist", "calibrated"),
    "A1rT_fixed_ABS_KP_FPRULE": ("rule", "calibrated"),
    "A1rT_fixed_ABS_KP_FPSHOW": ("show", "calibrated"),
    "A1rT_fixed_ABS_KP_FPRULE_FIX": ("rule", "fixed"),
    "A1rT_fixed_ABS_KP_FPSL_FIX": ("shortlist", "fixed"),
}
FP_COLS = ("coordinator_fault_type", "fault_evidence", "typing_applied", "typing_prompt_chars", "fp_mode", "fp_scoring", "fp_origin",
           "fp_origin_kind", "fp_signals", "fp_candidates", "fp_choice_in_shortlist", "fp_entity_switched_from")


def _plain(x):
    """defaultdicts (some with lambda factories) -> dicts, so the signal database can be pickled. Counters and lists stay."""
    if isinstance(x, defaultdict):
        x = dict(x)
    if isinstance(x, dict) and type(x) is dict:
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    return x


def _ref_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("case_id") and not (r.get("error") or "").strip()]


def prepare(args):
    from data.loader import Case
    from pipeline import fault_calibrated, fault_fingerprint
    mods = {"calibrated": fault_calibrated, "fixed": fault_fingerprint}
    for r in _ref_rows(args.ref):
        cid = r["case_id"]
        todo = [s for s in args.scoring if not os.path.exists(os.path.join(args.state, f"fpdb_{s}", f"{cid}.pkl"))]
        if not todo:
            continue
        t0 = time.time()
        case = Case(cid)
        for s in todo:
            db = mods[s].compute_signal_db(case, case.alert.alert_timestamp)
            if s == "calibrated" and "services" in db:
                fault_calibrated._calibrate(db)          # cache the calibrated p-values inside the pickle
            os.makedirs(os.path.join(args.state, f"fpdb_{s}"), exist_ok=True)
            with open(os.path.join(args.state, f"fpdb_{s}", f"{cid}.pkl"), "wb") as f:
                pickle.dump(_plain(db), f)
        print(f"{cid}: {', '.join(todo)} in {time.time() - t0:.0f}s", flush=True)


def _set_flags(mode, scoring):
    for k, v in dict(FAULT_FINGERPRINT=mode, FP_SCORING=scoring, ENTITY_TYPING=False, COORD_LEAN=False, TYPING_COMPARE=False,
                     USE_KNOWLEDGE_POS=True, FP_TOP_K=3, FP_ALPHA=0.05, FP_LEVEL_SWITCH=False, VERIFY_CHAIN=False, DEBUG_DUMP_DIR="").items():
        setattr(config, k, v)


def _rescore(row, pred, gt):
    from evaluation.scoring import fault_identification_score, fault_identification_tiered, fault_group_score
    from data.taxonomy import fault_group
    f = lambda k: float(row[k]) if row.get(k) not in (None, "") else 0.0
    fi = fault_identification_score(pred, gt)
    p, t = (pred or "").strip().lower(), gt.fault_type.strip().lower()
    row["predicted_fault_type"] = pred
    row["fault_identification"] = fi
    row["fault_identification_exact"] = 1.0 if (p and (p == t or p == t.split("-")[-1] or p.split("-")[-1] == t.split("-")[-1])) else 0.0
    row["fi_tiered"] = fault_identification_tiered(pred, gt)
    row["fault_group_identification"] = fault_group_score(pred, gt)
    row["predicted_fault_group"] = fault_group(pred)
    row["ft_binary"] = fi
    row["joint_strict"] = f("rcl_strict") * fi
    row["joint_exact"] = f("rcl_exact") * fi
    W = (config.WEIGHT_ENTITY_LOCALIZATION, config.WEIGHT_FAULT_IDENTIFICATION, config.WEIGHT_REASONING_PROCESS)
    row["final_score"] = round(W[0] * f("entity_localization") + W[1] * fi + W[2] * f("reasoning_process"), 4)
    row["final_score_chain"] = round(W[0] * f("entity_localization") + W[1] * fi + W[2] * f("reasoning_process_chain"), 4)


def run(args):
    from agents import multi_agent as ma
    from data.taxonomy import taxonomy_prompt_block
    from evaluation.scoring import load_ground_truth
    mode, scoring = ARMS[args.arm]
    _set_flags(mode, scoring)
    llm = None
    if mode != "rule":
        from agents.factory import get_llm_client
        llm = get_llm_client()
    out = []
    for r in _ref_rows(args.ref):
        cid = r["case_id"]
        with open(os.path.join(args.state, f"{cid}.json"), encoding="utf-8") as f:
            st = json.load(f)
        with open(os.path.join(args.state, f"fpdb_{scoring}", f"{cid}.pkl"), "rb") as f:
            st["fp_db"] = pickle.load(f)
        st["final_result"] = dict(st.get("final_result") or {})
        row = dict(r)
        ref_type = st["final_result"].get("predicted_fault_type")
        if ref_type != r.get("predicted_fault_type"):
            row["retype_warning"] = f"state type {ref_type} != reference CSV type {r.get('predicted_fault_type')}"
        calls0 = getattr(llm, "total_calls", 0) if llm else 0
        ma._apply_entity_typing(llm, st, taxonomy_prompt_block())
        fr = st["final_result"]
        for k in FP_COLS:
            row[k] = fr.get(k, "")
        _rescore(row, fr.get("predicted_fault_type", "unknown"), load_ground_truth(cid))
        row["arm"] = args.arm
        row["retyped_from"] = os.path.basename(args.ref)
        row["retype_llm_calls"] = (getattr(llm, "total_calls", 0) - calls0) if llm else 0
        out.append(row)
        print(f"{cid}: {ref_type} -> {row['predicted_fault_type']} | gt {r.get('gt_fault_type')} | cand {row['fp_candidates']}", flush=True)
    cols = list(out[0]) if out else []
    for r in out:
        cols += [k for k in r if k not in cols]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    n = len(out)
    fi = sum(float(r["fault_identification_exact"]) for r in out)
    print(f"\n{args.arm}: {n} cases, FI exact {fi:.0f}/{n}; written {args.out}")


def compare(args):
    a = {r["case_id"]: r for r in _ref_rows(args.a)}
    b = {r["case_id"]: r for r in _ref_rows(args.b)}
    common = sorted(set(a) & set(b))
    same = 0
    for c in common:
        ok = a[c].get("predicted_fault_type") == b[c].get("predicted_fault_type") and \
             a[c].get("predicted_entity_ids") == b[c].get("predicted_entity_ids")
        same += ok
        print(f"{c}: {'SAME' if ok else 'DIFF'}  type {a[c].get('predicted_fault_type')} / {b[c].get('predicted_fault_type')}  "
              f"cand {a[c].get('fp_candidates')} / {b[c].get('fp_candidates')}")
    print(f"\nidentical type and entity: {same}/{len(common)} common cases")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--ref", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--scoring", nargs="+", default=["calibrated", "fixed"])
    p = sub.add_parser("run")
    p.add_argument("--ref", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    a = ap.parse_args()
    {"prepare": prepare, "run": run, "compare": compare}[a.cmd](a)
