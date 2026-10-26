"""
fp_oracle_coverage.py -- EVALUATION-ONLY analysis of the fault-fingerprint layer (pipeline/fault_fingerprint.py). Reads answer_key/.

Question it answers: if the ROOT-CAUSE ENTITY WERE KNOWN (oracle entity from the answer key), how often is the true fault type among the
fingerprint candidates (top-1 / top-3 / any)? This separates the two failure sources of the fault type: wrong entity vs wrong evidence-to-label
mapping, and gives the ceiling of FI for the shortlist arm. No LLM, no GPU.

Leakage protocol (write it in the paper): the fingerprint table and thresholds are frozen BEFORE this script is run on held-out cases.
Run it on the dev-30 cases while developing; run it on HELDOUT exactly once, after the freeze, and report what it says, good or bad.
Do NOT edit FINGERPRINTS / thresholds after seeing held-out output.

Scoring: FP_SCORING=calibrated (default) or fixed. The calibrated run also prints the label-free false-alarm rate of the rank tests.

Usage (RCA100_ROOT set):
    python scripts/diagnostics/fp_oracle_coverage.py                 # dev-30 cases
    python scripts/diagnostics/fp_oracle_coverage.py --all           # all 103 (only after the freeze)
    python scripts/diagnostics/fp_oracle_coverage.py t004 t005       # specific cases
    python scripts/diagnostics/fp_oracle_coverage.py --csv out.csv
"""
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.loader import Case, list_case_ids
from data.taxonomy import fault_group
from evaluation.scoring import load_ground_truth
import config
if config.FP_SCORING == "fixed":
    from pipeline.fault_fingerprint import compute_signal_db, origin_signals, rank_candidates
    false_alarm_rate = None
else:
    from pipeline.fault_calibrated import compute_signal_db, origin_signals, rank_candidates, false_alarm_rate

DEV30 = ["t004", "t005", "t012", "t014", "t015", "t018", "t026", "t028", "t029", "t030", "t032", "t036", "t054", "t055", "t058",
         "t065", "t070", "t072", "t076", "t078", "t082", "t087", "t091", "t092", "t093", "t095", "t097", "t099", "t100", "t101"]


def main(argv):
    csv_path = None
    if "--csv" in argv:
        i = argv.index("--csv")
        csv_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    ids = list_case_ids() if "--all" in argv else ([a for a in argv if not a.startswith("--")] or DEV30)
    rows = []
    for cid in ids:
        t0 = time.time()
        try:
            case = Case(cid)
            gt = load_ground_truth(cid, case.name_index)
        except Exception as e:
            print(f"{cid}: load error {e!r}")
            continue
        names = [te.get("entity_name") for te in gt.target_entity_names if te.get("entity_name")] or list(gt.root_cause_entities)
        db = compute_signal_db(case, case.alert.alert_timestamp)
        osig = origin_signals(db, names[0] if names else None)
        cands = rank_candidates(osig, top_k=0)
        far = false_alarm_rate(db) if false_alarm_rate else {}
        types = [c["type"] for c in cands]
        truth = gt.fault_type
        rank = types.index(truth) + 1 if truth in types else None
        rows.append(dict(case=cid, gt_type=truth, group=fault_group(truth), origin=names[0] if names else "", kind=osig["kind"],
                         rank=rank, n_cand=len(types), top3="|".join(types[:3]), signals="|".join(sorted(osig["signals"])),
                         far_series=far.get("series"), far_fired=far.get("fired"), far_raw_fired=far.get("raw_fired"),
                         secs=round(time.time() - t0, 1)))
        r = rows[-1]
        print(f"{cid} {truth:32s} origin={r['origin']}({r['kind']}) rank={rank} n={len(types)} top3={r['top3']} | signals={r['signals']}")

    n = len(rows)
    if not n:
        return
    top1 = sum(1 for r in rows if r["rank"] == 1)
    top3 = sum(1 for r in rows if r["rank"] and r["rank"] <= 3)
    anyc = sum(1 for r in rows if r["rank"])
    empty = sum(1 for r in rows if r["n_cand"] == 0)
    print(f"\nORACLE ENTITY, {n} cases: top1={top1}/{n} ({100 * top1 / n:.1f}%)  top3={top3}/{n} ({100 * top3 / n:.1f}%)  "
          f"any={anyc}/{n}  no candidate={empty}/{n}")
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    for g, rs in sorted(by.items()):
        print(f"  {g:20s} n={len(rs):3d}  top1={sum(1 for r in rs if r['rank'] == 1)}  top3={sum(1 for r in rs if r['rank'] and r['rank'] <= 3)}")
    fs, ff = sum(r["far_series"] or 0 for r in rows), sum(r["far_fired"] or 0 for r in rows)
    fr = sum(r["far_raw_fired"] or 0 for r in rows)
    if fs:
        print(f"CALIBRATION (label-free): held-out pseudo-window false-alarm rate per signal = calibrated {ff}/{fs} = {ff / fs:.3f}, "
              f"uncalibrated rank test {fr}/{fs} = {fr / fs:.3f} (target alpha = {config.FP_ALPHA})")
    print(f"scoring = {config.FP_SCORING}")
    if csv_path:
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("written", csv_path)


if __name__ == "__main__":
    main(sys.argv[1:])
