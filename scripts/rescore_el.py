"""
rescore_el.py -- recompute entity_localization, rcl_strict and final scores of existing run CSVs with the corrected roll-up
(evaluation/scoring.py, 2026-10-30: a k8s.node answer was rolled up to an arbitrary service hosted on it and got full EL credit).
Uses the answer key (scoring only) and each case's topology; the predictions in the CSV are unchanged.

Usage (RCA100_ROOT set):  python scripts/rescore_el.py run1.csv run2.csv ...   -> writes run1_rescored.csv ... and prints a summary
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config
from data.loader import load_topology, build_name_index
from evaluation.scoring import load_ground_truth, entity_localization_score, _scoring_rollup


def main(paths):
    topo_cache = {}
    for path in paths:
        d = pd.read_csv(path)
        el, strict = [], []
        for r in d.itertuples(index=False):
            cid = r.case_id
            if cid not in topo_cache:
                t = load_topology(os.path.join(config.CASES_DIR, cid))
                topo_cache[cid] = (t, load_ground_truth(cid, build_name_index(t)))
            topo, gt = topo_cache[cid]
            ids = [e for e in str(r.predicted_entity_ids or "").split("|") if e and e != "nan"]
            el.append(entity_localization_score(ids, gt, topo))
            tg = gt.target_entity_ids or []
            strict.append(1.0 if ids and tg and _scoring_rollup(ids[0], topo) in {_scoring_rollup(t, topo) for t in tg} else 0.0)
        old_el = d["entity_localization"].copy()
        d["entity_localization_old"], d["entity_localization"], d["rcl_strict"] = old_el, el, strict
        for fin, rp in (("final_score", "reasoning_process"), ("final_score_chain", "reasoning_process_chain")):
            if fin in d and rp in d:
                d[fin] = (config.WEIGHT_ENTITY_LOCALIZATION * d["entity_localization"] + config.WEIGHT_FAULT_IDENTIFICATION
                          * d["fault_identification"] + config.WEIGHT_REASONING_PROCESS * d[rp]).round(4)
        out = path.replace(".csv", "_rescored.csv")
        d.to_csv(out, index=False)
        j = (d.rcl_exact * d.ft_binary).sum()
        print(f"{os.path.basename(path):34s} EL {old_el.mean():.3f} -> {d.entity_localization.mean():.3f} | rcl_strict {d.rcl_strict.sum():.0f} | "
              f"rcl_exact {d.rcl_exact.sum():.0f} | joint {j:.0f} | final {d.final_score.mean():.3f}  -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
