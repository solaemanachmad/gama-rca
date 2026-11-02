"""checkpoint_names_dev.py -- evaluation diagnostic (amendment 20), ODD CASES ONLY: which signal NAMES do the ground-truth checkpoints
use per step, and does the predicted chain step for that target cite a signal of the same name? Reads the answer key for the odd
(development) cases only, to explain the step-anchored RP; it does not feed any decision.

Usage (Kaggle): !RCA100_ROOT=... python scripts/diagnostics/checkpoint_names_dev.py /kaggle/working/GAMA_CORE_RED_7B_shard0of4.csv ... \
                 > /kaggle/working/checkpoint_names_odd.txt 2>&1
"""
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main(csvs):
    import pandas as pd
    import config
    from data.loader import Case
    from evaluation.scoring import load_ground_truth, _norm_target
    d = pd.concat([pd.read_csv(c) for c in csvs])
    d = d[d.case_id.map(lambda c: int(c[1:]) % 2 == 1)]
    sig_by_type, cited_same, cited_other, missing_step = Counter(), Counter(), Counter(), Counter()
    pairs = Counter()
    for _, r in d.iterrows():
        case = Case(r.case_id, cases_dir=config.CASES_DIR)
        gt = load_ground_truth(r.case_id, name_index=case.name_index)
        steps = [s for s in str(r.reasoning_chain or "").split(" || ")]
        for cp in gt.checkpoints:
            sig = str(cp.get("signal") or "")
            sig_by_type[(cp.get("step_type"), sig)] += 1
            tgt = _norm_target(cp.get("target"))
            mine = [s for s in steps if tgt and tgt in _norm_target(s)]
            if not mine:
                missing_step[cp.get("step_type")] += 1
                continue
            names = set(re.findall(r"([A-Za-z_][A-Za-z0-9_.]*)=[-+0-9.]", " ".join(mine)))
            if sig in names:
                cited_same[sig] += 1
            else:
                cited_other[sig] += 1
                for n in names:
                    pairs[(sig, n)] += 1
    print("ground-truth checkpoint signals by step type (odd cases):")
    for (st, sg), n in sig_by_type.most_common(30):
        print(f"  {n:4d}  {st:12s} {sg}")
    print("\ncheckpoint whose target has NO step in our chain, by step type:", dict(missing_step))
    print("checkpoint signal cited under the SAME name at its step:", sum(cited_same.values()), dict(cited_same.most_common(10)))
    print("checkpoint signal NOT cited at its step (step exists):", sum(cited_other.values()), dict(cited_other.most_common(10)))
    print("\nmost frequent (checkpoint signal, name we cited at that step) pairs when the name differs:")
    for (a, b), n in pairs.most_common(20):
        print(f"  {n:4d}  {a}  <->  {b}")


if __name__ == "__main__":
    main(sys.argv[1:])
