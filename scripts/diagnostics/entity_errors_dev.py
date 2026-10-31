"""entity_errors_dev.py -- v5 error analysis of entity localisation, DEVELOPMENT (ODD) CASES ONLY (PREREGISTRATION_V5.md).

For every odd case whose predicted entity is wrong in a results CSV, it re-runs the statistical core (no LLM) and prints, for the
PREDICTED and the TRUE entity: the observations that hold (signal: values), the onset, the hosting nodes, and whether the entity can
explain the alerted service; plus, for the whole case, the number of holding observations per entity (does one service always look
anomalous?). The answer key is read here ONLY to name the true entity for the analysis; the diagnosis itself never sees it.
Even cases are refused: they are the validation half and stay unseen until the v5 design is frozen.

Usage (on Kaggle, RCA100_ROOT set):  ALERT_T0_MODE=window python scripts/diagnostics/entity_errors_dev.py V4_dev.csv > entity_errors_odd.txt
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

import config
from data.loader import Case, find_service_ancestor
from evaluation.scoring import load_ground_truth
from pipeline import belief_diagnosis as bd
from pipeline.belief_rca import callee_map
from pipeline.belief_v4 import diagnose_v4, DEFAULT_PARTS
from pipeline.fault_calibrated import compute_signal_db
from pipeline.propagation_evidence import _node_name

PARTS = [p for p in DEFAULT_PARTS if p != "llmtype"]


def main(csv):
    df = pd.read_csv(csv)
    df = df[df.case_id.map(lambda c: int(c[1:]) % 2 == 1)]
    wrong = df[df.entity_localization < 0.999]
    print(f"odd cases: {len(df)}, entity wrong: {len(wrong)}\n")
    noisy = Counter()
    for _, r in wrong.iterrows():
        cid = r.case_id
        case = Case(cid, cases_dir=config.CASES_DIR)
        gt = load_ground_truth(cid, name_index=case.name_index)
        true_names = [_node_name(case.topology, e) for e in gt.target_entity_ids if e in case.topology] or gt.root_cause_entities
        entry = case.alert.entry_entity_id
        sid = find_service_ancestor(entry, case.topology) if entry else None
        alert = _node_name(case.topology, sid) if sid else None
        db = compute_signal_db(case, case.alert.alert_timestamp)
        callees = callee_map(case.topology)
        res = diagnose_v4(db, callees, {}, sensor=None, budget=0, alert=alert, parts=PARTS, case=case)
        obs = bd.observations(db)
        held = Counter(o["entity"] for o in obs if o["holds"])
        expl = bd.explainers(db, callees)
        rank = [e for e, _ in held.most_common()]
        print(f"################ {cid}  alert={alert}  true={true_names} ({gt.fault_type})  predicted={res['entity']} ({res['type']}) "
              f"p={res['p']:.3f} pi={res.get('pi')}")
        print("  top:", " | ".join(f"{t['entity']}:{t['type']}:{t['p']}" for t in res["top"]))
        for tag, ent in [("PREDICTED", res["entity"])] + [("TRUE", t) for t in true_names]:
            hold = [f"{o['signal']}: {o['ev']}" for o in obs if o["entity"] == ent and o["holds"]]
            on = (db.get("_cal") or {}).get(ent, {})
            onset = min([v["onset"] for v in on.values() if v.get("onset") is not None and v.get("p") is not None and v["p"] <= bd.ALPHA],
                        default=None)
            can = alert in db["services"] and (ent == alert or ent in expl.get(alert, ()))
            print(f"  {tag} {ent}: hosts={db.get('hosts', {}).get(ent, [])} holding={len(hold)} "
                  f"(rank {rank.index(ent) + 1 if ent in rank else '-'} of {len(rank)}) onset={onset} explains_alert={can}")
            for h in hold[:15]:
                print(f"      {h}")
        for e, n in held.most_common(5):
            noisy[e] += 1
        print()
    print("entities among the 5 with most holding observations, over the wrong odd cases:", noisy.most_common(10))


if __name__ == "__main__":
    main(sys.argv[1])
