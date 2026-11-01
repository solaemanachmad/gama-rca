"""signal_onsets_dev.py -- v5 diagnostic for the idea 'which signal moved first' (intra-entity precedence). ODD CASES ONLY.

For every odd case whose true root is a service, it runs the statistical core with the parts of an arm and prints, at the TRUE root,
the holding calibrated signals ordered by onset (seconds relative to the start of the alert window), next to the true and the
predicted type. Question: is the mechanism signal of the true type (memory for memoryPressure, CPU for cpuFullLoad, threads for
threadExhaustion, ...) among the first to move, while the others follow? The answer key is read ONLY to name the true root and type
for this analysis; even cases are refused.

Usage (Kaggle):  !python scripts/diagnostics/signal_onsets_dev.py --arm V4_RULE_MM_T3_OMSG_FI > /kaggle/working/signal_onsets_odd.txt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="V4_RULE_MM_T3_OMSG_FI")
    ap.add_argument("--data", default="/kaggle/input/datasets/solaemanachmad/rca100v1-1/RCA100v1.1")
    ap.add_argument("--cases", default="")
    a = ap.parse_args()
    from scripts.kaggle_job import definitions
    D = definitions(a.data)
    os.environ["RCA100_ROOT"] = a.data
    for k, v in D["ARMS"][a.arm].items():
        if not k.startswith("_"):
            os.environ[k] = v
    import config
    from data.loader import Case, find_service_ancestor
    from evaluation.scoring import load_ground_truth
    from pipeline import belief_diagnosis as bd
    from pipeline.belief_rca import callee_map
    from pipeline.belief_v4 import diagnose_v4
    from pipeline.fault_calibrated import compute_signal_db
    from pipeline.propagation_evidence import _node_name

    parts = [p for p in os.environ["V4_PARTS"].split(",") if p != "llmtype"]
    cases = [c for c in (a.cases.split(",") if a.cases else D["DEV_ODD"]) if int(c[1:]) % 2 == 1]
    print(f"arm {a.arm} | {len(cases)} odd cases | onsets in s from the start of the alert window (negative = before it)\n")
    for cid in cases:
        case = Case(cid, cases_dir=config.CASES_DIR)
        gt = load_ground_truth(cid, name_index=case.name_index)
        true_names = [_node_name(case.topology, e) for e in gt.target_entity_ids if e in case.topology] or gt.root_cause_entities
        entry = case.alert.entry_entity_id
        sid = find_service_ancestor(entry, case.topology) if entry else None
        alert = _node_name(case.topology, sid) if sid else None
        db = compute_signal_db(case, case.alert.alert_timestamp)
        res = diagnose_v4(db, callee_map(case.topology), {}, sensor=None, budget=0, alert=alert, parts=parts, case=case)
        w0 = db["w0"].timestamp() if hasattr(db["w0"], "timestamp") else None
        for ent in true_names:
            if ent not in db["services"]:
                continue
            cal = (db.get("_cal") or {}).get(ent, {})
            held = sorted(((v["onset"] - w0, s) for s, v in cal.items()
                           if v.get("onset") is not None and v.get("p") is not None and v["p"] <= bd.ALPHA), key=lambda x: x[0])
            other = sorted(s for s, v in cal.items() if v.get("onset") is None and v.get("p") is not None and v["p"] <= bd.ALPHA)
            ok = "OK " if res["type"] == gt.fault_type else "   "
            print(f"{ok}{cid} true={ent} {gt.fault_type.split('-')[-1]:22s} predicted={res['entity']} {str(res['type']).split('-')[-1]}")
            print("     by onset: " + ", ".join(f"{s}@{t:+.0f}" for t, s in held))
            if other:
                print("     no onset: " + ", ".join(other))
            dep = [f"{o['signal']}" for o in bd.observations(db) if o["entity"] == ent and o["holds"] and o["kind"] == "cat"]
            if dep:
                print("     categorical: " + ", ".join(dep[:12]))


if __name__ == "__main__":
    main()
