"""entity_errors_dev.py -- v5 error analysis of entity localisation, DEVELOPMENT (ODD) CASES ONLY (PREREGISTRATION_V5.md).

For every odd case whose predicted entity is wrong in a results CSV, it re-runs the statistical core (no LLM) and prints, for the
PREDICTED and the TRUE entity: the observations that hold (signal: values), the onset, the hosting nodes, and whether the entity can
explain the alerted service; plus, for the whole case, the number of holding observations per entity (does one service always look
anomalous?). The answer key is read here ONLY to name the true entity for the analysis; the diagnosis itself never sees it.
Even cases are refused: they are the validation half and stay unseen until the v5 design is frozen.

Usage (on Kaggle, RCA100_ROOT set):  ALERT_T0_MODE=window python scripts/diagnostics/entity_errors_dev.py V4_dev.csv > entity_errors_odd.txt
  with the parts of an arm (sets its env first):  python scripts/diagnostics/entity_errors_dev.py <results.csv> --arm V4_RULE_MM_T3_OMSG_FI_ORD2
For every holding calibrated signal it also prints 'self a/b': how many of the entity's OWN b pseudo-windows (cut from its baseline)
scored at least as high as the real window -- a service that moves this much in its own quiet periods is noisy, not faulty.
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


def compare(dbg, db, pred_ent, true_ents):
    """Why does the predicted root beat the true one? Best hypothesis of each entity; per observation log P(o | h) under the same
    EM-propagation rule as the likelihood; the observations with the largest difference (positive = favours the predicted root)."""
    import math
    import numpy as np
    model, obs, bg, expl, H, logp, pi = (dbg.get(k) for k in ("model", "obs", "bg", "expl", "H", "logp", "pi"))
    if model is None:
        return
    hosts = db.get("hosts", {})

    def best(ent):
        idx = [i for i in range(len(H) - 1) if H[i]["entity"] == ent]
        return max(idx, key=lambda i: logp[i]) if idx else None

    def term(o, h):
        if pi is not None and model.propagated(o, h, expl, hosts):
            return math.log(pi if o["holds"] else 1 - pi)
        p = model.p_holds(o, h, bg, expl, hosts, db)
        if not o["holds"] and model.may(o, h, hosts):
            p = model.p_holds(o, H[-1], bg, expl, hosts, db)
        return math.log(p if o["holds"] else 1.0 - p)

    ip = best(pred_ent)
    for t in true_ents:
        it = best(t)
        if ip is None or it is None:
            print(f"  COMPARE: no hypothesis for {t}")
            continue
        hp, ht = H[ip], H[it]
        d = [(term(o, hp) - term(o, ht), o) for o in obs]
        ll = sum(x for x, _ in d)
        print(f"  COMPARE {hp['entity']}:{hp['type']} vs {ht['entity']}:{ht['type']}: log posterior diff "
              f"{logp[ip] - logp[it]:+.2f} (likelihood {ll:+.2f}, prior/onset/mm/other {logp[ip] - logp[it] - ll:+.2f})")
        for x, o in sorted(d, key=lambda z: -abs(z[0]))[:14]:
            if abs(x) < 0.3:
                break
            print(f"      {x:+6.2f}  {o['entity']} {o['signal']} {'holds' if o['holds'] else 'absent'}")


def main(csv):
    if csv == "odd":                                     # no results file: diagnose every odd case, print the wrong ones
        ids = [l.strip() for l in open(os.path.join(config.CASES_DIR, "..", "manifest.txt")) if l.strip()]
        wrong = pd.DataFrame({"case_id": [c for c in ids if int(c[1:]) % 2 == 1]})
        print(f"odd cases: {len(wrong)} (all diagnosed; correct entities are skipped)\n")
    else:
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
        dbg = {}
        res = diagnose_v4(db, callees, {}, sensor=None, budget=0, alert=alert, parts=PARTS, case=case, debug=dbg)
        if csv == "odd" and res["entity"] in true_names:
            continue
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
                sig_name = h.split(":", 1)[0]
                r = on.get(sig_name) or {}
                z, ps = r.get("z", r.get("raw")), r.get("pseudo") or []
                selfn = f"  [self {sum(1 for x in ps if x is not None and z is not None and x >= z)}/{len(ps)}]" if ps and z is not None else ""
                print(f"      {h[:200]}{selfn}")
        compare(dbg, db, res["entity"], [t for t in true_names if t])
        for e, n in held.most_common(5):
            noisy[e] += 1
        print()
    print("entities among the 5 with most holding observations, over the wrong odd cases:", noisy.most_common(10))


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--arm" in args:
        i = args.index("--arm")
        arm = args[i + 1]
        args = args[:i] + args[i + 2:]
        from scripts.kaggle_job import definitions
        D = definitions(os.environ.get("RCA100_ROOT", "/kaggle/input/datasets/solaemanachmad/rca100v1-1/RCA100v1.1"))
        for k, v in D["ARMS"][arm].items():
            if not k.startswith("_"):
                os.environ[k] = v
        PARTS = [p for p in os.environ["V4_PARTS"].split(",") if p != "llmtype"]
        print(f"arm {arm}: parts {','.join(PARTS)}")
    main(args[0])
