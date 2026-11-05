"""tests/test_retype.py -- scripts/retype.py on synthetic state (no data, no model): pickling both signal databases, the rule arms
end-to-end, and that the re-typed case equals calling the typing step inside the pipeline.    python tests/test_retype.py"""
import csv
import json
import os
import pickle
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import config
from test_fingerprint import T0
from test_calibrated import make_case, wspan
from scripts import retype
from pipeline import fault_calibrated, fault_fingerprint


def main():
    c = make_case()
    for i in range(6):
        wspan(c, "checkout", i, err=True, code="429", msg="Too Many Requests")
    tmp = tempfile.mkdtemp()
    for s, mod in (("calibrated", fault_calibrated), ("fixed", fault_fingerprint)):
        db = mod.compute_signal_db(c, T0)
        if s == "calibrated":
            fault_calibrated._calibrate(db)
        os.makedirs(os.path.join(tmp, f"fpdb_{s}"))
        with open(os.path.join(tmp, f"fpdb_{s}", "t999.pkl"), "wb") as f:
            pickle.dump(retype._plain(db), f)                        # fixed db has lambda defaultdicts: must pickle after _plain
    fr = {"reasoning_chain": ["cause: checkout http_429=6", "impact: frontend error_count=6"], "predicted_entity_ids": ["checkout"],
          "predicted_fault_type": "F014-httpError5xx"}
    json.dump({"case_id": "t999", "alert_text": "a", "evidence_summary": {}, "propagation_text": "", "infra_text": "", "trace_text": "",
               "final_result": fr}, open(os.path.join(tmp, "t999.json"), "w"))
    ref = os.path.join(tmp, "ref.csv")
    cols = ["case_id", "entity_localization", "reasoning_process", "reasoning_process_chain", "rcl_strict", "rcl_exact",
            "predicted_fault_type", "gt_fault_type", "error"]
    with open(ref, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow(dict(case_id="t999", entity_localization="1.0", reasoning_process="0.5", reasoning_process_chain="0.4", rcl_strict="1",
                        rcl_exact="1", predicted_fault_type="F014-httpError5xx", gt_fault_type="F016-rateLimiting", error=""))
    gt = types.SimpleNamespace(fault_type="F016-rateLimiting")
    import evaluation.scoring as sc
    sc.load_ground_truth = lambda cid, *a, **k: gt
    sc.fault_identification_tiered = lambda p, g: None
    for arm in ("A1rT_fixed_ABS_KP_FPRULE", "A1rT_fixed_ABS_KP_FPRULE_FIX"):
        out = os.path.join(tmp, f"{arm}.csv")
        retype.run(types.SimpleNamespace(ref=ref, state=tmp, arm=arm, out=out))
        r = list(csv.DictReader(open(out)))[0]
        assert r["predicted_fault_type"] == "F016-rateLimiting", r
        assert float(r["fault_identification_exact"]) == 1.0 and abs(float(r["final_score"]) - (0.4 + 0.3 + 0.15)) < 1e-6, r
    # equivalence with the in-pipeline call on the same state
    from agents import multi_agent as ma
    retype._set_flags("rule", "calibrated")
    st = {"alert_text": "a", "evidence_summary": {}, "final_result": dict(fr), "fp_db": fault_calibrated.compute_signal_db(c, T0)}
    ma._apply_entity_typing(None, st, "taxonomy")
    assert st["final_result"]["predicted_fault_type"] == "F016-rateLimiting"
    print("retype: 3/3 passed")


if __name__ == "__main__":
    saved = {k: getattr(config, k) for k in ("FAULT_FINGERPRINT", "FP_SCORING", "ENTITY_TYPING", "USE_KNOWLEDGE_POS")}
    try:
        main()
    finally:
        for k, v in saved.items():
            setattr(config, k, v)
