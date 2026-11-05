"""
logpre_check.py -- LABEL-FREE check of the v3 log preprocessing (pipeline/log_preprocess.py, Amendment 2) on real RCA100 cases.
Never reads answer_key/ and prints no score. CPU only (no model).

Per case: log lines, templates in the window, significant templates, entities that become probe-able, seconds; and the same on a
PSEUDO window (the last window length of the log baseline, tested against the earlier baseline), where nothing was injected:
significant templates there are false alarms. BH at q=0.05 bounds the chance of any false alarm per case at ~5% if the Poisson model
holds; bursty logs (overdispersion) would show up here as a higher rate.

Usage (RCA100_ROOT set, ALERT_T0_MODE=window):  python scripts/diagnostics/logpre_check.py [--csv out.csv] [t001 t002 ...]  (default: all)
"""
import datetime as dt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from data.loader import Case
from pipeline.fault_calibrated import compute_signal_db
from pipeline.log_preprocess import preprocess_texts


def all_ids():
    root = config.DATASET_ROOT if hasattr(config, "DATASET_ROOT") else os.environ["RCA100_ROOT"]
    return [l.strip() for l in open(os.path.join(root, "manifest.txt")) if l.strip()]


def main(argv):
    csv_path = None
    if "--csv" in argv:
        i = argv.index("--csv")
        csv_path, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    ids = [a for a in argv if not a.startswith("--")] or all_ids()
    rows = []
    for cid in ids:
        t0 = time.time()
        try:
            case = Case(cid, cases_dir=config.CASES_DIR)
            db = compute_signal_db(case, case.alert.alert_timestamp)
            texts, notes, st = preprocess_texts(case, db)
            secs = time.time() - t0
            tw = db["w1"] - db["w0"]
            pseudo = dict(db, w0=db["w0"] - tw, w1=db["w0"] - dt.timedelta(microseconds=1))
            _, _, sp = preprocess_texts(case, pseudo)
            n_log_ent = sum(1 for n in notes.values() if any(x.startswith("log template") for x in n))
            rows.append({"case_id": cid, "seconds": round(secs, 1), "log_lines": st["logpre_lines"],
                         "templates_window": st["logpre_templates"], "significant": st["logpre_significant"],
                         "untestable": st.get("logpre_untestable", 0), "baseline_min": st["logpre_baseline_min"],
                         "probe_entities": st["logpre_entities"], "entities_with_log_items": n_log_ent,
                         "pseudo_templates": sp["logpre_templates"], "pseudo_significant": sp["logpre_significant"],
                         "pseudo_untestable": sp.get("logpre_untestable", 0)})
        except Exception as e:
            rows.append({"case_id": cid, "error": repr(e)[:200]})
        r = rows[-1]
        print(cid, {k: v for k, v in r.items() if k != "case_id"}, flush=True)
    ok = [r for r in rows if "error" not in r]
    if ok:
        tested = [r for r in ok if not r["pseudo_untestable"]]
        print("\n=== summary ===")
        print(f"cases {len(ok)} (errors {len(rows) - len(ok)}); seconds/case mean {sum(r['seconds'] for r in ok) / len(ok):.1f}, "
              f"max {max(r['seconds'] for r in ok):.1f}")
        print(f"real window: significant templates/case mean {sum(r['significant'] for r in ok) / len(ok):.1f}; "
              f"untestable cases {sum(r['untestable'] for r in ok)}; probe-able entities/case mean "
              f"{sum(r['probe_entities'] for r in ok) / len(ok):.1f}")
        if tested:
            fa = sum(1 for r in tested if r["pseudo_significant"] > 0)
            print(f"PSEUDO window (no fault): cases with >=1 significant template {fa}/{len(tested)} = {fa / len(tested):.1%} "
                  f"(target <= ~5%); significant/tested templates {sum(r['pseudo_significant'] for r in tested)}/"
                  f"{sum(r['pseudo_templates'] for r in tested)}")
    if csv_path:
        import csv
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "case_id", k))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("saved", csv_path)


if __name__ == "__main__":
    main(sys.argv[1:])
