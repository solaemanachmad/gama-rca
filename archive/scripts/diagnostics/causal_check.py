"""
causal_check.py -- LABEL-FREE check of the causal attribution prototype (pipeline/causal_attribution.py) on real RCA100 cases.
Never reads answer_key/ and prints no score. Answers: are the needed series there (latency / error / request per service, CPU per
node), how well does the baseline model predict held-out baseline data, is the residual test calibrated (false alarms ~ alpha),
and how concentrated is the attribution (how many entities are own-anomalous, share of the top entity).

Usage (RCA100_ROOT set):  python scripts/diagnostics/causal_check.py [--csv out.csv] [t004 t005 ...]   (default: dev-30)
Set ALERT_T0_MODE=window to use the benchmark's alert_window (the v2 setting).
"""
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from data.loader import Case, find_service_ancestor
from pipeline.fault_calibrated import compute_signal_db
from pipeline.propagation_evidence import _calls_graph, _node_name, _node_type
from pipeline import causal_attribution as ca

DEV30 = ["t004", "t005", "t012", "t014", "t015", "t018", "t026", "t028", "t029", "t030", "t032", "t036", "t054", "t055", "t058",
         "t065", "t070", "t072", "t076", "t078", "t082", "t087", "t091", "t092", "t093", "t095", "t097", "t099", "t100", "t101"]


def graph(case, db):
    topo = case.topology
    callees = defaultdict(list)
    for u, v in _calls_graph(topo).edges():
        if _node_type(topo, u) in ("apm.service", "apm.svc") and _node_type(topo, v) in ("apm.service", "apm.svc"):
            a, b = _node_name(topo, u), _node_name(topo, v)
            if a != b and b not in callees[a]:
                callees[a].append(b)
    return dict(callees), db.get("hosts", {})


def holdout_r2(target, parents, var):
    """Fit on the first 70% of the baseline bins, R^2 on the last 30% (baseline only)."""
    bins = sorted(b for b in var[target] if b < 0)
    cut = int(len(bins) * 0.7)
    y, X, _ = ca._rows(target, parents, var, bins[:cut])
    yt, Xt, _ = ca._rows(target, parents, var, bins[cut:])
    if len(y) < ca.MIN_FIT or len(yt) < 5 or not parents:
        return None
    pred, _ = ca._fit(y, X)
    mu, _ = pred(Xt)
    ss = float(np.sum((yt - yt.mean()) ** 2))
    return None if ss == 0 else 1 - float(np.sum((yt - mu) ** 2)) / ss


def main(argv):
    csv_path = None
    if "--csv" in argv:
        i = argv.index("--csv")
        csv_path, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    ids = [a for a in argv if not a.startswith("--")] or DEV30
    rows, names_seen = [], defaultdict(Counter)
    far_n = far_f = 0
    for cid in ids:
        t0 = time.time()
        try:
            case = Case(cid)
            db = compute_signal_db(case, case.alert.alert_timestamp)
            callees, hosts = graph(case, db)
            entry = case.alert.entry_entity_id
            sid = find_service_ancestor(entry, case.topology) if entry else None
            alert = _node_name(case.topology, sid) if sid else None
            r = ca.attribute(db, callees, hosts, alert)
        except Exception as e:
            print(f"{cid}: ERROR {e!r}")
            continue
        var = ca.build_variables(db)[0]
        for k, n in r["names"].items():
            names_seen[k[1]][n] += 1
        n_svc = len(db["services"])
        have = Counter(k[1] for k in r["vars"])
        r2 = [x for x in (holdout_r2(t, v["parents"], var) for t, v in r["vars"].items() if t[1] == "latency" and v["parents"]) if x is not None]
        sk = [v["skill"] for t, v in r["vars"].items() if t[1] in ("latency", "error") and v.get("skill") is not None]
        with_par = sum(1 for t, v in r["vars"].items() if t[1] in ("latency", "error") and v["parents"])
        # calibration: pseudo-window 1 against the null of pseudo-windows 2..K, per variable kind
        pools = defaultdict(list)
        for t, v in r["vars"].items():
            pools[t[1]].extend(x for x in v["own"][2:] if x is not None)
        for t, v in r["vars"].items():
            x = v["own"][1] if len(v["own"]) > 1 else None
            if x is not None and pools[t[1]]:
                far_n += 1
                far_f += (1 + sum(1 for y in pools[t[1]] if y >= x)) / (len(pools[t[1]]) + 1) <= ca.ALPHA
        own = sorted({t[0] for t, v in r["vars"].items() if v["p_own"] is not None and v["p_own"] <= ca.ALPHA})
        raw = sorted({t[0] for t, v in r["vars"].items() if v["p_raw"] is not None and v["p_raw"] <= ca.ALPHA})
        top = r["ranking"][:3]
        row = dict(case=cid, alert=alert, target=f"{r['target'][0]}:{r['target'][1]}" if r["target"] else "",
                   services=n_svc, with_latency=have["latency"], with_error=have["error"], with_request=sum(1 for k in var if k[1] == "request"),
                   nodes_cpu=have["cpu"], with_parents=with_par, skill_median=round(float(np.median(sk)), 2) if sk else None,
                   r2_median=round(float(np.median(r2)), 2) if r2 else None, n_raw_anom=len(raw), n_own_anom=len(own),
                   top=" | ".join(f"{x['entity']} {x['share']:.2f} p={x['p_own'] if x['p_own'] is None else round(x['p_own'], 3)}" for x in top),
                   secs=round(time.time() - t0))
        rows.append(row)
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    if not rows:
        return
    med = lambda k: float(np.median([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None
    print(f"\nSUMMARY {len(rows)} cases (label-free)")
    print(f"  services per case (median) {med('services')}, with latency {med('with_latency')}, error {med('with_error')}, request {med('with_request')}; nodes with cpu {med('nodes_cpu')}")
    print(f"  latency/error variables with selected parents (median per case): {med('with_parents')}; selection skill vs intercept-only (median): {med('skill_median')}")
    print(f"  baseline hold-out R^2 of latency models with parents (median of case medians): {med('r2_median')}")
    print(f"  calibration: pseudo-window false-alarm rate {far_f}/{far_n} = {far_f / max(far_n, 1):.3f} (target {ca.ALPHA})")
    print(f"  anomalous entities per case before / after explaining away (median): {med('n_raw_anom')} / {med('n_own_anom')}")
    print(f"  cases with no own-anomalous entity: {sum(1 for r in rows if r['n_own_anom'] == 0)}")
    for kind, c in names_seen.items():
        print(f"  metric names used for {kind}: {dict(c.most_common(5))}")
    if csv_path:
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("written", csv_path)


if __name__ == "__main__":
    main(sys.argv[1:])
