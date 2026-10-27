"""tests/test_causal.py -- synthetic propagation scenarios for the causal attribution (pipeline/causal_attribution.py). No data, no model.

    python tests/test_causal.py

Fake topology: frontend -> checkout -> cart (calls); node-a hosts cart and frontend, node-b hosts checkout. Series follow a linear
structural model with baseline noise, sampled every 30 s for 55 min of baseline and a 6 min window, like RCA100."""
import datetime as dt
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from schema import Observation
from test_fingerprint import T0
from test_calibrated import make_case
from pipeline.fault_calibrated import compute_signal_db
from pipeline.causal_attribution import attribute

W0 = T0 - dt.timedelta(minutes=5)
CALLEES = {"frontend": ["checkout"], "checkout": ["cart"]}
HOSTS = {"cart": ["node-a"], "frontend": ["node-a"], "checkout": ["node-b"]}


def put(c, eset, name, metric, ts, vals):
    for t, v in zip(ts, vals):
        c.observations["metrics"].append(Observation(None, t, "metrics", "", {"entity_set": eset, "entity_name": name, "metric": metric,
                                                                             "value": v}))


def scenario(shift_cart=0.0, node_cpu_win=None, seed=0):
    """cart latency = 0.1 + 0.4*cpu(node-a) + noise (+ shift in the window); checkout = 0.05 + cart + noise; frontend = 0.02 +
    checkout + noise. Request counts are exogenous noise."""
    rnd = random.Random(seed)
    c = make_case()
    c.case_id = "t999"
    ts = [W0 - dt.timedelta(seconds=30 * (110 - i)) for i in range(110)] + [W0 + dt.timedelta(seconds=30 * i) for i in range(12)]
    win = [t >= W0 for t in ts]
    cpu_a = [(node_cpu_win if (w and node_cpu_win) else 0.3) + rnd.gauss(0, 0.05) for w in win]
    cpu_b = [0.3 + rnd.gauss(0, 0.05) for _ in ts]
    cart = [0.1 + 0.4 * u + rnd.gauss(0, 0.01) + (shift_cart if w else 0.0) for u, w in zip(cpu_a, win)]
    chk = [0.05 + x + rnd.gauss(0, 0.01) for x in cart]
    fe = [0.02 + x + rnd.gauss(0, 0.01) for x in chk]
    put(c, "k8s.node", "node-a", "node_cpu_usage_rate", ts, cpu_a)
    put(c, "k8s.node", "node-b", "node_cpu_usage_rate", ts, cpu_b)
    for s, lat in (("cart", cart), ("checkout", chk), ("frontend", fe)):
        put(c, "apm.service.legacy", s, "rt", ts, lat)
        put(c, "apm.service.legacy", s, "request_count", ts, [100 + rnd.gauss(0, 5) for _ in ts])
    db = compute_signal_db(c, T0)
    assert "error" not in db, db
    return attribute(db, CALLEES, HOSTS, "frontend")


T = []


def test(fn):
    T.append(fn)
    return fn


@test
def origin_service_beats_its_victims():
    r = scenario(shift_cart=0.4)
    top = r["ranking"][0]
    assert top["entity"] == "cart" and top["own_anomalous"] and top["share"] > 0.6, r["ranking"][:3]
    v = {k[0]: x for k, x in r["vars"].items() if k[1] == "latency"}
    assert v["frontend"]["p_raw"] <= 0.05, "the alerted service is anomalous before explaining away"
    assert v["frontend"]["p_own"] > 0.05 and v["checkout"]["p_own"] > 0.05, (v["frontend"]["p_own"], v["checkout"]["p_own"])


@test
def node_origin_is_found_through_hosted_service():
    r = scenario(node_cpu_win=0.95)
    top = r["ranking"][0]
    assert top["entity"] == "node-a" and top["own_anomalous"], r["ranking"][:3]
    v = r["vars"][("cart", "latency")]
    assert v["p_raw"] <= 0.05 and v["p_own"] > 0.05, (v["p_raw"], v["p_own"])


@test
def no_fault_no_own_anomaly():
    r = scenario(seed=3)
    own = [k for k, v in r["vars"].items() if v["p_own"] is not None and v["p_own"] <= 0.05 and k[0] in ("cart", "checkout", "frontend", "node-a")]
    assert len(own) <= 1, own


if __name__ == "__main__":
    bad = 0
    for fn in T:
        try:
            fn()
            print("ok  ", fn.__name__)
        except AssertionError as e:
            bad += 1
            print("FAIL", fn.__name__, e)
    print(f"{len(T) - bad}/{len(T)} passed")
    sys.exit(1 if bad else 0)
