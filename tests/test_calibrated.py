"""tests/test_calibrated.py -- synthetic checks of the calibrated fingerprint layer (pipeline/fault_calibrated.py). No data, no model.

    python tests/test_calibrated.py

Telemetry geometry mirrors RCA100 (check_baseline_length.py): metrics every 30 s with ~55 min of baseline, alert window 6 min,
traces only inside the window, logs with ~15 min of baseline."""
import datetime as dt
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from test_fingerprint import FakeCase, T0
from pipeline.fault_calibrated import (compute_signal_db, origin_signals, rank_candidates, mwu_greater, false_alarm_rate)

W0 = T0 - dt.timedelta(minutes=5)


def metric(c, eset, name, metric_name, base, win, noise=0.05, seed=0):
    rnd = random.Random(seed)
    for i in range(110):                                   # baseline: 55 min at 30 s
        t = W0 - dt.timedelta(seconds=30 * (110 - i))
        c.observations["metrics"].append(_obs(t, eset, name, metric_name, base * (1 + rnd.uniform(-noise, noise))))
    for i in range(12):                                    # window: 6 min at 30 s
        t = W0 + dt.timedelta(seconds=30 * i)
        c.observations["metrics"].append(_obs(t, eset, name, metric_name, win * (1 + rnd.uniform(-noise, noise))))


def _obs(t, eset, name, m, v):
    from schema import Observation
    return Observation(None, t, "metrics", "", {"entity_set": eset, "entity_name": name, "metric": m, "value": v})


def wspan(c, svc, i, **kw):
    c.span(svc, W0 + dt.timedelta(seconds=10 + 5 * i), **kw)


def types(c, origin):
    db = compute_signal_db(c, T0)
    assert "error" not in db, db
    osig = origin_signals(db, origin)
    return [(x["type"], x["entity"]) for x in rank_candidates(osig, top_k=0)], osig, db


T = []


def test(fn):
    T.append(fn)
    return fn


@test
def mwu_sanity():
    assert mwu_greater([10, 11, 12, 13], [1, 2, 3, 4, 5]) < 0.05
    assert mwu_greater([1, 2, 3, 4], [10, 11, 12, 13, 14]) > 0.9
    assert mwu_greater([1, 1, 1], [1, 1, 1]) >= 0.5
    assert mwu_greater([1], [1, 2, 3]) is None


@test
def node_cpu_hypothesis_on_host():
    c = FakeCase()
    metric(c, "k8s.node", "node-a", "node_cpu_usage_rate", 0.30, 0.95)
    metric(c, "k8s.node", "node-b", "node_cpu_usage_rate", 0.30, 0.30, seed=1)
    h, _, _ = types(c, "node-a")
    assert h[0] == ("F026-nodeCpuHigh", "node-a"), h
    h2, _, _ = types(c, "cart")                          # cart is hosted on node-a: the node hypothesis names the node
    assert ("F026-nodeCpuHigh", "node-a") in h2, h2
    h3, _, _ = types(c, "node-b")
    assert h3 == [], h3


@test
def stable_high_metric_does_not_hold():
    c = FakeCase()
    metric(c, "k8s.node", "node-a", "node_memory_usage_rate", 0.92, 0.92)   # always high = normal for this node
    h, osig, _ = types(c, "node-a")
    assert h == [] and not osig["signals"], (h, osig["signals"])


@test
def rate_limit_categorical_beats_metric_symptom():
    c = FakeCase()
    metric(c, "apm.service.legacy", "checkout", "rt", 0.1, 0.4)              # latency symptom
    for i in range(6):
        wspan(c, "checkout", i, err=True, code="429", msg="Too Many Requests")
    h, osig, _ = types(c, "checkout")
    assert h[0][0] == "F016-rateLimiting", (h, osig["signals"])


@test
def gc_rise_vs_ordinary_gc():
    c = FakeCase()
    metric(c, "apm.metric.jvm", "checkout", "arms_jvm_gc_delta", 2.0, 2.0, noise=0.5)   # bursty but unchanged
    h, _, _ = types(c, "checkout")
    assert all(t != "F022-fullGC" for t, _ in h), h
    c2 = FakeCase()
    metric(c2, "apm.metric.jvm", "checkout", "arms_jvm_gc_delta", 2.0, 30.0, noise=0.5)
    h2, _, _ = types(c2, "checkout")
    assert h2[0][0] == "F022-fullGC", h2


@test
def scale_down_excludes_flapping():
    c = FakeCase()
    c.event("Deployment", "cart", "ScalingReplicaSet", "Scaled down replica set cart-7d9f8c6b5d to 0", when=W0 + dt.timedelta(minutes=1))
    c.event("Pod", "cart-7d9f8c6b5d-x2x9k", "Killing", "Stopping container cart", when=W0 + dt.timedelta(minutes=1))
    h, _, _ = types(c, "cart")
    assert h[0][0] == "F036-replicaScaleDown" and all(t != "F052-podRestartFlapping" for t, _ in h), h


@test
def events_outside_window_ignored():
    c = FakeCase()
    c.event("Pod", "cart-7d9f8c6b5d-x2x9k", "BackOff", "Back-off restarting", when=W0 - dt.timedelta(minutes=40))
    h, _, _ = types(c, "cart")
    assert h == [], h


@test
def slow_db_within_window():
    c = FakeCase()
    for i in range(10):
        wspan(c, "checkout", i, name="GET /order", dur_ns=20e6)
        wspan(c, "checkout", i, name="SELECT orders", attrs='"db.system":"postgresql",', dur_ns=2500e6)
    h, _, _ = types(c, "checkout")
    assert h[0][0] in ("F010-slowSQL", "F018-dbNetworkLatency"), h


@test
def residual_only_when_nothing_specific():
    c2 = FakeCase()
    for i in range(5):
        wspan(c2, "checkout", i, err=True, code="500", msg="internal")
    h2, _, _ = types(c2, "checkout")
    assert h2 and h2[0][0] == "F014-httpError5xx", h2


@test
def false_alarm_rate_near_alpha():
    c = FakeCase()
    for k in range(40):                                   # 40 unchanged series: the pseudo-window test should rarely hold
        metric(c, "k8s.node", "node-a", f"m{k}_usage_rate", 0.4, 0.4, noise=0.2, seed=k)
    db = compute_signal_db(c, T0)
    far = false_alarm_rate(db)
    assert far["series"] == 40 and far["rate"] <= 0.15, far


@test
def typing_uses_hypothesis_entity_with_level_switch():
    from agents import multi_agent as ma
    c = FakeCase()
    metric(c, "k8s.node", "node-a", "node_cpu_usage_rate", 0.30, 0.95)

    class Stub:
        def generate_json(self, prompt, system=None):
            return {"fault_evidence": "node cpu", "predicted_fault_type": "F026-nodeCpuHigh", "confidence": 0.8}

    saved = (config.FAULT_FINGERPRINT, config.FP_SCORING, config.FP_LEVEL_SWITCH, config.ENTITY_TYPING)
    try:
        config.FAULT_FINGERPRINT, config.FP_SCORING, config.FP_LEVEL_SWITCH, config.ENTITY_TYPING = "shortlist", "calibrated", True, False
        st = {"alert_text": "a", "fp_db": compute_signal_db(c, T0), "evidence_summary": {},
              "final_result": {"reasoning_chain": ["cause: cart latency=2", "impact: frontend latency=3"], "predicted_entity_ids": ["cart"],
                               "predicted_fault_type": "F014-httpError5xx"}}
        ma._apply_entity_typing(Stub(), st, "taxonomy")
        fr = st["final_result"]
        assert fr["predicted_fault_type"] == "F026-nodeCpuHigh" and fr["predicted_entity_ids"] == ["node-a"], fr
    finally:
        config.FAULT_FINGERPRINT, config.FP_SCORING, config.FP_LEVEL_SWITCH, config.ENTITY_TYPING = saved


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
