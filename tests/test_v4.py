"""tests/test_v4.py -- synthetic acceptance tests of the v4 evidence layer and diagnosis (pipeline/fault_v4.py, pipeline/belief_v4.py).
No data, no model. Scenarios come from the failure mechanisms in claude/RCA_v4_design.md, built by hand, not from RCA100 cases.

    python tests/test_v4.py

Fake topology (test_fingerprint.FakeCase): frontend -> checkout -> cart; node-a hosts cart and frontend, node-b hosts checkout; plus 8 extra
nodes and 8 extra services with ordinary telemetry (test_calibrated.make_case)."""
import datetime as dt
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from test_fingerprint import T0
from test_calibrated import make_case, metric, _obs, W0
from pipeline.fault_calibrated import compute_signal_db, entity_signals
from pipeline.fault_v4 import shift_z, family_v4, calibrate_v4, EXCLUDE_NAME
from pipeline import belief_v4 as b4
from pipeline import belief_diagnosis as bd

CALLEES = {"frontend": ["checkout"], "checkout": ["cart"]}
T = []


def test(fn):
    T.append(fn)
    return fn


def series(c, eset, name, m, f, n_base=110, keep=1.0, seed=0):
    """Custom series: value f(i) at 30 s steps; i < n_base is baseline (before W0); keep = share of baseline points kept."""
    rnd = random.Random(seed)
    for i in range(n_base + 12):
        if i < n_base and rnd.random() > keep:
            continue
        t = W0 - dt.timedelta(seconds=30 * (n_base - i))
        c.observations["metrics"].append(_obs(t, eset, name, m, f(i)))


def run(c, parts=b4.ALL_PARTS, alert="frontend"):
    db = compute_signal_db(c, T0)
    assert "error" not in db, db
    return b4.diagnose_v4(db, CALLEES, {}, sensor=None, budget=0, alert=alert, parts=parts), db


@test
def shift_size_is_scored_not_only_order():
    base = [(float(i * 30), 1.0 + 0.01 * (i % 3)) for i in range(48)]
    drift = base + [(48 * 30.0 + j * 30, 1.05 + 0.001 * j) for j in range(12)]       # small but consistent: rank-extreme
    spike = base + [(48 * 30.0 + j * 30, 5.0) for j in range(12)]
    a, b, span = 48 * 30.0, 60 * 30.0, 48 * 30.0
    zd, zs = shift_z(drift, a, b, span, "up"), shift_z(spike, a, b, span, "up")
    assert zs > 10 * zd > 0, (zd, zs)


@test
def sparse_series_is_not_tested():
    pts = [(float(i * 30), 1.0) for i in range(0, 48, 6)] + [(48 * 30.0 + j * 30, 5.0) for j in range(12)]   # 1/6 of the history
    assert shift_z(pts, 48 * 30.0, 60 * 30.0, 48 * 30.0, "up") is None


@test
def constants_and_injectors_excluded():
    assert family_v4("arms_jvm_mem_max_bytes") is None and family_v4("arms_jvm_mem_init_bytes") is None
    assert family_v4("arms_jvm_mem_used_bytes") == "mem" and family_v4("deployment_availability_rate") == "avail"
    assert family_v4("deployment_desired_replicas") is None and family_v4("slow") == "latency"
    assert EXCLUDE_NAME.search("l3reg-node-cpu-stress") and EXCLUDE_NAME.search("chaosblade-operator") and not EXCLUDE_NAME.search("cart")


@test
def strong_node_cpu_detected_despite_drifting_null():
    c = make_case()
    for i in range(8):                                                   # other nodes drift slowly all the time (rank-extreme windows)
        series(c, "k8s.node", f"node-x{i}", "node_cpu_usage_rate", lambda k, i=i: 0.2 + 0.002 * k + 0.0005 * i, seed=i)
    series(c, "k8s.node", "node-a", "node_cpu_usage_rate", lambda k: 0.3 + 0.01 * (k % 3) if k < 110 else 0.95)
    db = compute_signal_db(c, T0)
    cal = calibrate_v4(db)
    assert cal["node-a"]["node_cpu_rise"]["p"] <= 0.05, cal["node-a"]["node_cpu_rise"]
    assert all(cal[f"node-x{i}"]["node_cpu_rise"]["p"] > 0.05 for i in range(8))


@test
def sparse_node_gives_no_signal():
    c = make_case()
    series(c, "k8s.node", "node-b", "node_memory_usage_rate", lambda k: 0.5 if k < 110 else 0.9, keep=0.2, seed=3)
    db = compute_signal_db(c, T0)
    cal = calibrate_v4(db)
    assert "node_mem_rise" not in cal.get("node-b", {}), cal.get("node-b")


@test
def node_cpu_fault_is_answered_at_the_node():
    c = make_case()
    series(c, "k8s.node", "node-a", "node_cpu_usage_rate", lambda k: 0.3 + 0.01 * (k % 3) if k < 110 else 0.95)
    metric(c, "apm.service.legacy", "cart", "rt", 0.1, 0.3, seed=5)
    metric(c, "apm.service.legacy", "frontend", "rt", 0.1, 0.3, seed=6)
    r, _ = run(c)
    assert (r["entity"], r["type"]) == ("node-a", "F026-nodeCpuHigh"), r["top"]


def ordinary(c, eset, m, base, seed0, noise=0.05):
    """The same metric, ordinary behaviour, for the 8 extra services (RCA100: every service has deployment / APM / memory series)."""
    for i in range(8):
        metric(c, eset, f"svc{i}", m, base, base, noise=noise, seed=seed0 + i)


@test
def silent_scaled_down_callee_beats_its_caller():
    c = make_case()
    ordinary(c, "k8s.deployment", "deployment_available_replicas", 3.0, 500, noise=0.0)
    ordinary(c, "apm.service.legacy", "workload", 100.0, 520)
    ordinary(c, "apm.service.legacy", "error", 1.0, 540, noise=0.5)
    series(c, "k8s.deployment", "cart", "deployment_available_replicas", lambda k: 3.0 if k < 110 else 0.0)
    series(c, "apm.service.legacy", "cart", "workload", lambda k: 100.0 + (k % 5) if k < 110 else 0.0)
    metric(c, "apm.service.legacy", "checkout", "error", 1.0, 40.0, seed=7)
    metric(c, "apm.service.legacy", "checkout", "rt", 0.1, 0.5, seed=8)
    metric(c, "apm.service.legacy", "frontend", "rt", 0.1, 0.5, seed=9)
    r, _ = run(c)
    assert (r["entity"], r["type"]) == ("cart", "F036-replicaScaleDown"), r["top"]


@test
def memory_pressure_beats_cpu_when_heap_and_gc_rise():
    c = make_case()
    ordinary(c, "apm.service.legacy", "mem", 0.5, 560)
    ordinary(c, "apm.service.legacy", "cpu", 0.3, 580)
    series(c, "apm.metric.jvm", "cart", "arms_jvm_mem_used_bytes", lambda k: 4e8 + 2e7 * (k % 4) if k < 110 else 9.5e8)
    series(c, "apm.metric.jvm", "cart", "arms_jvm_mem_max_bytes", lambda k: 1e9)
    series(c, "apm.metric.jvm", "cart", "arms_jvm_gc_delta", lambda k: 2.0 + (k % 2) if k < 110 else 20.0)
    metric(c, "apm.service.legacy", "cart", "cpu", 0.3, 0.7, seed=10)
    r, _ = run(c, alert="cart")
    assert (r["entity"], r["type"]) == ("cart", "F007-memoryPressure"), r["top"]


@test
def single_service_jvm_signal_can_reach_alpha():
    c = make_case()                                                     # GC series exist for cart only (RCA100: only inventory is Java)
    c.observations["metrics"] = [o for o in c.observations["metrics"] if o.payload["metric"] != "arms_jvm_gc_delta"]
    ordinary(c, "apm.service.legacy", "cpu", 0.3, 600)
    series(c, "apm.metric.jvm", "cart", "arms_jvm_gc_delta", lambda k: 2.0 + (k % 2) if k < 110 else 30.0)
    db = compute_signal_db(c, T0)
    cal = calibrate_v4(db)
    g = cal["cart"]["svc_gc_rise"]
    assert g.get("pool") == "kind" and g["p"] <= 0.05, g


@test
def unfingerprinted_anomaly_is_still_localised():
    c = make_case()
    metric(c, "apm.service.legacy", "cart", "rt", 0.1, 0.9, seed=11)      # latency only: a symptom, key of no specific type
    r_open, _ = run(c, alert="cart")
    assert r_open["entity"] == "cart", r_open["top"]


@test
def v3_engine_untouched_when_v4_unused():
    c = make_case()
    for i in range(6):
        c.span("checkout", W0 + dt.timedelta(seconds=10 + 5 * i), err=True, code="429", msg="Too Many Requests")
    db = compute_signal_db(c, T0)
    r3 = bd.diagnose_belief(db, CALLEES, {}, sensor=None, budget=0, alert="frontend")
    assert (r3["entity"], r3["type"]) == ("checkout", "F016-rateLimiting"), r3["top"]
    r4, _ = run(c)
    assert (r4["entity"], r4["type"]) == ("checkout", "F016-rateLimiting"), r4["top"]


@test
def ablation_parts_switch_independently():
    c = make_case()
    metric(c, "apm.service.legacy", "cart", "rt", 0.1, 0.9, seed=12)
    for parts in ([], ["score"], ["open"], ["twostage"], ["fp"], ["score", "open", "twostage", "fp"]):
        r, _ = run(c, parts=parts, alert="cart")
        assert r["parts"] == sorted(parts) and r["type"] is not None, (parts, r)


@test
def end_to_end_v4_with_stub_llm():
    from pipeline.belief_rca import run_case
    from test_hypothesis_rca import case as v2case, Stub
    from test_calibrated import wspan

    class SensorStub(Stub):
        def generate_json(self, prompt, system=None):
            if "Which ONE category" in prompt:
                return {"category": "ratelimit", "quote": "Too Many Requests"}
            return super().generate_json(prompt, system)
    c = v2case()
    for i in range(6):
        wspan(c, "checkout", i, err=True, code="429", msg="Too Many Requests")
    r = run_case(c, SensorStub(), budget=5, narrate=True, v4=True)
    assert r.predicted_fault_type == "F016-rateLimiting" and r.reasoning_chain, (r.predicted_fault_type, r.retrieval_stats)
    assert r.retrieval_stats["v4_parts"] == "fp,open,score,twostage", r.retrieval_stats
    r0 = run_case(c, None, budget=0, narrate=False, v4=True)
    assert r0.predicted_fault_type == "F016-rateLimiting" and r0.reasoning_chain


if __name__ == "__main__":
    bad = 0
    for fn in T:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:
            bad += 1
            print("FAIL", fn.__name__, repr(e)[:600])
    print(f"{len(T) - bad}/{len(T)} passed")
    sys.exit(1 if bad else 0)
