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


def run(c, parts=b4.DEFAULT_PARTS, alert="frontend"):
    db = compute_signal_db(c, T0)
    assert "error" not in db, db
    return b4.diagnose_v4(db, CALLEES, {}, sensor=None, budget=0, alert=alert, parts=parts, case=c), db


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
def ratio_signals_separate_load_from_cpu_fault():
    """v5 'ratio': load entering at frontend flows down to checkout (requests x3, CPU x3: CPU per request flat, request share flat);
    cart burns 4x CPU at flat load (a CPU fault). Only cart's CPU per request rises; checkout's request share does not."""
    c = make_case()
    ordinary(c, "apm.service.legacy", "workload", 100.0, 520)
    ordinary(c, "apm.service.legacy", "cpu", 0.2, 560)
    for name, w_win, cpu_win, seed in (("frontend", 300.0, 0.6, 1), ("checkout", 300.0, 0.6, 2), ("cart", 100.0, 0.8, 3)):
        metric(c, "apm.service.legacy", name, "workload", 100.0, w_win, seed=600 + seed)
        metric(c, "apm.service.legacy", name, "cpu", 0.2, cpu_win, seed=700 + seed)
    r, db = run(c, parts=list(b4.DEFAULT_PARTS) + ["ratio"], alert="frontend")
    cal = db["_cal"]
    p = lambda e, s_: cal.get(e, {}).get(s_, {}).get("p", 1.0)
    assert p("cart", "svc_cpu_per_req_rise") <= 0.05, p("cart", "svc_cpu_per_req_rise")
    assert p("checkout", "svc_cpu_per_req_rise") > 0.05 and p("checkout", "svc_cpu_rise") <= 0.05
    assert p("checkout", "svc_req_share_rise") > 0.05 and p("checkout", "svc_request_rise") <= 0.05
    assert p("frontend", "svc_req_share_rise") <= 0.05                    # entry service: its own request rise is its share
    assert (r["entity"], r["type"]) in (("cart", "F034-cpuFullLoad"), ("cart", "F012-cpuDeadLoop")), r["top"]


@test
def trace_origin_points_at_the_failing_callee_end_to_end():
    """v5 'torigin' end to end: cart fails (HTTP 503) and its callers checkout and frontend fail with it in every trace; the trace
    structure names cart as the error origin and the diagnosis stays at cart (not at a caller that shows more symptoms)."""
    from schema import Observation
    c = make_case()
    for i in range(8):
        t0 = W0 + dt.timedelta(seconds=20 + 10 * i)
        for sid, par, svc, code in ((f"a{i}", None, "frontend", "500"), (f"b{i}", f"a{i}", "checkout", "500"),
                                    (f"c{i}", f"b{i}", "cart", "503")):
            c.observations["traces"].append(Observation(None, t0, "traces", "", {
                "traceId": f"T{i}", "spanId": sid, "parentSpanId": par or "", "serviceName": svc, "statusCode": "2",
                "statusMessage": "upstream failed", "attributes": f'"http.status_code":"{code}"', "spanName": "op", "duration": 3e8}))
    r0, _ = run(c, parts=list(b4.DEFAULT_PARTS), alert="frontend")
    r, db = run(c, parts=list(b4.DEFAULT_PARTS) + ["torigin"], alert="frontend")
    ts = db["_trace_structure"]
    assert ts["n_err_traces"] == 8 and ts["err_origin"] == {"cart": 8}, ts
    top_svc = lambda res: next(t for t in res["top"] if t["entity"] in ("cart", "checkout", "frontend"))
    assert top_svc(r)["entity"] == "cart" and top_svc(r)["p"] > top_svc(r0)["p"], (r0["top"], r["top"])
    assert not any(e["signal"].startswith("trace_") and not e["holds"] for e in r["evidence"]), r["evidence"]


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
    for parts in ([], ["score"], ["open"], ["twostage"], ["fp"], ["score", "peer"], ["score", "rnull"], ["prop"], ["events"], ["avail"], ["score", "onset"], ["em"], ["score", "ops"], list(b4.ALL_PARTS), list(b4.DEFAULT_PARTS)):
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
    assert r.retrieval_stats["v4_parts"] == ",".join(sorted(b4.DEFAULT_PARTS)), r.retrieval_stats
    r0 = run_case(c, None, budget=0, narrate=False, v4=True)
    assert r0.predicted_fault_type == "F016-rateLimiting" and r0.reasoning_chain


def raw_event(c, kind, name, reason, first, last, count, msg=""):
    import json
    from schema import Observation
    e = {"involvedObject": {"kind": kind, "name": name}, "reason": reason, "message": msg, "count": count,
         "firstTimestamp": first.isoformat() + "Z", "lastTimestamp": last.isoformat() + "Z"}
    c.observations["events"].append(Observation(None, last, "events", "", {"eventId": json.dumps(e)}))


@test
def common_mode_load_step_cancels_with_peers():
    c = make_case()
    for i in range(8):                                                   # every service: CPU x3 in the window (load step)
        series(c, "apm.service.legacy", f"svc{i}", "cpu", lambda k, i=i: (0.1 + 0.002 * (k % 3)) * (3 if k >= 110 else 1), seed=i)
    for s in ("cart", "checkout", "frontend"):
        series(c, "apm.service.legacy", s, "cpu", lambda k: (0.1 + 0.002 * (k % 3)) * (3 if k >= 110 else 1))
    series(c, "k8s.node", "node-a", "node_cpu_usage_rate", lambda k: 0.3 + 0.01 * (k % 3) if k < 110 else 0.95)
    db = compute_signal_db(c, T0)
    no_peer = calibrate_v4(db)["svc0"]["svc_cpu_rise"]["p"]
    db = compute_signal_db(c, T0)
    with_peer = calibrate_v4(db, peer=True)["svc0"]["svc_cpu_rise"]["p"]
    assert no_peer <= 0.05 < with_peer, (no_peer, with_peer)
    r, _ = run(c, alert="cart")
    assert (r["entity"], r["type"]) == ("node-a", "F026-nodeCpuHigh"), r["top"]


@test
def earlier_fault_in_history_does_not_mask_the_real_one():
    c = make_case()
    for i in range(8):                                                   # three other nodes had a 100% CPU episode in the history
        series(c, "k8s.node", f"node-x{i}", "node_cpu_usage_rate",
               lambda k, i=i: (0.95 if (i < 3 and 30 + 20 * i <= k < 42 + 20 * i) else 0.2 + 0.005 * (k % 3)), seed=i)
    series(c, "k8s.node", "node-a", "node_cpu_usage_rate", lambda k: 0.3 + 0.01 * (k % 3) if k < 110 else 0.95)
    db = compute_signal_db(c, T0)
    conformal = calibrate_v4(db)["node-a"]["node_cpu_rise"]["p"]
    db = compute_signal_db(c, T0)
    central = calibrate_v4(db, robust_null=True)["node-a"]["node_cpu_rise"]["p"]
    assert central <= 0.05 and central < conformal, (conformal, central)


@test
def absent_symptoms_do_not_penalise_a_wide_hypothesis():
    c = make_case()
    series(c, "k8s.node", "node-a", "node_cpu_usage_rate", lambda k: 0.3 + 0.01 * (k % 3) if k < 110 else 0.95)
    metric(c, "apm.service.legacy", "frontend", "rt", 0.1, 0.1, seed=13)            # frontend (on node-a) shows no symptom
    lls = {}
    for parts in (("score",), ("score", "prop")):
        db = compute_signal_db(c, T0)
        m = b4.Model(parts)
        calibrate_v4(db)
        obs = bd.observations(db)
        H = m.testable(m.hypotheses(db), obs)
        i = next(j for j, h in enumerate(H) if h["entity"] == "node-a" and h["type"] == "F026-nodeCpuHigh")
        lls[parts] = m.log_likelihood(obs, H, bd.background(obs, db), bd.explainers(db, CALLEES), db)[i]
    assert lls[("score", "prop")] > lls[("score",)], lls


@test
def chronic_events_are_not_new_but_fresh_ones_are():
    from pipeline.fault_v4 import new_events
    c = make_case()
    month = dt.timedelta(days=28)
    for j in range(12):                                                 # node-b: eviction pressure every 30 s for four weeks
        t = W0 - dt.timedelta(minutes=30) + dt.timedelta(minutes=3 * j)
        raw_event(c, "Node", "node-b", "EvictionThresholdMet", W0 - month, t, int((t - (W0 - month)).total_seconds() // 30))
    raw_event(c, "Pod", "cart-7d9f8c6b5d-x2x9k", "OOMKilling", W0 + dt.timedelta(minutes=1), W0 + dt.timedelta(minutes=2), 2,
              "Memory cgroup out of memory")
    db = compute_signal_db(c, T0)
    assert db["nodes"]["node-b"]["events"].get("evicted"), "precondition: v3 counts the chronic event"
    new_events(c, db, 4 * (db["w1"] - db["w0"]).total_seconds())
    assert not db["nodes"]["node-b"]["events"].get("evicted"), dict(db["nodes"]["node-b"]["events"])
    assert db["services"]["cart"]["events"].get("oom"), dict(db["services"]["cart"]["events"])


@test
def unavailability_is_a_state():
    from pipeline.fault_v4 import unavailability
    c = make_case()
    series(c, "k8s.deployment", "cart", "deployment_desired_replicas", lambda k: 3.0)
    for k in range(110, 122):                                            # quote-like: exported only inside the window, value 0
        c.observations["metrics"].append(_obs(W0 + dt.timedelta(seconds=30 * (k - 110)), "k8s.deployment", "cart",
                                              "deployment_available_replicas", 0.0))
    series(c, "k8s.deployment", "checkout", "deployment_desired_replicas", lambda k: 3.0)
    series(c, "k8s.deployment", "checkout", "deployment_available_replicas", lambda k: 2.0)       # chronically degraded
    series(c, "k8s.deployment", "frontend", "deployment_desired_replicas", lambda k: 3.0)
    series(c, "k8s.deployment", "frontend", "deployment_available_replicas", lambda k: 3.0)
    u = unavailability(compute_signal_db(c, T0))
    assert u == {"cart": True, "checkout": False, "frontend": False}, u


@test
def scoring_does_not_roll_a_node_up_to_a_service():
    import networkx as nx
    from schema import Entity
    from evaluation.scoring import entity_localization_score
    g = nx.DiGraph()
    for eid, et in (("svc", "apm.service"), ("inst", "apm.instance"), ("pod", "k8s.pod"), ("node", "k8s.node")):
        g.add_node(eid, entity=Entity(entity_id=eid, entity_type=et, name=eid))
    g.add_edge("svc", "inst"); g.add_edge("pod", "inst"); g.add_edge("node", "pod")

    class GT:
        target_entity_ids, target_entity_names = ["svc"], []
    assert entity_localization_score(["inst"], GT, g) == 1.0                 # an instance of the service: same logical entity
    assert entity_localization_score(["node"], GT, g) == 0.125               # the hosting node: adjacent (3 hops), not the service


@test
def onset_order_separates_root_from_later_victim():
    """cart's CPU and latency break at the window start; node-a (which hosts cart) rises 3 minutes later as a consequence."""
    c = make_case()
    ordinary(c, "apm.service.legacy", "cpu", 0.1, 700)
    ordinary(c, "apm.service.legacy", "rt", 0.1, 720)
    series(c, "apm.service.legacy", "cart", "cpu", lambda k: 0.1 + 0.002 * (k % 3) if k < 110 else 0.9)
    series(c, "apm.service.legacy", "cart", "rt", lambda k: 0.1 + 0.002 * (k % 3) if k < 110 else 0.8)
    series(c, "k8s.node", "node-a", "node_cpu_usage_rate", lambda k: 0.3 + 0.01 * (k % 3) if k < 116 else 0.9)
    lp = {}
    for parts in (tuple(p for p in b4.DEFAULT_PARTS if p != "onset"), b4.DEFAULT_PARTS):
        r, db = run(c, parts=parts, alert="cart")
        lp["onset" in parts] = r
    assert lp[True]["entity"] == "cart", lp[True]["top"]
    assert lp[True]["onset_root"] is not None and lp[True]["n_onsets"] >= 2, lp[True]
    node_p = lambda r: sum(t["p"] for t in r["top"] if t["entity"] == "node-a")
    assert node_p(lp[True]) < node_p(lp[False]), (lp[False]["top"], lp[True]["top"])     # the later-starting node loses mass


@test
def em_estimates_a_propagation_probability_and_penalises_wide_quiet_hypotheses():
    c = make_case()
    ordinary(c, "apm.service.legacy", "rt", 0.1, 740)
    series(c, "apm.service.legacy", "cart", "error", lambda k: 1.0 if k < 110 else 30.0)
    series(c, "apm.service.legacy", "checkout", "rt", lambda k: 0.1 + 0.002 * (k % 3) if k < 110 else 0.5)   # its only caller
    r, _ = run(c, alert="checkout")
    assert r["pi"] is not None and b4.PI_MIN <= r["pi"] <= b4.PI_MAX, r["pi"]
    assert r["entity"] == "cart", r["top"]


@test
def failing_operation_is_seen_when_the_service_total_is_diluted():
    c = make_case()
    for i in range(6):                                                   # six healthy operations of checkout, one failing
        series(c, "apm.operation", f"checkout::op{i}", "request_count", lambda k: 100.0 + (k % 3), seed=i)
    series(c, "apm.operation", "checkout::PlaceOrder", "error_count", lambda k: 1.0 if k < 110 else 25.0)
    for i in range(8):
        series(c, "apm.operation", f"svc{i}::op", "error_count", lambda k, i=i: float((k + i) % 2), seed=i)
    db = compute_signal_db(c, T0)
    cal = calibrate_v4(db, ops_level=True)
    assert cal["checkout"]["op_error_rise"]["p"] <= 0.05, cal["checkout"]["op_error_rise"]
    r, _ = run(c, parts=b4.DEFAULT_PARTS + ("ops",), alert="checkout")          # ops is opt-in since dev run 3
    assert (r["entity"], r["type"]) == ("checkout", "F014-httpError5xx"), r["top"]


@test
def chain_follows_the_topology_path_to_the_alerted_operation():
    from pipeline.belief_rca import call_path, rule_chain
    hosts = {"cart": ["node-a"], "frontend": ["node-a"], "checkout": ["node-b"]}
    assert call_path(CALLEES, hosts, "frontend", "cart") == ["cart", "checkout", "frontend"]
    assert call_path(CALLEES, hosts, "frontend", "node-b") == ["node-b", "checkout", "frontend"]
    assert call_path(CALLEES, hosts, "frontend", "frontend") == ["frontend"]
    res = {"entity": "cart", "type": "F029-redisUnavailable",
           "evidence": [{"entity": "cart", "signal": "redis_err", "holds": True, "ev": "12 failing redis spans"},
                        {"entity": "checkout", "signal": "svc_error_rise", "holds": True, "ev": "error: window max=40"}]}
    ch = rule_chain(res, ["cart", "checkout", "frontend"], "frontend::POST /api/checkout")
    assert ch[0].startswith("cause: cart F029") and ch[1].startswith("propagation: checkout svc_error_rise") \
        and ch[2] == "propagation: frontend" and ch[-1] == "impact: frontend::POST /api/checkout", ch


@test
def chain_node_match_counts_named_nodes_not_word_overlap():
    from evaluation.scoring import chain_node_match, _chain_overlap_score
    gt = ["cause: payment", "propagation: checkout", "impact: checkout::/oteldemo.CheckoutService/PlaceOrder"]
    terse = ["cause: payment", "propagation: checkout"]
    rich = ["cause: payment error_count=8829 error_rate=0.6", "propagation: checkout error_count=8828",
            "impact: checkout::/oteldemo.CheckoutService/PlaceOrder"]
    assert chain_node_match(rich, gt) == 1.0 and abs(chain_node_match(terse, gt) - 2 / 3) < 1e-9
    assert _chain_overlap_score(rich, gt) < 1.0          # the word-Jaccard proxy penalises the evidence words


@test
def volatile_entity_loses_to_one_that_changed_now():
    """svc0 swings x3 in every window of its history (a noisy service); cart rises x2 only now. With selfnorm cart's request rise is
    the stronger, calibrated signal."""
    from pipeline.fault_v4 import calibrate_v4 as cal4
    c = make_case()
    ordinary(c, "apm.service.legacy", "workload", 100.0, 760)
    c.observations["metrics"] = [o for o in c.observations["metrics"] if not (o.payload["metric"] == "workload" and o.payload["entity_name"] == "svc0")]
    series(c, "apm.service.legacy", "svc0", "workload", lambda k: 100.0 * (3 if (k // 12) % 2 else 1) + (k % 3))
    series(c, "apm.service.legacy", "cart", "workload", lambda k: 100.0 + (k % 3) if k < 110 else 200.0)
    db = compute_signal_db(c, T0)
    plain = cal4(db)
    db = compute_signal_db(c, T0)
    normed = cal4(db, selfnorm=True)
    assert normed["cart"]["svc_request_rise"]["p"] <= 0.05, normed["cart"]["svc_request_rise"]
    assert normed["svc0"]["svc_request_rise"]["p"] > plain["svc0"]["svc_request_rise"]["p"] or normed["svc0"]["svc_request_rise"]["p"] > 0.05


@test
def llm_typing_needs_a_candidate_and_a_verbatim_line():
    from pipeline.llm_typing import llm_type, evidence_lines
    c = make_case()
    series(c, "apm.metric.jvm", "cart", "arms_jvm_mem_used_bytes", lambda k: 4e8 + 2e7 * (k % 4) if k < 110 else 9.5e8)
    ordinary(c, "apm.service.legacy", "mem", 0.5, 780)
    r, db = run(c, alert="cart")
    lines = evidence_lines(db, r["entity"], {})
    assert lines, "evidence summary is empty"

    class L:
        def __init__(self, ans):
            self.ans = ans

        def generate_json(self, prompt, system=None):
            return self.ans
    from pipeline.llm_typing import shortlist
    sl = shortlist(r)
    assert r["type"] in sl and 2 <= len(sl) <= 5, sl
    alt = next(t for t in sl if t != r["type"])
    good = llm_type(L({"fault_type": alt, "evidence_line": lines[0]}), db, r, {})
    assert good["accepted"] and good["changed"] and good["llm_type"] == alt, good
    bad_quote = llm_type(L({"fault_type": alt, "evidence_line": "made up evidence line here"}), db, r, {})
    assert not bad_quote["accepted"], bad_quote
    outside = next(t for t in r["candidate_types"] if t not in sl)
    bad_type = llm_type(L({"fault_type": outside, "evidence_line": lines[0]}), db, r, {})
    assert not bad_type["accepted"], bad_type                           # outside the statistical shortlist
    # v5 'xcard': same rules; the prompt carries modality tags and signatures, and a quote copied WITH the tag is still grounded
    seen = {}

    class LP(L):
        def generate_json(self, prompt, system=None):
            seen["p"] = prompt
            return self.ans
    rx = {**r, "parts": list(r.get("parts", [])) + ["xcard"]}
    g2 = llm_type(LP({"fault_type": alt, "evidence_line": "[metrics] " + lines[0]}), db, rx, {})
    assert g2["accepted"] and "expected: one of" in seen["p"] and "[metrics] " in seen["p"], g2


@test
def v4_narrator_is_called_and_rule_chain_skips_the_alert():
    from pipeline.belief_rca import run_case, rule_chain
    from test_hypothesis_rca import case as v2case, Stub
    from test_calibrated import wspan

    class Rec(Stub):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.prompts = []

        def generate_json(self, prompt, system=None):
            self.prompts.append(prompt)
            if "Which ONE category" in prompt:
                return {"category": "ratelimit", "quote": "Too Many Requests"}
            return super().generate_json(prompt, system)
    c = v2case()
    for i in range(6):
        wspan(c, "checkout", i, err=True, code="429", msg="Too Many Requests")
    llm = Rec()
    run_case(c, llm, budget=5, narrate=True, v4=True)
    assert any("Task: write the causal chain" in p for p in llm.prompts), [p[:60] for p in llm.prompts]
    res = {"entity": "cart", "type": "F029-redisUnavailable",
           "evidence": [{"entity": "frontend", "signal": "alert", "holds": True, "ev": "the alert fired on this service"}]}
    assert rule_chain(res, ["cart", "checkout", "frontend"], "frontend::GET") == ["cause: cart F029-redisUnavailable",
                                                                                "propagation: checkout", "propagation: frontend",
                                                                                "impact: frontend::GET"]


@test
def oom_and_restarts_after_memory_growth_are_memory_pressure():
    c = make_case()
    ordinary(c, "apm.service.legacy", "mem", 0.5, 800)
    series(c, "apm.service.legacy", "cart", "mem", lambda k: 0.5 + 0.005 * (k % 3) if k < 110 else 0.95)
    c.event("Pod", "cart-7d9f8c6b5d-x2x9k", "OOMKilling", "Memory cgroup out of memory: Killed process", when=W0 + dt.timedelta(minutes=2))
    c.event("Pod", "cart-7d9f8c6b5d-x2x9k", "Killing", "Container cart failed liveness probe, will be restarted", when=W0 + dt.timedelta(minutes=3))
    r, _ = run(c, alert="cart")
    assert (r["entity"], r["type"]) == ("cart", "F007-memoryPressure"), r["top"]
    r0, _ = run(c, parts=tuple(p for p in b4.DEFAULT_PARTS if p != "conseq"), alert="cart")
    assert r0["type"] != "F007-memoryPressure", r0["top"]               # without 'conseq' the OOM kill excludes memory pressure


@test
def restarts_after_cpu_saturation_are_cpu_types():
    c = make_case()
    ordinary(c, "apm.service.legacy", "cpu", 0.2, 820)
    series(c, "apm.service.legacy", "cart", "cpu", lambda k: 0.2 + 0.004 * (k % 3) if k < 110 else 0.98)
    c.event("Pod", "cart-7d9f8c6b5d-x2x9k", "Unhealthy", "Liveness probe failed: timeout", when=W0 + dt.timedelta(minutes=2))
    c.event("Pod", "cart-7d9f8c6b5d-x2x9k", "Killing", "Container cart failed liveness probe, will be restarted", when=W0 + dt.timedelta(minutes=3))
    r, _ = run(c, alert="cart")
    assert r["entity"] == "cart" and r["type"] in ("F034-cpuFullLoad", "F012-cpuDeadLoop"), r["top"]
    r0, _ = run(c, parts=tuple(p for p in b4.DEFAULT_PARTS if p != "conseq"), alert="cart")
    assert r0["type"] == "F052-podRestartFlapping", r0["top"]          # the dev-run-4 error this part removes


@test
def retry_request_rise_upstream_is_explained_by_the_failing_callee():
    """cart fails (errors); checkout (its caller) sees more requests (retries). Without 'retry' that rise is trafficSurge evidence at
    checkout; with it, it is a propagated symptom of the cart fault."""
    c = make_case()
    ordinary(c, "apm.service.legacy", "workload", 100.0, 840)
    ordinary(c, "apm.service.legacy", "error", 1.0, 860, noise=0.5)
    series(c, "apm.service.legacy", "cart", "error", lambda k: 1.0 if k < 110 else 30.0)
    series(c, "apm.service.legacy", "checkout", "workload", lambda k: 100.0 + (k % 3) if k < 110 else 180.0)
    surge = lambda r: sum(t["p"] for t in r["top"] if t["entity"] == "checkout" and t["type"] == "F006-trafficSurge")
    r, _ = run(c, alert="checkout")
    r0, _ = run(c, parts=tuple(p for p in b4.DEFAULT_PARTS if p != "retry"), alert="checkout")
    assert r["entity"] == "cart" and surge(r) < surge(r0), (r["top"], r0["top"])


@test
def a_single_new_event_counts():
    from pipeline.fault_v4 import new_events
    c = make_case()
    c.event("Pod", "cart-7d9f8c6b5d-x2x9k", "OOMKilling", "Memory cgroup out of memory", when=W0 + dt.timedelta(minutes=2))
    db = compute_signal_db(c, T0)
    new_events(c, db, 4 * (db["w1"] - db["w0"]).total_seconds())
    assert db["services"]["cart"]["events"].get("oom"), dict(db["services"]["cart"]["events"])


@test
def depbase_db_slow_against_history_end_to_end():
    """v5 'depbase': cart's database calls take 0.4 ms before the window and 5 s inside it (timeouts) while its server spans grow from
    2 ms to 5 s too. The window-only test (DB spans vs the other window spans) cannot see it; the history test can, the slowSQL /
    dbNetworkLatency hypotheses at cart gain, and entity_signals() reports the same value to the LLM typing step."""
    from schema import Observation
    c = make_case()
    db0 = compute_signal_db(c, T0)
    w0, w1 = db0["w0"], db0["w1"]
    L = (w1 - w0).total_seconds()
    def add(t, name, dur, attrs, i):
        c.observations["traces"].append(Observation(None, t, "traces", "", {
            "traceId": f"D{i}", "spanId": f"{name[:3]}{i}", "parentSpanId": "", "serviceName": "cart", "statusCode": "0",
            "statusMessage": "", "attributes": attrs, "spanName": name, "duration": dur * 1e6}))
    for i in range(200):                                                     # history: the B_SPAN window lengths before w0
        t = w0 - dt.timedelta(seconds=(i + 1) * 4 * L / 201)
        add(t, "SELECT cart.items", 0.4 + 0.1 * (i % 2), '{"db.system":"mysql"}', i)
        add(t, "GET /cart", 2.0 + 0.1 * (i % 3), "{}", 1000 + i)
    for i in range(6):
        t = w0 + dt.timedelta(seconds=(i + 1) * L / 8)
        add(t, "SELECT cart.items", 5000.0 + i, '{"db.system":"mysql"}', 2000 + i)
        add(t, "GET /cart", 5003.0 + i, "{}", 3000 + i)
    base = list(b4.DEFAULT_PARTS)
    r0, db_a = run(c, parts=base, alert="frontend")
    r1, db_b = run(c, parts=base + ["depbase"], alert="frontend")
    o0 = [o for o in bd.observations(db_a) if o["entity"] == "cart" and o["signal"] == "db_slow"]
    assert o0 and not o0[0]["holds"], o0                                     # window-only test misses it
    assert entity_signals(db_b, "cart")["db_slow"]["p"] <= 0.05             # history test sees it, also for entity_signals()
    p_dep = lambda r: sum(t["p"] for t in r["top"] if t["entity"] == "cart" and t["type"] in ("F010-slowSQL", "F018-dbNetworkLatency"))
    assert r1["entity"] == "cart" and r1["type"] == "F010-slowSQL" and p_dep(r1) > p_dep(r0), (r0["top"], r1["top"])
    r2, db_c = run(c, parts=base + ["depbase", "depshare"], alert="frontend")              # the time moved into the DB calls
    assert entity_signals(db_c, "cart")["db_slow"]["p"] <= 0.05 and "share of request time" in entity_signals(db_c, "cart")["db_slow"]["ev"]


@test
def sigorder_runs_end_to_end():
    c = make_case()
    r, _ = run(c, parts=list(b4.DEFAULT_PARTS) + ["sigorder"], alert="frontend")
    assert r["entity"] and r["type"]


@test
def sigorder2_keeps_the_entity_marginals():
    c = make_case()
    r1, _ = run(c, parts=list(b4.DEFAULT_PARTS), alert="frontend")
    r2, _ = run(c, parts=list(b4.DEFAULT_PARTS) + ["sigorder2"], alert="frontend")
    assert r1["entity"] == r2["entity"] and abs(r1["p"] - r2["p"]) < 1e-9, (r1["entity"], r1["p"], r2["entity"], r2["p"])


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
