"""tests/test_fingerprint.py -- synthetic checks of the fault-fingerprint layer and the chain verifier (no data, no model).

    python tests/test_fingerprint.py

Each scenario builds a tiny topology and telemetry by hand and checks which fault types the fingerprint layer lists as candidates.
The scenarios come from the type DEFINITIONS (data/taxonomy.py), not from RCA100 cases."""
import datetime as dt
import json
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    import sentence_transformers  # noqa: F401
except Exception:
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = type("SentenceTransformer", (), {})
    sys.modules["sentence_transformers"] = st

import networkx as nx

import config
from schema import Entity, Relation, Observation
from pipeline.fault_fingerprint import compute_signal_db, origin_signals, rank_candidates, profile_text

T0 = dt.datetime(2026, 1, 1, 12, 0, 0)          # alert time; window = [T0-5min, T0+1min], baseline before that


class FakeAlert:
    def __init__(self):
        self.alert_timestamp = T0
        self.raw = {}


class FakeCase:
    def __init__(self):
        g = nx.DiGraph()
        for eid, et, name in (("s1", "apm.service", "frontend"), ("s2", "apm.service", "checkout"), ("s3", "apm.service", "cart"),
                              ("n1", "k8s.node", "node-a"), ("n2", "k8s.node", "node-b"),
                              ("p1", "k8s.pod", "cart-7d9f8c6b5d-x2x9k"), ("p2", "k8s.pod", "checkout-5c6d7e8f9a-ab12c"),
                              ("p3", "k8s.pod", "frontend-6b7c8d9e0f-zz9zz")):
            g.add_node(eid, entity=Entity(entity_id=eid, entity_type=et, name=name))
        for u, v, r in (("s1", "s2", "calls"), ("s2", "s3", "calls"), ("n1", "p1", "contains"), ("n1", "p3", "contains"), ("n2", "p2", "contains")):
            g.add_edge(u, v, relation=Relation(source_id=u, target_id=v, relation_type=r))
        self.topology = g
        self.alert = FakeAlert()
        self.observations = {"metrics": [], "traces": [], "logs": [], "events": []}

    def metric(self, eset, name, metric, base, win, n_base=30, n_win=6):
        for i in range(n_base):
            self.observations["metrics"].append(Observation(None, T0 - dt.timedelta(minutes=40 - i), "metrics", "",
                                                            {"entity_set": eset, "entity_name": name, "metric": metric, "value": base + (i % 3) * 0.01 * base}))
        for i in range(n_win):
            self.observations["metrics"].append(Observation(None, T0 - dt.timedelta(minutes=4 - i * 0.5), "metrics", "",
                                                            {"entity_set": eset, "entity_name": name, "metric": metric, "value": win}))

    def span(self, svc, when, err=False, code=None, msg="", name="GET /x", attrs="", dur_ns=5e6):
        a = attrs + (f'"http.status_code":"{code}"' if code else "")
        self.observations["traces"].append(Observation(None, when, "traces", "", {"serviceName": svc, "statusCode": "2" if err else "1",
                                                                                     "statusMessage": msg, "attributes": a, "spanName": name, "duration": dur_ns}))

    def log(self, container, when, content):
        self.observations["logs"].append(Observation(None, when, "logs", "", {"_container_name_": container, "content": content}))

    def event(self, kind, name, reason, msg="", when=None, count=1):
        when = when or T0 - dt.timedelta(minutes=2)
        e = {"involvedObject": {"kind": kind, "name": name}, "reason": reason, "message": msg, "count": count,
             "firstTimestamp": when.isoformat() + "Z", "lastTimestamp": when.isoformat() + "Z"}
        self.observations["events"].append(Observation(None, when, "events", "", {"eventId": json.dumps(e)}))


def baseline_traffic(c, svc="checkout", n=40):
    for i in range(n):
        c.span(svc, T0 - dt.timedelta(minutes=50 - i))
    for i in range(10):
        c.span(svc, T0 - dt.timedelta(minutes=4, seconds=-i * 10))


def types_for(c, origin):
    db = compute_signal_db(c, T0)
    assert "error" not in db, db
    osig = origin_signals(db, origin)
    cands = rank_candidates(osig, top_k=0)
    return [x["type"] for x in cands], osig, cands


def win(i=0):
    return T0 - dt.timedelta(minutes=3, seconds=-i * 5)


SCENARIOS = []


def scenario(fn):
    SCENARIOS.append(fn)
    return fn


@scenario
def node_cpu_high():
    c = FakeCase()
    c.metric("k8s.node", "node-a", "node_cpu_usage_rate", 0.30, 0.96)
    c.metric("k8s.node", "node-b", "node_cpu_usage_rate", 0.30, 0.31)
    t, osig, _ = types_for(c, "node-a")
    assert t and t[0] == "F026-nodeCpuHigh", (t, osig)
    t2, _, _ = types_for(c, "cart")                    # cart runs on node-a: node signal reaches the service origin
    assert "F026-nodeCpuHigh" in t2, t2
    t3, _, _ = types_for(c, "node-b")
    assert "F026-nodeCpuHigh" not in t3, t3


@scenario
def rate_limiting_429():
    c = FakeCase()
    baseline_traffic(c)
    for i in range(8):
        c.span("checkout", win(i), err=True, code="429", msg="Too Many Requests")
    t, osig, _ = types_for(c, "checkout")
    assert t[0] == "F016-rateLimiting", (t, osig["signals"])


@scenario
def null_pointer():
    c = FakeCase()
    baseline_traffic(c)
    for i in range(6):
        c.log("checkout", win(i), "ERROR java.lang.NullPointerException at OrderService.place")
        c.span("checkout", win(i), err=True, code="500", msg="NullPointerException")
    t, osig, _ = types_for(c, "checkout")
    assert t[0] == "F023-nullPointerException", (t, osig["signals"])
    assert "F014-httpError5xx" in t


@scenario
def crash_loop():
    c = FakeCase()
    c.event("Pod", "cart-7d9f8c6b5d-x2x9k", "BackOff", "Back-off restarting failed container cart", count=7)
    c.event("Pod", "cart-7d9f8c6b5d-x2x9k", "Killing", "Stopping container cart", count=3)
    t, osig, _ = types_for(c, "cart")
    assert t[0] == "F050-podCrashLoop", (t, osig["signals"])
    assert "F052-podRestartFlapping" not in t[:1]


@scenario
def full_gc():
    c = FakeCase()
    c.metric("apm.metric.jvm", "checkout", "arms_jvm_gc_delta", 2.0, 40.0)
    t, osig, _ = types_for(c, "checkout")
    assert t[0] == "F022-fullGC", (t, osig["signals"])


@scenario
def scale_down():
    c = FakeCase()
    c.metric("k8s.deployment", "cart", "deployment_available_replicas", 2.0, 0.0)
    c.event("Deployment", "cart", "ScalingReplicaSet", "Scaled down replica set cart-7d9f8c6b5d to 0")
    t, osig, _ = types_for(c, "cart")
    assert t[0] == "F036-replicaScaleDown", (t, osig["signals"])


@scenario
def slow_sql():
    c = FakeCase()
    for i in range(40):
        c.span("checkout", T0 - dt.timedelta(minutes=50 - i), name="SELECT orders", attrs='"db.system":"postgresql",', dur_ns=20e6)
    for i in range(5):
        c.span("checkout", win(i), name="SELECT orders", attrs='"db.system":"postgresql",', dur_ns=3000e6)
    t, osig, _ = types_for(c, "checkout")
    assert t[0] in ("F010-slowSQL", "F018-dbNetworkLatency"), (t, osig["signals"])


@scenario
def quiet_case_has_no_candidate():
    c = FakeCase()
    baseline_traffic(c)
    t, osig, cands = types_for(c, "checkout")
    assert t == [], (t, osig["signals"])
    assert "none of the checked signals" in profile_text("checkout", osig, cands)


@scenario
def unknown_origin():
    c = FakeCase()
    db = compute_signal_db(c, T0)
    assert origin_signals(db, "does-not-exist")["kind"] == "unknown"


@scenario
def verifier_grounding():
    from agents import multi_agent as ma
    state = {"propagation_text": "- cart error_rate window=4.0%", "infra_text": "- service cart | request_count base=120 max=15; latency base=0.12 max=2.8",
             "trace_text": "", "evidence_summary": {}}
    pool, _ = ma._evidence_pool(state)
    g, n, bad = ma._grounding(["cause: cart latency=2.8s request_count=15", "impact: frontend error_count=999"], pool)
    assert n == 3 and bad == ["999"], (g, n, bad)
    g2, _, _ = ma._grounding(["cause: cart latency=2800ms"], pool)        # ms vs s conversion
    assert g2 == 1.0


@scenario
def typing_shortlist_with_stub_llm():
    from agents import multi_agent as ma
    c = FakeCase()
    baseline_traffic(c)
    for i in range(8):
        c.span("checkout", win(i), err=True, code="429", msg="Too Many Requests")

    class Stub:
        def __init__(self, answer):
            self.answer, self.prompts = answer, []

        def generate_json(self, prompt, system=None):
            self.prompts.append(prompt)
            return {"fault_evidence": "http 429", "predicted_fault_type": self.answer, "confidence": 0.7}

    saved = (config.FAULT_FINGERPRINT, config.ENTITY_TYPING, config.FP_SCORING)
    config.FP_SCORING = "fixed"
    try:
        for mode, answer, expect in (("shortlist", "F014-httpError5xx", "F016-rateLimiting"),      # outside the shortlist -> top candidate
                                     ("shortlist", "F016", "F016-rateLimiting"),
                                     ("rule", None, "F016-rateLimiting"),
                                     ("show", "F011-codeDefect", "F011-codeDefect")):              # show: any type allowed
            config.FAULT_FINGERPRINT, config.ENTITY_TYPING = mode, False
            st = {"alert_text": "a", "fp_db": compute_signal_db(c, T0), "evidence_summary": {},
                  "final_result": {"reasoning_chain": ["cause: checkout http_429=8", "impact: frontend error_count=8"],
                                   "predicted_entity_ids": ["checkout"], "predicted_fault_type": "F014-httpError5xx"}}
            llm = Stub(answer)
            ma._apply_entity_typing(llm, st, "taxonomy")
            fr = st["final_result"]
            assert fr["predicted_fault_type"] == expect, (mode, fr)
            assert (len(llm.prompts) == 0) == (mode == "rule")
            assert fr["fp_candidates"].startswith("F016-rateLimiting")
    finally:
        config.FAULT_FINGERPRINT, config.ENTITY_TYPING, config.FP_SCORING = saved


if __name__ == "__main__":
    failed = 0
    for fn in SCENARIOS:
        try:
            fn()
            print("ok  ", fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", fn.__name__, e)
    print(f"{len(SCENARIOS) - failed}/{len(SCENARIOS)} passed")
    sys.exit(1 if failed else 0)
