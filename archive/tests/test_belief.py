"""tests/test_belief.py -- synthetic acceptance tests of the v3 computed-belief diagnosis (design section 5). No data, no model.

    python tests/test_belief.py

Fake topology: frontend -> checkout -> cart (calls); node-a hosts cart and frontend, node-b hosts checkout."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from test_fingerprint import T0
from test_calibrated import make_case, metric, wspan
from test_hypothesis_rca import case as v2case, Stub
from pipeline.fault_calibrated import compute_signal_db
from pipeline import belief_diagnosis as bd

CALLEES = {"frontend": ["checkout"], "checkout": ["cart"]}


def run(c, texts=None, sensor=None, budget=5, alert="frontend"):
    db = compute_signal_db(c, T0)
    assert "error" not in db, db
    return bd.diagnose_belief(db, CALLEES, texts or {}, sensor=sensor, budget=budget, alert=alert)


T = []


def test(fn):
    T.append(fn)
    return fn


@test
def bayes_concentrates_on_categorical_mechanism():
    c = make_case()
    for i in range(6):
        wspan(c, "checkout", i, err=True, code="429", msg="Too Many Requests")
    r = run(c)
    assert (r["entity"], r["type"]) == ("checkout", "F016-rateLimiting"), r["top"]


@test
def single_node_cpu_is_a_node_fault():
    c = make_case()
    metric(c, "k8s.node", "node-a", "node_cpu_usage_rate", 0.30, 0.95)
    r = run(c)
    assert (r["entity"], r["type"]) == ("node-a", "F026-nodeCpuHigh"), r["top"]


@test
def origin_beats_upstream_victims():
    c = make_case()
    c.event("Pod", "cart-7d9f8c6b5d-x2x9k", "OOMKilled", "Container cart was OOMKilled")
    metric(c, "apm.service.legacy", "cart", "latency", 0.1, 0.6, seed=1)
    metric(c, "apm.service.legacy", "checkout", "latency", 0.1, 0.6, seed=2)
    metric(c, "apm.service.legacy", "frontend", "latency", 0.1, 0.6, seed=3)
    r = run(c)
    assert r["entity"] == "cart" and r["type"] == "F039-resourceLimitMisconfig", r["top"]


@test
def probe_resolves_ambiguity_and_is_not_repeated():
    c = make_case()
    for i in range(6):
        wspan(c, "checkout", i, err=True, code="500", msg="internal error")
    calls = []

    def sensor(e, texts):
        calls.append(e)
        return "npe" if e == "checkout" else "none"
    r = run(c, texts={"checkout": ["java.lang.NullPointerException at Order.place"], "frontend": ["upstream 500"]}, sensor=sensor)
    assert (r["entity"], r["type"]) == ("checkout", "F023-nullPointerException"), (r["top"], r["probes"])
    assert len(calls) == len(set(calls)) and calls[0] == "checkout", calls


@test
def loop_terminates_without_trapping():
    c = make_case()
    for i in range(6):
        wspan(c, "checkout", i, err=True, code="500", msg="internal error")
    calls = []
    texts = {e: ["something"] for e in ("frontend", "checkout", "cart", "svc0", "svc1", "svc2")}
    r = run(c, texts=texts, sensor=lambda e, t: calls.append(e) or "none", budget=5)
    assert len(calls) <= 5 and len(calls) == len(set(calls)) and len(r["probes"]) == len(calls), calls


@test
def no_evidence_leaves_mass_on_h0():
    r = run(make_case(), alert=None)
    assert r["p_h0"] > r["p"], (r["p_h0"], r["top"])


@test
def unrelated_false_alarm_does_not_beat_alert_path():
    """svc0..7 are not connected to the alerted service: a false-positive rise there must not win over evidence on the alert path."""
    c = make_case()
    metric(c, "apm.metric.jvm", "svc5", "arms_jvm_gc_delta", 2.0, 30.0, noise=0.5)          # strong, but off the alert path
    for i in range(6):
        wspan(c, "checkout", i, err=True, code="429", msg="Too Many Requests")
    r = run(c)
    assert (r["entity"], r["type"]) == ("checkout", "F016-rateLimiting"), r["top"]


@test
def eig_prefers_the_discriminating_probe():
    c = make_case()
    for i in range(6):
        wspan(c, "checkout", i, err=True, code="500", msg="internal error")
    db = compute_signal_db(c, T0)
    obs = bd.observations(db)
    H = bd.testable(bd.hypotheses(db), obs)
    post = bd.posterior(bd.prior(H) + bd.log_likelihood(obs, H, bd.background(obs, db), bd.explainers(db, CALLEES), db))
    g_chk, g_other = bd.eig_probe(post, H, "checkout"), bd.eig_probe(post, H, "svc3")
    assert g_chk > g_other, (g_chk, g_other)


@test
def sensor_requires_a_real_quote():
    from pipeline.semantic_probes import make_sensor

    class L:
        def __init__(self, ans):
            self.ans = ans

        def generate_json(self, prompt, system=None):
            return self.ans
    log = []
    s = make_sensor(L({"category": "npe", "quote": "NullPointerException"}), log)
    assert s("checkout", ["java.lang.NullPointerException at x"]) == "npe"
    s2 = make_sensor(L({"category": "npe", "quote": "made up"}), log)
    assert s2("checkout", ["java.lang.NullPointerException at x"]) == "none"


@test
def end_to_end_with_stub_llm():
    from pipeline.belief_rca import run_case

    class SensorStub(Stub):
        def generate_json(self, prompt, system=None):
            if "Which ONE category" in prompt:
                self.prompts.append(prompt)
                return {"category": "ratelimit", "quote": "Too Many Requests"}
            return super().generate_json(prompt, system)
    c = v2case()
    for i in range(6):
        wspan(c, "checkout", i, err=True, code="429", msg="Too Many Requests")
    r = run_case(c, SensorStub(), budget=5, narrate=True)
    assert r.predicted_fault_type == "F016-rateLimiting" and r.reasoning_chain, (r.predicted_fault_type, r.retrieval_stats)
    r0 = run_case(c, None, budget=0, narrate=False)
    assert r0.predicted_fault_type == "F016-rateLimiting" and r0.reasoning_chain


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
