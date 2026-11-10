"""tests/test_gama.py -- gama/ on synthetic scenarios (no data, no model): outputs equal the golden file for every module configuration
(core, each leave-one-out, base, full with a deterministic stub LLM: prompts included). The golden file was generated when gama/ was
proven identical to the frozen development code (tag v5-frozen; 206 / 206 on the 103 RCA100 cases). Regenerate ONLY for an intended,
recorded change:  python tests/test_gama.py --regen
"""
import datetime as dt
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from test_calibrated import make_case, metric, wspan, W0
import hashlib
import json
from golden import StubLLM, snapshot, diff
from gama.config import Modules, CORE

T = []


def test(fn):
    T.append(fn)
    return fn


def _case(alert="s1"):
    c = make_case()
    c.case_id = "t999"
    c.alert.alert_text = "latency alert"
    c.alert.entry_entity_id = alert
    return c


def _ordinary(c, eset, m, base, seed0, noise=0.05):
    for i in range(8):
        metric(c, eset, f"svc{i}", m, base, base, noise=noise, seed=seed0 + i)


def sc_ratelimit():
    c = _case()
    for i in range(6):
        wspan(c, "checkout", i, err=True, code="429", msg="Too Many Requests")
    return c


def sc_trace_origin():
    from schema import Observation
    c = _case()
    for i in range(8):
        t0 = W0 + dt.timedelta(seconds=20 + 10 * i)
        for sid, par, svc, code in ((f"a{i}", None, "frontend", "500"), (f"b{i}", f"a{i}", "checkout", "500"),
                                    (f"c{i}", f"b{i}", "cart", "503")):
            c.observations["traces"].append(Observation(None, t0, "traces", "", {
                "traceId": f"T{i}", "spanId": sid, "parentSpanId": par or "", "serviceName": svc, "statusCode": "2",
                "statusMessage": "upstream failed", "attributes": f'"http.status_code":"{code}"', "spanName": "op", "duration": 3e8}))
    return c


def sc_load_and_cpu():
    c = _case()
    _ordinary(c, "apm.service.legacy", "workload", 100.0, 520)
    _ordinary(c, "apm.service.legacy", "cpu", 0.2, 560)
    for name, w_win, cpu_win, seed in (("frontend", 300.0, 0.6, 1), ("checkout", 300.0, 0.6, 2), ("cart", 100.0, 0.8, 3)):
        metric(c, "apm.service.legacy", name, "workload", 100.0, w_win, seed=600 + seed)
        metric(c, "apm.service.legacy", name, "cpu", 0.2, cpu_win, seed=700 + seed)
        metric(c, "apm.service.legacy", name, "rt", 0.1, 0.3 if name != "cart" else 0.1, seed=800 + seed)
    return c


def sc_msgtarget_and_text():
    from schema import Observation
    c = _case()
    for i in range(12):
        t0 = W0 + dt.timedelta(seconds=20 + 10 * i)
        for sid, par, svc in ((f"a{i}", None, "frontend"), (f"b{i}", f"a{i}", "checkout")):
            c.observations["traces"].append(Observation(None, t0, "traces", "", {
                "traceId": f"M{i}", "spanId": sid, "parentSpanId": par or "", "serviceName": svc, "statusCode": "2",
                "statusMessage": "error sending request for url (http://cart:8080/getcart): connection refused",
                "attributes": '{"exception.type":"java.lang.NullPointerException"}' if svc == "checkout" else "",
                "spanName": "op", "duration": 3e8}))
    for i in range(40):
        c.observations["logs"].append(Observation(None, W0 - dt.timedelta(seconds=15 * (i + 1)), "logs", "",
                                                  {"_container_name_": "cart", "content": "GET /cart 200 ok"}))
    for i in range(12):
        c.observations["logs"].append(Observation(None, W0 + dt.timedelta(seconds=10 * i), "logs", "",
                                                  {"_container_name_": "cart", "content": "request timed out after 3000 ms"}))
        c.observations["logs"].append(Observation(None, W0 + dt.timedelta(seconds=10 * i), "logs", "",
                                                  {"_container_name_": "checkout", "content": "java.lang.OutOfMemoryError: GC overhead"}))
    return c


def sc_memory_gc_order():
    c = _case()
    _ordinary(c, "apm.service.legacy", "workload", 100.0, 520)
    for name, m, base, win, seed in (("cart", "heap_used", 100.0, 400.0, 1), ("cart", "gc_time", 1.0, 30.0, 2),
                                     ("cart", "cpu", 0.2, 0.5, 3), ("checkout", "rt", 0.1, 0.6, 4), ("frontend", "rt", 0.1, 0.6, 5),
                                     ("checkout", "error", 1.0, 20.0, 6)):
        metric(c, "apm.service.legacy", name, m, base, win, seed=900 + seed)
    return c


def sc_node():
    c = _case()
    _ordinary(c, "k8s.node", "node_cpu_usage_rate", 0.3, 300)
    metric(c, "k8s.node", "node-a", "node_cpu_usage_rate", 0.3, 0.95, seed=11)
    for name, seed in (("cart", 12), ("frontend", 13)):
        metric(c, "apm.service.legacy", name, "rt", 0.1, 0.5, seed=seed)
        metric(c, "apm.service.legacy", name, "cpu", 0.2, 0.5, seed=seed + 50)
    return c


SCENARIOS = [sc_ratelimit, sc_trace_origin, sc_load_and_cpu, sc_msgtarget_and_text, sc_memory_gc_order, sc_node]


GOLDEN = os.path.join(HERE, "snapshots", "gama_golden.json")
CONFIGS = ["core"] + [f"core,-{x}" for x in CORE] + ["base", "full", "full,-chain", "full,-typing"]
FROZEN_CORE_PARTS = ("score", "open", "twostage", "fp", "peer", "prop", "events", "avail", "onset", "em", "selfnorm", "llmtype",
                     "conseq", "retry", "mm", "torigin3", "omsg", "hostcpu", "errclass", "tiebreak", "sigorder2", "loadx2",
                     "msgtarget", "textx2", "redchain")       # V4_DEFAULT + V5_FINAL of the frozen code


def gama_snapshot(sc, spec):
    from gama.run import run_case
    m = Modules.parse(spec)
    llm = StubLLM() if m.llm else None
    return snapshot(run_case(sc(), llm, m), llm)


def digest(snap):
    return hashlib.sha256(json.dumps(snap, sort_keys=True, default=str).encode()).hexdigest()


def summary(snap):
    return {"entity": snap["entity"], "type": snap["type"], "confidence": snap["confidence"], "chain": snap["chain"],
            "n_llm_calls": len(snap["llm_calls"] or [])}


def regen():
    out = {f"{sc.__name__}|{spec}": {"sha256": digest(s), **summary(s)}
           for sc in SCENARIOS for spec in CONFIGS for s in [gama_snapshot(sc, spec)]}
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    json.dump(out, open(GOLDEN, "w"), indent=1, sort_keys=True)
    print(f"wrote {len(out)} golden outputs to {GOLDEN}")


@test
def module_spec_parses_and_maps_to_the_frozen_parts():
    assert Modules.parse("core").on() == CORE and not Modules.parse("core").llm
    assert Modules.parse("full").llm and Modules.parse("core,-trace").on() == tuple(x for x in CORE if x != "trace")
    assert Modules.parse("base").on() == () and Modules.parse("base,mm").on() == ("mm",)
    assert set(Modules.parse("core").legacy_parts()) == set(FROZEN_CORE_PARTS)
    assert set(Modules.parse("full").legacy_parts()) == set(FROZEN_CORE_PARTS) | {"xcard", "keepchain", "typecheck"}


@test
def outputs_equal_the_golden_file_for_every_configuration():
    gold = json.load(open(GOLDEN))
    bad = []
    for sc in SCENARIOS:
        for spec in CONFIGS:
            key = f"{sc.__name__}|{spec}"
            s = gama_snapshot(sc, spec)
            if digest(s) != gold[key]["sha256"]:
                bad.append((key, diff({k: gold[key][k] for k in summary(s)}, summary(s))[:3] or ["(trail / stats differ)"]))
    assert not bad, bad
    assert len(gold) == len(SCENARIOS) * len(CONFIGS)


@test
def the_stub_llm_is_called_in_full_configurations():
    for sc in SCENARIOS:
        assert gama_snapshot(sc, "full")["llm_calls"], sc.__name__


@test
def scenarios_reach_the_module_branches():
    """The scenarios are not trivial: modules change something in at least one of them."""
    from gama.run import run_case
    changed = set()
    for sc in SCENARIOS:
        ref = run_case(sc(), None, Modules.parse("core"))
        for x in ("mm", "trace", "typing", "load", "text", "chain"):
            r = run_case(sc(), None, Modules.parse(f"core,-{x}"))
            if (r.predicted_entity_ids, r.predicted_fault_type, r.reasoning_chain, round(r.confidence, 9),
                    r.retrieval_stats.get("top_hyp")) != (ref.predicted_entity_ids, ref.predicted_fault_type, ref.reasoning_chain,
                                                         round(ref.confidence, 9), ref.retrieval_stats.get("top_hyp")):
                changed.add(x)
    assert changed >= {"trace", "load", "chain"}, changed
    print("   modules that changed an output:", sorted(changed))


@test
def belief_llm_adds_only_the_belief_block_to_the_evidence_prompt():
    """Belief->LLM: the Evidence->LLM prompt + the statistical ranking; the LLM's free answer is used (here: it overrides the type)."""
    from baselines.belief_llm import run_belief_llm
    from gama.run import run_case

    class Rec:
        def __init__(self):
            self.prompts = []
        def reset_usage(self):
            pass
        def usage_stats(self):
            return {"total_calls": len(self.prompts)}
        def generate_json(self, prompt, system=None):
            self.prompts.append(prompt)
            return {"root_cause_entity": "checkout", "predicted_fault_type": "F011-codeDefect", "reasoning_chain": ["cause: x"],
                    "confidence": 0.5}
    ref = run_case(sc_ratelimit(), None, Modules.parse("core"))
    llm = Rec()
    r = run_belief_llm(sc_ratelimit(), llm)
    p = llm.prompts[0]
    assert len(llm.prompts) == 1 and "Statistical diagnosis" in p and "P(root)" in p and "keeping the statistical model's answer" in p
    assert f"fault type = {ref.predicted_fault_type}" in p, p[-1500:]
    assert p.index("Statistical diagnosis") < p.index("Task:") and "http429" in p          # the evidence block is there too
    assert r.predicted_fault_type == "F011-codeDefect" and r.retrieval_stats["belief_type"] == ref.predicted_fault_type
    assert r.retrieval_stats["belief_kept_type"] is False


def test_suite():
    for fn in T:
        fn()


if __name__ == "__main__":
    if "--regen" in sys.argv:
        regen()
        sys.exit(0)
    bad = 0
    for fn in T:
        try:
            fn()
            print("ok  ", fn.__name__)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("FAIL", fn.__name__, str(e)[:2000])
            bad += 1
    print(f"{len(T) - bad}/{len(T)} passed")
    sys.exit(1 if bad else 0)
