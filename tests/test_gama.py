"""tests/test_gama.py -- the refactored method (gama/) against the frozen development code (pipeline/belief_v4 via belief_rca.run_case):
identical outputs on synthetic scenarios for every module configuration (core, each leave-one-out, base) and, with a deterministic
stub LLM, identical prompts and outputs for the full configuration. No data, no model.

    python tests/test_gama.py
"""
import datetime as dt
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, HERE)

from test_calibrated import make_case, metric, wspan, W0
from equivalence import run_pair, diff, configs
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


@test
def module_spec_parses_and_maps_to_the_frozen_parts():
    assert Modules.parse("core").on() == CORE and not Modules.parse("core").llm
    assert Modules.parse("full").llm and Modules.parse("core,-trace").on() == tuple(x for x in CORE if x != "trace")
    assert Modules.parse("base").on() == () and Modules.parse("base,mm").on() == ("mm",)
    src = open(os.path.join(HERE, "..", "scripts", "kaggle_run.py"), encoding="utf-8").read()
    block = src.split("# %% Cell 3 ")[1].split("# %% Cell ")[0].split("\n", 1)[1]
    D = {"os": os, "DATA": "/nonexistent"}
    try:                                                   # the arms are defined before the case lists (which need the data)
        exec(block, D)
    except FileNotFoundError:
        pass
    assert set(Modules.parse("core").legacy_parts()) == set(D["ARMS"]["GAMA_CORE"]["V4_PARTS"].split(","))
    assert set(Modules.parse("full").legacy_parts()) == set(D["ARMS"]["GAMA_LLM"]["V4_PARTS"].split(","))
    for mod in ("MM", "TRACE", "TYPE", "LOAD", "TEXT"):
        name = {"TYPE": "typing"}.get(mod, mod.lower())
        assert set(Modules.parse(f"core,-{name}").legacy_parts()) == set(D["ARMS"][f"GAMA_CORE_NO_{mod}"]["V4_PARTS"].split(",")), mod


@test
def identical_to_frozen_code_for_every_configuration():
    bad = []
    for sc in SCENARIOS:
        for name, m in configs(["core", "loo", "base"]):
            s1, s2 = run_pair(sc, m)
            d = diff(s1, s2)
            if d:
                bad.append((sc.__name__, name, d[:3]))
    assert not bad, bad


@test
def identical_prompts_and_outputs_with_the_llm_module():
    bad = []
    for sc in SCENARIOS:
        for name, m in configs(["llm", "full,-chain", "full,-typing"]):
            s1, s2 = run_pair(sc, m)
            assert s1["llm_calls"], (sc.__name__, "the stub LLM was never called")
            d = diff(s1, s2)
            if d:
                bad.append((sc.__name__, name, d[:3]))
    assert not bad, bad


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
                    r.retrieval_stats.get("v3_top")) != (ref.predicted_entity_ids, ref.predicted_fault_type, ref.reasoning_chain,
                                                         round(ref.confidence, 9), ref.retrieval_stats.get("v3_top")):
                changed.add(x)
    assert changed >= {"trace", "load", "chain"}, changed
    print("   modules that changed an output:", sorted(changed))


@test
def belief_llm_adds_only_the_belief_block_to_the_evidence_prompt():
    """Belief->LLM: the Evidence->LLM prompt + the statistical ranking; the LLM's free answer is used (here: it overrides the type)."""
    from gama.belief_llm import run_belief_llm
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


if __name__ == "__main__":
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
