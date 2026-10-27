"""
smoke_synthetic_llm.py -- run the v2 system (hypothesis_rca) with the REAL local LLM on synthetic cases only. Needs no RCA100 data and
no labels: checks the LLM mechanics (prompt length, JSON parsing, constrained choice, chain, verifier, abstain path) before the frozen
held-out run. Scenarios come from tests/test_hypothesis_rca.py (fake topology frontend -> checkout -> cart; node-a, node-b).

    python scripts/smoke_synthetic_llm.py          (model settings from the environment, as in main.py)
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_hypothesis_rca import case                                      # noqa: E402
from test_calibrated import metric, wspan                                 # noqa: E402
from agents.factory import get_llm_client                                 # noqa: E402
from pipeline.hypothesis_rca import diagnose                              # noqa: E402


def scenarios():
    c1 = case()
    metric(c1, "k8s.node", "node-a", "node_cpu_usage_rate", 0.30, 0.95)
    c2 = case()
    metric(c2, "apm.service.legacy", "frontend", "rt", 0.1, 0.5)
    for i in range(6):
        wspan(c2, "checkout", i, err=True, code="429", msg="Too Many Requests")
    c3 = case()
    metric(c3, "k8s.node", "node-a", "node_cpu_usage_rate", 0.30, 0.95)
    for i in range(6):
        wspan(c3, "checkout", i, err=True, code="429", msg="Too Many Requests")
    c4 = case()                                                            # nothing admitted -> abstain path
    return [("node cpu", c1, "node-a:F026-nodeCpuHigh"), ("rate limit", c2, "checkout:F016-rateLimiting"),
            ("two hypotheses", c3, None), ("nothing", c4, None)]


def main():
    llm = get_llm_client()
    ok = 0
    for name, c, expect in scenarios():
        t0 = time.time()
        r = diagnose(c, llm)
        st = r.retrieval_stats
        got = f"{st.get('v2_entity')}:{r.predicted_fault_type}"
        good = (expect is None or got == expect) and bool(r.reasoning_chain)
        ok += good
        print(f"\n=== {name}: mode={st.get('v2_mode')} answer={got} expected={expect} choice_valid={st.get('v2_choice_valid')} "
              f"grounded={st.get('v2_grounded_before')}->{st.get('v2_grounded_after')} calls={st.get('total_calls')} "
              f"prompt_chars={st.get('v2_prompt_chars')} {time.time() - t0:.0f}s {'OK' if good else 'CHECK'}")
        for step in r.reasoning_chain:
            print("   ", step)
    print(f"\n{ok}/4 scenarios as expected (expected = the top hypothesis is the injected fault; the LLM may still pick another "
          f"admitted one in 'two hypotheses')")
    # v3: computed belief + LLM text sensor + narrator
    from pipeline.belief_rca import run_case
    print("\n##### v3 belief_rca (sensor budget 5) #####")
    ok3 = 0
    for name, c, expect in scenarios():
        t0 = time.time()
        r = run_case(c, llm, budget=5, narrate=True)
        st = r.retrieval_stats
        got = f"{st.get('v3_top', '').split('|')[0].rsplit(':', 1)[0]}"
        good = (expect is None or got == expect) and bool(r.reasoning_chain)
        ok3 += good
        print(f"\n=== {name}: answer={got} p={st.get('v3_answer_p')} p_h0={st.get('v3_p_h0')} probes={st.get('v3_probe_log')} "
              f"quote_ok={st.get('v3_sensor_quote_ok')} grounded={st.get('v3_grounded')} calls={st.get('total_calls')} "
              f"{time.time() - t0:.0f}s {'OK' if good else 'CHECK'}")
        for step in r.reasoning_chain:
            print("   ", step)
    print(f"\nv3: {ok3}/4 scenarios as expected")


if __name__ == "__main__":
    main()
