"""tests/test_hypothesis_rca.py -- synthetic checks of the v2 hypothesis-constrained RCA (no data, no model).

    python tests/test_hypothesis_rca.py

Topology of the fake case: frontend -> checkout -> cart (calls); node-a hosts cart and frontend, node-b hosts checkout."""
import datetime as dt
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import config
from test_calibrated import make_case, metric, wspan
from pipeline.hypothesis_rca import diagnose


def case(alert_entity="s1"):
    c = make_case()
    c.case_id = "t999"
    c.alert.alert_text = "latency alert"
    c.alert.entry_entity_id = alert_entity
    return c


class Stub:
    def __init__(self, choice="H1", entity=None):
        self.choice, self.entity, self.prompts, self.total_calls = choice, entity, [], 0

    def reset_usage(self):
        pass

    def usage_stats(self):
        return {"total_calls": len(self.prompts)}

    def generate_json(self, prompt, system=None):
        self.prompts.append(prompt)
        if "Admitted hypotheses" in prompt:
            return {"choice": self.choice, "reasoning_chain": ["cause: x cpu=0.95", "impact: frontend latency"], "confidence": 0.7}
        if "Rewrite the chain" in prompt:
            return {"reasoning_chain": ["cause: x"]}
        return {"reasoning_chain": ["cause: cart error"], "predicted_entity": self.entity or "cart", "predicted_fault_type": "F014-httpError5xx"}


T = []


def test(fn):
    T.append(fn)
    return fn


@test
def node_fault_is_answered_at_node_level():
    c = case()
    metric(c, "k8s.node", "node-a", "node_cpu_usage_rate", 0.30, 0.95)
    r = diagnose(c, None, use_llm=False)
    assert r.retrieval_stats["v2_entity"] == "node-a" and r.predicted_fault_type == "F026-nodeCpuHigh", r.retrieval_stats
    assert r.predicted_entity_ids == ["n1"], r.predicted_entity_ids          # topology id of node-a


@test
def shared_node_rise_is_not_a_node_fault():
    """Both nodes hosting the services rise together (load), and checkout shows 429s: the service hypothesis must win."""
    c = case()
    metric(c, "k8s.node", "node-a", "node_cpu_usage_rate", 0.30, 0.95)
    metric(c, "k8s.node", "node-b", "node_cpu_usage_rate", 0.30, 0.95, seed=7)
    for i in range(6):
        wspan(c, "checkout", i, err=True, code="429", msg="Too Many Requests")
    r = diagnose(c, None, use_llm=False)
    assert (r.retrieval_stats["v2_entity"], r.predicted_fault_type) == ("checkout", "F016-rateLimiting"), r.retrieval_stats
    assert r.retrieval_stats["v2_n_diffuse"] == 2


@test
def downstream_categorical_root_wins():
    c = case()
    metric(c, "apm.service.legacy", "frontend", "rt", 0.1, 0.5)               # symptom at the alerted service
    for i in range(6):
        wspan(c, "checkout", i, err=True, code="429", msg="Too Many Requests")
    r = diagnose(c, None, use_llm=False)
    assert (r.retrieval_stats["v2_entity"], r.predicted_fault_type) == ("checkout", "F016-rateLimiting"), r.retrieval_stats


@test
def upstream_anomaly_cannot_explain_a_downstream_alert():
    c = case(alert_entity="s3")                                               # alert on cart; frontend is upstream of cart
    for i in range(6):
        wspan(c, "frontend", i, err=True, code="429", msg="Too Many Requests")
    r = diagnose(c, None, use_llm=False)
    assert r.retrieval_stats["v2_n_consistent"] == 0 and r.retrieval_stats["v2_mode"] == "abstain", r.retrieval_stats


@test
def ablation_no_topo_admits_it():
    c = case(alert_entity="s3")
    for i in range(6):
        wspan(c, "frontend", i, err=True, code="429", msg="Too Many Requests")
    config.V2_NO_TOPO = True
    try:
        r = diagnose(c, None, use_llm=False)
    finally:
        config.V2_NO_TOPO = False
    assert r.predicted_fault_type == "F016-rateLimiting", r.retrieval_stats


@test
def llm_choice_is_constrained():
    c = case()
    metric(c, "k8s.node", "node-a", "node_cpu_usage_rate", 0.30, 0.95)
    for i in range(6):
        wspan(c, "checkout", i, err=True, code="429", msg="Too Many Requests")
    for choice, valid in (("H2", True), ("H9", False)):
        llm = Stub(choice)
        r = diagnose(c, llm)
        top = r.retrieval_stats["v2_top"].split("|")
        exp = top[1] if valid else top[0]
        assert f"{r.retrieval_stats['v2_entity']}:{r.predicted_fault_type}" == exp and r.retrieval_stats["v2_choice_valid"] == valid, \
            (choice, r.retrieval_stats)
        assert r.retrieval_stats["v2_mode"] == "select" and len(llm.prompts) >= 1


@test
def abstain_uses_one_free_call():
    c = case()
    llm = Stub(entity="cart")
    r = diagnose(c, llm)
    assert r.retrieval_stats["v2_mode"] == "abstain" and r.predicted_entity_ids == ["s3"] and r.predicted_fault_type == "F014-httpError5xx", \
        r.retrieval_stats


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
