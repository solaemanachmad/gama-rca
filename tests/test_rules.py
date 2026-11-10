"""v6 rules (PREREGISTRATION_V6.md) on synthetic scenarios with scripted LLMs (no data, no model):
G1 gate: a second opinion is asked only when P(e*) < STOP_P, and its entity is typed by the model;
G2 defer: a type change quoting a signal the computed type also expects is rejected;
G3 calsensor: a probe re-weights the probed entity's types and leaves P(entity) unchanged.
v5 configurations are unchanged (tests/test_gama.py golden file).

    python tests/test_v6.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

from test_gama import sc_load_and_cpu, sc_trace_origin, sc_node  # noqa: E402
from gama.config import Modules  # noqa: E402

T = []


def test(fn):
    T.append(fn)
    return fn


class Scripted:
    """Answers by prompt kind; records the calls."""

    def __init__(self, entity=None, fault_type=None, quote_signal=None):
        self.entity, self.fault_type, self.quote_signal, self.calls = entity, fault_type, quote_signal, []
        self.reset_usage()

    def reset_usage(self):
        self.total_calls = 0

    def usage_stats(self):
        return {"total_calls": self.total_calls}

    def generate_json(self, prompt, system=None):
        self.calls.append(prompt)
        self.total_calls += 1
        if '"root_cause_entity"' in prompt:
            return {"root_cause_entity": self.entity or "", "predicted_fault_type": "x", "reasoning_chain": []}
        if '"fault_type"' in prompt:
            lines = re.findall(r"^- (.+)$", prompt, flags=re.M)
            q = next((l for l in lines if self.quote_signal and f" {self.quote_signal}:" in l), "")
            return {"fault_type": self.fault_type or "", "evidence_line": q}
        if "Which ONE category" in prompt:
            return {"category": "none", "quote": ""}
        return {}


def _state(sc, spec="full"):
    from gama import evidence
    from gama.chain import callee_map
    from gama.decide import diagnose
    import config
    config.ALERT_T0_MODE = "window"
    c = sc()
    m = Modules.parse(spec)
    db = evidence.signal_db(c, c.alert.alert_timestamp, m)
    dbg = {}
    res = diagnose(db, callee_map(c.topology), {}, m, alert=None, case=c, debug=dbg)
    return c, db, m, res, dbg


@test
def spec_v6_adds_only_the_three_rules():
    v6, full = Modules.parse("v6"), Modules.parse("full")
    assert set(v6.on()) - set(full.on()) == {"gate", "defer", "calsensor"}
    assert not (set(full.on()) & {"gate", "defer", "calsensor"}) and Modules.parse("core").on() == Modules.parse("core").on()
    assert v6.legacy_parts() == full.legacy_parts()


@test
def gate_asks_only_when_unsure_and_types_the_new_entity_with_the_model():
    import gama.run as gr
    from gama.run import run_case
    other = "checkout"
    saved = gr.STOP_P
    try:
        gr.STOP_P = 0.0                                   # never unsure -> no second opinion
        llm = Scripted(entity=other)
        r = run_case(sc_load_and_cpu(), llm, Modules.parse("hybrid"))
        st = r.retrieval_stats
        assert st["gate_called"] is False and not any('"root_cause_entity"' in p for p in llm.calls), st
        before = st["prior_entity"]
        gr.STOP_P = 1.01                                  # always unsure -> one second opinion
        target = other if before != other else "frontend"
        llm = Scripted(entity=target)
        r = run_case(sc_load_and_cpu(), llm, Modules.parse("hybrid"))
        st = r.retrieval_stats
        assert st["gate_called"] and st["gate_changed"] and st["gate_entity"] == target, st
        assert sum('"root_cause_entity"' in p for p in llm.calls) == 1
        assert "Statistical diagnosis" not in next(p for p in llm.calls if '"root_cause_entity"' in p)   # independent opinion
        assert r.retrieval_stats["_trace"]["decided"]["entity"] == target
        assert r.predicted_fault_type != "x"              # the LLM's type is not used: the model types the entity
        llm = Scripted(entity="no-such-entity")           # unknown name -> computed decision stands
        r = run_case(sc_load_and_cpu(), llm, Modules.parse("hybrid"))
        assert r.retrieval_stats["gate_changed"] is False and r.retrieval_stats["_trace"]["decided"]["entity"] == before
    finally:
        gr.STOP_P = saved


@test
def defer_rejects_a_type_change_that_quotes_a_shared_signal():
    from pipeline.llm_typing import llm_type
    c, db, m, res, dbg = _state(sc_load_and_cpu)
    model = dbg["model"]
    comp = res["type"]
    cands = [t for t, _ in res["type_rank"]][:5]
    shared = None                                         # a shortlist type whose fingerprint also uses a K/U signal of comp
    for t in cands:
        if t == comp:
            continue
        for sgn, role in model.roles.get(comp, {}).items():
            if role in ("K", "U") and model.roles.get(t, {}).get(sgn) in ("K", "U", "M") and sgn in db["_cal"].get(res["entity"], {}):
                shared = (t, sgn)
                break
        if shared:
            break
    assert shared, (comp, cands)
    t, sgn = shared
    a = llm_type(Scripted(fault_type=t, quote_signal=sgn), db, res, {}, model=model, xcard=False, typecheck=True)
    b = llm_type(Scripted(fault_type=t, quote_signal=sgn), db, res, {}, model=model, xcard=False, typecheck=True, discrim=True)
    if a["accepted"]:                                     # v5 accepts the change; v6 must reject it
        assert a["changed"] and not b["accepted"] and b["discriminating"] is False, (a, b)
    assert not b["accepted"], b


@test
def calsensor_keeps_the_probed_entitys_probability():
    from gama.decide import _probes
    from pipeline import belief_diagnosis as bd
    for sc in (sc_trace_origin, sc_node, sc_load_and_cpu):
        c, db, m, res, dbg = _state(sc)
        H, logp, model = dbg["H"], dbg["logp"], dbg["model"]
        ents = [e for e in {h["entity"] for h in H if h["entity"]}]
        texts = {e: ["some text"] for e in ents}
        cats = bd.CATS + ["none"]
        for cat in cats:
            def sensor(e, t, cat=cat):
                return cat
            lp0 = logp - 50.0 * (bd.posterior(logp).max() >= bd.STOP_P)   # force the loop to start
            flat = np.log(np.full(len(H), 1.0 / len(H)))                   # a flat posterior is surely below STOP_P
            for start in (flat,):
                new, log = _probes(model, H, start, texts, sensor, 1, 8, neutral=True)
                if not log:
                    continue
                e = log[0]["probe"]

                def marg(lp):
                    p = bd.posterior(lp)
                    out = defaultdict(float)
                    for h, x in zip(H, p):
                        out[h["entity"]] += float(x)
                    return out
                m0, m1 = marg(start), marg(new)
                assert abs(m0[e] - m1[e]) < 1e-9, (sc.__name__, cat, e, m0[e], m1[e])
                assert all(abs(m0[k] - m1[k]) < 1e-9 for k in m0), (sc.__name__, cat)
            del lp0


def test_suite():
    for fn in T:
        fn()


if __name__ == "__main__":
    bad = 0
    for fn in T:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except Exception as e:
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"{len(T) - bad}/{len(T)} passed")
    sys.exit(1 if bad else 0)
