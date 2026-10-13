"""tests/prompt_snapshot.py -- equivalence test for the agent prompts (no data, no model).

Builds a synthetic AgentState, runs the specialist and Coordinator nodes against a stub LLM
that records every prompt/system string, for several configurations, and writes (or compares)
md5 + length per prompt. Used to prove that a refactor/cleanup leaves the prompts of the
retained configurations byte-identical.

    python tests/prompt_snapshot.py write tests/snapshots/x.json
    python tests/prompt_snapshot.py compare tests/snapshots/x.json
"""
import hashlib, json, os, sys, types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:                                   # heavy optional dependency, not needed for prompt building
    import sentence_transformers      # noqa: F401
except Exception:
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = type("SentenceTransformer", (), {})
    sys.modules["sentence_transformers"] = st

import config
from agents import multi_agent as ma


class StubLLM:
    temperature = 0.0

    def __init__(self):
        self.calls = []

    def _canned(self, system):
        if system and "fault-typing" in system:
            return {"fault_evidence": "cart 500 x12", "predicted_fault_type": "F029-redisUnavailable", "confidence": 0.6}
        if system and "Coordinator" in system:
            return {"reasoning_chain": ["cause: a cpu=1", "propagation: b err=2", "impact: c lat=3"],
                    "predicted_entity_ids": ["a"], "predicted_fault_type": "F014-httpError5xx", "confidence": 0.5}
        return {"entity_id": "svc-a", "summary": "s", "supporting_evidence": ["e1"], "confidence": 0.5,
                "segment": ["cause: svc-a cpu=0.9"]}

    def generate_json(self, prompt, system=None):
        self.calls.append((prompt, system))
        if "Respond as JSON: {\"observations\"" in prompt:
            return {"observations": ["cart cpu=0.9"]}
        return self._canned(system)

    def generate(self, prompt, system=None, **kw):
        self.calls.append((prompt, system))
        return json.dumps(self._canned(system))

    def usage_stats(self): return {}
    def reset_usage(self): pass


def make_state():
    return {
        "case_id": "tXXX",
        "alert_text": "frontend响应时间突增告警 latency spike",
        "evidence_summary": {
            "frontend": ["Metric observed: frontend cpu=0.25", "Metric observed: frontend latency=0.13",
                         "Log observed: connection refused to cart", "Error/exception patterns in frontend: timeout x12",
                         "3 span(s) with error/timeout status on frontend->cart", "Event: ScalingReplicaSet x2"],
            "cart": ["Metric observed: cart cpu=0.9", "Log pattern: slow query", "12 span(s) with error/timeout status"],
            "node-1": ["Metric observed: node-1 cpu_usage=95"],
        },
        "graph_neighbors": {"upstream": ["web"], "downstream": ["cart", "inventory"]},
        "candidate_entities": ["frontend", "cart", "node-1"],
        "topology_view": None,
        "graph_anchor": {"anchor_entity_ids": ["cart"], "anchor_fault_type": "F010-slowSQL", "anchor_confidence": 0.8},
        "propagation_path": ["frontend", "cart"],
        "keyword_fault_candidates": ["F010-slowSQL"],
        "zeroshot_suggested_type": "F010-slowSQL", "zeroshot_similarity": 0.3,
        "zeroshot_topk": [("F010-slowSQL", 0.3), ("F014-httpError5xx", 0.2)],
        "twist_top_entity": "cart", "twist_top_score": 0.7,
        "infra_text": "Infra table:\n- node-1 cpu base=8 max=95\n- cart-deploy replicas 3->1",
        "trace_text": "Trace errors:\n- cart 500 x12",
        "propagation_text": "Call-graph neighbourhood:\n- frontend (alert service) error_rate 0.0%\n- cart (downstream callee, 1 hop) error_rate 4%",
        "metrics_finding": None, "logs_finding": None, "trace_finding": None, "topology_finding": None,
    }


# Overrides applied on top of a neutral base; every other flag keeps its config default.
NEUTRAL = dict(GROUNDED_FAULT=False, CHAIN_FIRST=True, USE_KNOWLEDGE_POS=False, CHAIN_ABSOLUTE_SIGNALS=False,
               STRICT_MODALITY_ROUTING=False, SPECIALIST_ENGLISH=False, SPEC_CHAIN_SEGMENTS=False,
               USE_KEYWORD_HINT=False, TOPOLOGY_NAMES=False, ENTITY_TYPING=False, COORD_LEAN=False, NO_SPECIALISTS=False, SPEC_EVIDENCE_ONLY=False)
CONFIGS = {
    "frozen_ABS_KP": dict(USE_KNOWLEDGE_POS=True, CHAIN_ABSOLUTE_SIGNALS=True, STRICT_MODALITY_ROUTING=True, SPECIALIST_ENGLISH=True),
    "frozen_ABS_KP_SEG": dict(USE_KNOWLEDGE_POS=True, CHAIN_ABSOLUTE_SIGNALS=True, STRICT_MODALITY_ROUTING=True, SPECIALIST_ENGLISH=True, SPEC_CHAIN_SEGMENTS=True),
    "frozen_ABS_KP_P1": dict(USE_KNOWLEDGE_POS=True, CHAIN_ABSOLUTE_SIGNALS=True, STRICT_MODALITY_ROUTING=True, SPECIALIST_ENGLISH=True, GROUNDED_FAULT=True),
    "frozen_ABS_KP_TYPE": dict(USE_KNOWLEDGE_POS=True, CHAIN_ABSOLUTE_SIGNALS=True, STRICT_MODALITY_ROUTING=True, SPECIALIST_ENGLISH=True, ENTITY_TYPING=True),
    "frozen_ABS_KP_TYPECMP": dict(USE_KNOWLEDGE_POS=True, CHAIN_ABSOLUTE_SIGNALS=True, STRICT_MODALITY_ROUTING=True, SPECIALIST_ENGLISH=True, ENTITY_TYPING=True, TYPING_COMPARE=True),
    "frozen_ABS_KP_TYPELEAN": dict(USE_KNOWLEDGE_POS=True, CHAIN_ABSOLUTE_SIGNALS=True, STRICT_MODALITY_ROUTING=True, SPECIALIST_ENGLISH=True, ENTITY_TYPING=True, COORD_LEAN=True),
    "frozen_ABS_KP_NOSPEC": dict(USE_KNOWLEDGE_POS=True, CHAIN_ABSOLUTE_SIGNALS=True, STRICT_MODALITY_ROUTING=True, SPECIALIST_ENGLISH=True, NO_SPECIALISTS=True),
    "frozen_ABS_KP_EVOBS": dict(USE_KNOWLEDGE_POS=True, CHAIN_ABSOLUTE_SIGNALS=True, STRICT_MODALITY_ROUTING=True, SPECIALIST_ENGLISH=True, SPEC_EVIDENCE_ONLY=True),
    "base_A1rT_fixed": dict(STRICT_MODALITY_ROUTING=True, SPECIALIST_ENGLISH=True),
    "chain_first_off_baseline_path": dict(CHAIN_FIRST=False),
}


def run_config(overrides):
    saved = {k: getattr(config, k) for k in {**NEUTRAL, **overrides} if hasattr(config, k)}
    try:
        for k, v in {**NEUTRAL, **overrides}.items():
            setattr(config, k, v)
        llm = StubLLM()
        state = make_state()
        if not config.NO_SPECIALISTS:
            for name, filt in (("metrics", "metric"), ("logs", "log"), ("trace", "span")):
                state = ma._make_agent_node(name, filt, llm)(state)
            state = ma.topology_agent_node(llm)(state)
        state = ma.coordinator_node(llm)(state)
        out = []
        for prompt, system in llm.calls:
            out.append({"prompt_md5": hashlib.md5(prompt.encode()).hexdigest(), "prompt_len": len(prompt),
                        "system_md5": hashlib.md5((system or "").encode()).hexdigest()})
        out.append({"final": json.dumps(state.get("final_result"), sort_keys=True, default=str)})
        return out
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def main():
    mode, path = sys.argv[1], sys.argv[2]
    snap = {name: run_config(ov) for name, ov in CONFIGS.items()}
    if mode == "write":
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(snap, open(path, "w"), indent=1, sort_keys=True)
        print("written", path, {k: len(v) - 1 for k, v in snap.items()}, "prompts")
        return
    old = json.load(open(path))
    bad = [k for k in CONFIGS if k in old and old[k] != snap[k]]
    new = [k for k in CONFIGS if k not in old]
    print("IDENTICAL for all configs in the snapshot" if not bad else f"DIFFERENT: {bad}", ("| not in snapshot: %s" % new) if new else "")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
