"""tests/test_v5.py -- v5 protocol pieces (PREREGISTRATION_V5.md): EVIDENCE_LLM prompt, chat templates without a system role, the
odd/even case split, and a static no-label check of the agent-facing code. No data, no model.

    python tests/test_v5.py"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

T = []


def test(fn):
    T.append(fn)
    return fn


@test
def evidence_prompt_lists_only_given_evidence():
    from pipeline.evidence_llm import build_prompt
    p = build_prompt("cart latency alert", "cart", "cart::GetCart", {"frontend": ["cart"], "cart": ["redis"]},
                     {"cart": ["node-a"]}, {"redis": ["redis_err: 37 failing calls"], "cart": ["svc_latency_rise: 0.002 -> 0.019"]},
                     {"redis": ["ERR connection refused <*>"]})
    for s in ("frontend -> cart", "cart: node-a", "redis_err: 37 failing calls", "log: ERR connection refused", "F029-redisUnavailable",
              '"root_cause_entity"'):
        assert s in p, s
    assert "answer" not in p.lower().replace("answer_", "")


class _Tok:
    def __init__(self, allow_system):
        self.allow = allow_system

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        if not self.allow and any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


@test
def system_role_fallback():
    from agents.kaggle_client import render_chat, merge_system
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert render_chat(_Tok(True), msgs, False) == "system:S|user:U"
    t = _Tok(False)
    assert render_chat(t, msgs, False) == "user:S\n\nU" and t.system_merged
    assert merge_system([{"role": "user", "content": "U"}]) == [{"role": "user", "content": "U"}]


@test
def odd_even_split_is_a_partition():
    ids = [f"t{i:03d}" for i in range(1, 104)]
    odd = [c for c in ids if int(c[1:]) % 2 == 1]
    even = [c for c in ids if int(c[1:]) % 2 == 0]
    assert len(odd) == 52 and len(even) == 51 and not set(odd) & set(even) and set(odd) | set(even) == set(ids)
    src = open(os.path.join(ROOT, "scripts", "kaggle_run.py")).read()
    assert "DEV_ODD = [c for c in ALL_IDS if int(c[1:]) % 2 == 1]" in src and "VAL_EVEN" in src


@test
def agent_code_never_reads_labels():
    """Static check: no agent-facing module refers to the answer key or the ground-truth loader in CODE (docstrings and comments may
    mention it)."""
    mods = ["pipeline/belief_v4.py", "pipeline/belief_rca.py", "pipeline/belief_diagnosis.py", "pipeline/fault_v4.py",
            "pipeline/fault_calibrated.py", "pipeline/llm_typing.py", "pipeline/semantic_probes.py", "pipeline/log_preprocess.py",
            "pipeline/evidence_llm.py", "data/taxonomy.py", "data/loader.py"]
    bad = []
    for m in mods:
        tree = ast.parse(open(os.path.join(ROOT, m)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and isinstance(getattr(node, "parent", None), ast.Expr):
                continue
            if isinstance(node, ast.Name) and node.id in ("load_ground_truth", "ANSWER_KEY_DIR"):
                bad.append((m, node.id))
            if isinstance(node, ast.Attribute) and node.attr in ("load_ground_truth", "ANSWER_KEY_DIR"):
                bad.append((m, node.attr))
            if isinstance(node, (ast.Import, ast.ImportFrom)) and "evaluation" in (getattr(node, "module", "") or ""):
                bad.append((m, "imports evaluation"))
    assert not bad, bad


@test
def llm_calls_are_traced_as_json_lines():
    import json, tempfile
    from agents.trace import install_llm_trace, write_case_trace, stage_of
    from pipeline.semantic_probes import SENSOR_SYSTEM
    from pipeline.llm_typing import TYPING_SYSTEM
    from pipeline.belief_rca import NARRATOR_SYSTEM
    from pipeline.evidence_llm import EVIDENCE_SYSTEM
    assert [stage_of(x) for x in (SENSOR_SYSTEM, TYPING_SYSTEM, NARRATOR_SYSTEM, EVIDENCE_SYSTEM, "x")] == \
        ["sensor", "typing", "narrator", "evidence_llm", "other"]

    class Fake:
        total_tokens_used, total_calls = 0, 0

        def generate(self, prompt, system=None, json_mode=False):
            self.total_tokens_used += 7
            self.total_calls += 1
            return '{"category": "redis", "quote": "ERR"}'

        def generate_json(self, prompt, system=None):
            return json.loads(self.generate(prompt, system=system, json_mode=True))
    with tempfile.TemporaryDirectory() as td:
        llm = Fake()
        install_llm_trace(llm, f"{td}/llm_calls.jsonl")
        llm.trace_ctx = {"case_id": "t001", "system": "belief_v4"}
        assert llm.generate_json("messages ...", system=SENSOR_SYSTEM)["category"] == "redis"
        rec = json.loads(open(f"{td}/llm_calls.jsonl").read().splitlines()[0])
        assert rec["case_id"] == "t001" and rec["stage"] == "sensor" and rec["prompt"] == "messages ..." and rec["tokens_total_delta"] == 7
        p = write_case_trace(td, "t001", "belief_v4", {"pi": 0.4}, {"final_score_nodes": 0.5})
        assert json.load(open(p))["trail"]["pi"] == 0.4


@test
def multimodal_corroboration():
    from pipeline import belief_v4 as b4
    assert [b4.modality(x) for x in ("svc_mem_rise", "ev_oom", "http429", "redis_err", "log_npe_rise", "msg_dns", "svc_unavailable",
                                     "message")] == ["metrics", "events", "traces", "traces", "logs", "logs", "metrics", "logs"]
    model = b4.Model(list(b4.DEFAULT_PARTS) + ["mm"])
    H = [{"entity": "inventory", "level": "service", "type": "F007-memoryPressure"},
         {"entity": "recommendation", "level": "service", "type": "F034-cpuFullLoad"},
         {"entity": "recommendation", "level": "service", "type": b4.ANOM},
         {"entity": "checkout", "level": "service", "type": "F014-httpError5xx"},
         {"entity": None, "level": None, "type": None}]
    obs = [{"entity": "inventory", "signal": "svc_mem_rise", "holds": True, "kind": "cont"},
           {"entity": "inventory", "signal": "ev_oom", "holds": True, "kind": "cat"},          # memory (metrics) + OOM kill (events)
           {"entity": "recommendation", "signal": "svc_cpu_rise", "holds": True, "kind": "cont"},
           {"entity": "recommendation", "signal": "svc_mem_rise", "holds": True, "kind": "cont"},   # noisy, but metrics only
           {"entity": "checkout", "signal": "http5xx", "holds": True, "kind": "cat"},             # generic symptoms do not corroborate
           {"entity": "checkout", "signal": "svc_error_rise", "holds": True, "kind": "cont"}]
    out, beta, k = b4.mm_loglik(model, H, obs)
    assert list(k) == [2, 1, 0, 0, 0] and out[0] > 0 and list(out[1:]) == [0, 0, 0, 0]
    assert abs(out[0] - __import__("math").log(b4.THETA_M / beta)) < 1e-9 and beta >= b4.MM_B0
    assert "mm" not in b4.DEFAULT_PARTS and "xcard" not in b4.DEFAULT_PARTS


@test
def xcard_signature_text():
    from pipeline import belief_v4 as b4
    from pipeline.llm_typing import signature_card
    fp = b4.Model(list(b4.DEFAULT_PARTS) + ["xcard"]).fp
    c = signature_card(fp, "F007-memoryPressure")
    assert "svc_mem_rise [metrics]" in c and "ev_oom [events]" in c
    assert "http429 [traces]" in signature_card(fp, "F016-rateLimiting")


if __name__ == "__main__":
    ok = 0
    for fn in T:
        try:
            fn()
            ok += 1
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"{ok}/{len(T)} passed")
