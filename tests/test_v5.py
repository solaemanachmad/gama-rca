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


if __name__ == "__main__":
    ok = 0
    for fn in T:
        try:
            fn()
            ok += 1
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"{ok}/{len(T)} passed")
