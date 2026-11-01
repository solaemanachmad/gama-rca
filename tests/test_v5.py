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
            "pipeline/evidence_llm.py", "pipeline/trace_structure.py", "data/dataset_profile.py", "data/taxonomy.py", "data/loader.py"]
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


@test
def load_from_upstream_is_explained():
    from pipeline import belief_v4 as b4
    callees = {"frontend": ["recommendation", "ad", "cart"], "cart": ["redis"]}
    obs = [{"entity": "frontend", "signal": "op_request_rise", "holds": True},
           {"entity": "recommendation", "signal": "svc_request_rise", "holds": True},
           {"entity": "recommendation", "signal": "svc_cpu_rise", "holds": True},
           {"entity": "cart", "signal": "svc_cpu_rise", "holds": True},              # CPU without its own request rise: NOT explained
           {"entity": "redis", "signal": "svc_request_rise", "holds": True}]         # caller cart has no request rise: NOT explained
    le = b4.load_explained(obs, callees)
    assert ("recommendation", "svc_request_rise") in le and ("recommendation", "svc_cpu_rise") in le
    assert not any(e in ("cart", "redis", "frontend") for e, _ in le), le
    m = b4.Model(list(b4.DEFAULT_PARTS) + ["loadx"])
    db = {"nodes": {}, "_load_explained": le}
    o = {"entity": "recommendation", "signal": "svc_cpu_rise", "holds": True, "kind": "cont"}
    h0 = {"entity": None, "level": None, "type": None}
    hs = {"entity": "recommendation", "level": "service", "type": "F006-trafficSurge", "q": 0.8}
    assert m.p_holds(o, h0, {}, {}, {}, db) == 0.6                 # expected under no fault at recommendation
    assert m.p_holds({**o, "signal": "svc_request_rise"}, hs, {}, {}, {}, db) == 0.8   # still the key of trafficSurge there
    assert m.p_holds(o, h0, {}, {}, {}, {"nodes": {}}) == 0.05      # without loadx: the usual alpha


def _sp(tid, sid, parent, svc, start, dur, err=False, name="op"):
    return {"traceId": tid, "spanId": sid, "parentSpanId": parent, "service": svc, "start": start, "duration": dur, "error": err,
            "name": name}


@test
def trace_error_origin_is_the_deepest_failing_span():
    from pipeline.trace_structure import analyse
    spans = []
    for i in range(6):                                  # frontend -> checkout -> inventory; inventory fails, callers fail with it
        t = f"e{i}"
        spans += [_sp(t, f"{t}a", None, "frontend", 0, 1.0, True), _sp(t, f"{t}b", f"{t}a", "checkout", 0.1, 0.8, True),
                  _sp(t, f"{t}c", f"{t}b", "inventory", 0.2, 0.5, True)]
    for i in range(4):                                  # healthy traces
        t = f"ok{i}"
        spans += [_sp(t, f"{t}a", None, "frontend", 0, 0.2), _sp(t, f"{t}b", f"{t}a", "checkout", 0.05, 0.1)]
    r = analyse(spans)
    assert r["n_err_traces"] == 6 and r["err_origin"] == {"inventory": 6}, r
    assert ("inventory", "trace_err_origin") in r["holds"] and ("checkout", "trace_err_origin") not in r["holds"]


@test
def trace_slow_origin_uses_self_time_not_inclusive_latency():
    from pipeline.trace_structure import analyse
    spans = []
    for i in range(20):                                 # 18 fast traces: frontend self time dominates; 2... slow: cart self time
        t = f"t{i}"
        slow = i >= 16
        cart = 3.0 if slow else 0.01
        spans += [_sp(t, f"{t}a", None, "frontend", 0, 0.3 + cart, name="GET /"),
                  _sp(t, f"{t}b", f"{t}a", "checkout", 0.05, 0.05 + cart, name="PlaceOrder"),
                  _sp(t, f"{t}c", f"{t}b", "cart", 0.06, cart, name="GetCart")]
    r = analyse(spans)
    assert r["slow_origin"].get("cart", 0) >= 3 and ("cart", "trace_slow_origin") in r["holds"], r
    assert ("checkout", "trace_slow_origin") not in r["holds"] and ("frontend", "trace_slow_origin") not in r["holds"]


@test
def trace_origin_is_own_evidence_and_absence_is_uninformative():
    from pipeline import belief_v4 as b4
    m = b4.Model(list(b4.DEFAULT_PARTS) + ["torigin"])
    db = {"nodes": {"node-a": {}}}
    hosts = {"cart": ["node-a"]}
    o = {"entity": "cart", "signal": "trace_err_origin", "holds": True, "kind": "cat"}
    bg = {("trace_err_origin", False): 0.05}
    h_cart = {"entity": "cart", "level": "service", "type": "F029-redisUnavailable", "q": 0.8}
    h_node = {"entity": "node-a", "level": "node", "type": "F026-nodeCpuHigh", "q": 0.8}
    h_caller = {"entity": "checkout", "level": "service", "type": "F014-httpError5xx", "q": 0.8}
    expl = {"cart": {"checkout"}}                       # even if checkout could explain cart's symptoms, an origin is not a symptom
    assert m.p_holds(o, h_cart, bg, expl, hosts, db) == 0.6 and m.p_holds(o, h_node, bg, expl, hosts, db) == 0.6
    assert m.p_holds(o, h_caller, bg, expl, hosts, db) == 0.05
    assert m.may({**o, "holds": False}, h_cart, hosts) and not m.may({**o, "holds": False}, h_caller, hosts)


@test
def torigin2_node_needs_two_origin_services_and_counts():
    from pipeline import belief_v4 as b4
    from pipeline.trace_structure import analyse, origin_nodes
    spans = []
    for i in range(5):                                  # only 5 failing traces: an origin in v1 (>= 3), not in v2 (>= 10)
        t = f"e{i}"
        spans += [_sp(t, f"{t}a", None, "frontend", 0, 1.0, True), _sp(t, f"{t}b", f"{t}a", "cart", 0.1, 0.5, True)]
    assert ("cart", "trace_err_origin") in analyse(spans)["holds"]
    assert ("cart", "trace_err_origin") not in analyse(spans, 10, 5)["holds"]
    hosts = {"cart": ["node-a"], "frontend": ["node-a"], "checkout": ["node-b"]}
    one = [{"entity": "cart", "signal": "trace_err_origin", "holds": True}]
    two = one + [{"entity": "frontend", "signal": "trace_slow_origin", "holds": True}]
    assert origin_nodes(one, hosts) == set() and origin_nodes(two, hosts) == {"node-a"}
    m = b4.Model(list(b4.DEFAULT_PARTS) + ["torigin2"])
    o = {"entity": "cart", "signal": "trace_err_origin", "holds": True, "kind": "cat"}
    h_node = {"entity": "node-a", "level": "node", "type": "F026-nodeCpuHigh", "q": 0.8}
    h_cart = {"entity": "cart", "level": "service", "type": "F029-redisUnavailable", "q": 0.8}
    bg, db = {("trace_err_origin", False): 0.05}, {"nodes": {"node-a": {}}}
    m.origin_nodes = origin_nodes(one, hosts)
    assert m.p_holds(o, h_node, bg, {}, hosts, db) == 0.05 and m.p_holds(o, h_cart, bg, {}, hosts, db) == 0.6
    m.origin_nodes = origin_nodes(two, hosts)
    assert m.p_holds(o, h_node, bg, {}, hosts, db) == 0.6
    m1 = b4.Model(list(b4.DEFAULT_PARTS) + ["torigin"])                     # v1 unchanged: the hosting node always gets it
    assert m1.p_holds(o, h_node, bg, {}, hosts, db) == 0.6


@test
def origin_messages_are_read_where_the_failure_starts():
    from pipeline.trace_structure import analyse, origin_messages
    spans = []
    for i in range(12):            # inventory throttles (429 + 'rate limit'); checkout relays it as 500 with the same wording
        t = f"{i:032x}"
        spans += [{**_sp(t, f"{t}a", None, "checkout", 0, 1.0, True), "msg": "upstream: rate limit exceeded",
                   "attrs": '"http.status_code":"500"'},
                  {**_sp(t, f"{t}b", f"{t}a", "inventory", 0.1, 0.5, True), "msg": "rate limit exceeded",
                   "attrs": '"http.status_code":"429"'}]
    res = analyse(spans, 10, 5)
    om = origin_messages(res, logs=[("inventory", f"WARN throttled trace_id={0:032x} Too Many Requests")])
    assert ("inventory", "omsg_ratelimit") in om and ("inventory", "ohttp429") in om, om
    assert not any(k[0] == "checkout" for k in om), om               # the caller only relays the failure
    assert origin_messages(res, logs=None).keys() == om.keys()


@test
def cpu_deadloop_needs_starvation_evidence():
    from pipeline import belief_v4 as b4
    fp = b4.fingerprints_v4(set(b4.DEFAULT_PARTS) | {"cpuspec"})
    assert fp["F012-cpuDeadLoop"]["key"] == ["svc_cpu_rise&svc_request_drop"] and fp["F034-cpuFullLoad"]["key"] == ["svc_cpu_rise"]
    m = b4.Model(list(b4.DEFAULT_PARTS) + ["cpuspec"])
    obs = [{"entity": "cart", "signal": "svc_cpu_rise", "holds": True, "kind": "cont"}]
    H = [{"entity": "cart", "level": "service", "type": "F034-cpuFullLoad"}, {"entity": "cart", "level": "service", "type": "F012-cpuDeadLoop"},
         {"entity": None, "level": None, "type": None}]
    T = m.testable(H, obs)
    assert [h["type"] for h in T] == ["F034-cpuFullLoad", None]          # deadLoop untestable without the request drop
    T2 = m.testable(H, obs + [{"entity": "cart", "signal": "svc_request_drop", "holds": False, "kind": "cont"}])
    assert "F012-cpuDeadLoop" in [h["type"] for h in T2]


@test
def unreachable_callee_takes_the_origin_from_its_caller():
    """torigin3: shipping is scaled to zero, so checkout's client span fails with 'connection refused' and has no child span; the
    origin is shipping (named in the span), not checkout. A timeout with a (slow, non-failing) child goes to that child."""
    from pipeline.trace_structure import analyse, callee_of
    svcs = ["frontend", "checkout", "shipping", "email", "product-catalog", "frontend-proxy"]
    assert callee_of({"service": "checkout", "name": "oteldemo.ShippingService/GetQuote", "attrs": ""}, svcs) == "shipping"
    assert callee_of({"service": "checkout", "name": "POST", "attrs": '"server.address":"product-catalog"'}, svcs) == "product-catalog"
    assert callee_of({"service": "frontend-proxy", "name": "ingress", "attrs": ""}, svcs) == ""
    spans = []
    for i in range(12):
        t = f"r{i}"
        spans += [{**_sp(t, f"{t}a", None, "frontend", 0, 1.0, True), "msg": "rpc error"},
                  {**_sp(t, f"{t}b", f"{t}a", "checkout", 0.1, 0.5, True, name="oteldemo.ShippingService/GetQuote"),
                   "msg": "dial tcp 10.0.0.9:50051: connect: connection refused"}]
    for i in range(12):
        t = f"o{i}"
        spans += [{**_sp(t, f"{t}a", None, "frontend", 0, 6.0, True), "msg": "context deadline exceeded"},
                  {**_sp(t, f"{t}b", f"{t}a", "product-catalog", 0.1, 5.5, False, name="GetProduct"), "msg": ""}]
    v2 = analyse(spans, 10, 5)
    v3 = analyse(spans, 10, 5, services=svcs)
    assert v2["err_origin"].get("checkout") == 12 and "shipping" not in v2["err_origin"]
    assert v3["err_origin"].get("shipping") == 12 and "checkout" not in v3["err_origin"], v3["err_origin"]
    assert v3["err_origin"].get("product-catalog") == 12 and v3["relayed_to_callee"] == {"shipping": 12, "product-catalog": 12}
    assert ("shipping", "trace_err_origin") in v3["holds"]


if __name__ == "__main__":
    ok = 0
    for fn in T:
        try:
            fn()
            ok += 1
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"{ok}/{len(T)} passed")
