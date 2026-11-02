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


@test
def hostcpu_errclass_tiebreak():
    import numpy as np
    from pipeline import belief_v4 as b4
    hosts, db = {"inventory": ["node-a"]}, {"nodes": {"node-a": {}}}
    o_node = {"entity": "node-a", "signal": "node_cpu_rise", "holds": True, "kind": "cont"}
    h_mem = {"entity": "inventory", "level": "service", "type": "F007-memoryPressure", "q": 0.8}
    h_cpu = {"entity": "inventory", "level": "service", "type": "F034-cpuFullLoad", "q": 0.8}
    m0, m1 = b4.Model(list(b4.DEFAULT_PARTS)), b4.Model(list(b4.DEFAULT_PARTS) + ["hostcpu"])
    m1.cpu_up = {"inventory"}
    assert m0.p_holds(o_node, h_mem, {}, {}, hosts, db) == 0.05 and m0.p_holds(o_node, h_cpu, {}, {}, hosts, db) == 0.6
    assert m1.p_holds(o_node, h_mem, {}, {}, hosts, db) == 0.6          # memory pressure with a CPU rise also heats the host
    m1.cpu_up = set()
    assert m1.p_holds(o_node, h_mem, {}, {}, hosts, db) == 0.05
    m2 = b4.Model(list(b4.DEFAULT_PARTS) + ["torigin2", "errclass"])
    o_err = {"entity": "inventory", "signal": "trace_err_origin", "holds": True, "kind": "cat"}
    h_npe = {"entity": "inventory", "level": "service", "type": "F023-nullPointerException", "q": 0.8}
    bg = {("trace_err_origin", False): 0.05}
    assert m2.p_holds(o_err, h_npe, bg, {}, hosts, db) == 0.6 and m2.p_holds(o_err, h_cpu, bg, {}, hosts, db) == 0.05
    o_slow = {**o_err, "signal": "trace_slow_origin"}
    assert m2.p_holds(o_slow, h_cpu, {("trace_slow_origin", False): 0.05}, {}, hosts, db) == 0.6
    post = np.array([0.1, 0.4, 0.4, 0.1])
    assert list(np.argsort(-post, kind="stable"))[:2] == [1, 2]                  # tie: the first listed wins
    order = list(b4.Model(list(b4.DEFAULT_PARTS)).fp)
    assert order.index("F034-cpuFullLoad") < order.index("F012-cpuDeadLoop")


@test
def depconseq_jvm_signals_are_consequences_of_slow_dependencies():
    from pipeline import belief_v4 as b4
    m0, m1 = b4.Model(list(b4.DEFAULT_PARTS)), b4.Model(list(b4.DEFAULT_PARTS) + ["depconseq"])
    for t in b4.DEP_LATENCY_TYPES:
        for a in b4.DEP_CONSEQUENCES:
            assert m1.roles[t].get(a) in ("M", "U", "K"), (t, a)                # explained when present
    assert m0.roles["F010-slowSQL"].get("svc_thread_rise") is None
    assert m1.roles["F010-slowSQL"]["svc_thread_rise"] == "M" and m1.roles["F010-slowSQL"]["db_slow"] == "K"
    assert m1.roles["F007-memoryPressure"] == m0.roles["F007-memoryPressure"]     # other types unchanged
    hosts, db = {"inventory": ["node-a"]}, {"nodes": {}}
    o = {"entity": "inventory", "signal": "svc_thread_rise", "holds": True, "kind": "cat"}
    h = {"entity": "inventory", "level": "service", "type": "F010-slowSQL", "q": 0.8}
    bg = {("svc_thread_rise", False): 0.05}
    assert m1.p_holds(o, h, bg, {}, hosts, db) > m0.p_holds(o, h, bg, {}, hosts, db)


@test
def depbase_history_test_needs_rank_and_effect():
    from pipeline.dep_latency import dep_kind, slow_test
    assert dep_kind("SELECT inventory.inventory", '{"component.name":"mysql"}') == "db"
    assert dep_kind("HikariDataSource.getConnection", '{"otel.scope.name":"io.opentelemetry.hikaricp"}') == "db"     # pool wait
    assert dep_kind("HGET", '{"otel.scope.name":"OpenTelemetry.Instrumentation.StackExchangeRedis"}') == "redis"
    assert dep_kind("orders receive", '{"messaging.system":"kafka"}') == "queue"
    assert dep_kind("GET /api/cart", "{}") == "other"
    hist = [0.4] * 4000 + [0.5] * 500
    assert slow_test([5000.0, 5001.0, 5000.5, 5001.2], hist)[0]                         # timeouts: few spans, huge shift
    assert slow_test([0.4] * 900 + [1.0] * 300, hist)[0]                                # upper tail doubles
    tiny = slow_test([0.4] * 3500 + [0.5] * 600, [0.4] * 4000 + [0.5] * 400)           # large n, ~no change: rank test alone fires
    assert tiny[1] <= 0.05 and not tiny[0]
    assert slow_test([0.4, 0.5], hist) is None                                          # too few spans: no observation


@test
def depshare_separates_slow_dependency_from_starved_service():
    from pipeline.dep_latency import share_rise
    n = 1000
    assert share_rise(5000.0 * 6, 5003.0 * 6, 0.45 * n, 2.0 * n)[0]                   # DB timeouts: the time is in the calls
    assert not share_rise(1.0 * n, 50.0 * n, 0.45 * n, 2.0 * n)[0]                    # GC / CPU starvation: calls 2x, requests 25x
    assert share_rise(1.0, 0.0, 1.0, 1.0) is None                                     # no request spans: no comparison


@test
def sigorder_first_mover_decides_among_jvm_types():
    import math
    from pipeline import belief_v4 as b4
    m = b4.Model(list(b4.DEFAULT_PARTS) + ["sigorder"])
    types = ["F034-cpuFullLoad", "F012-cpuDeadLoop", "F007-memoryPressure", "F022-fullGC", "F002-threadExhaustion", "F010-slowSQL"]
    H = [{"entity": "inventory", "level": "service", "type": t, "q": 0.8} for t in types]
    sig = lambda onset, p=0.01: {"p": p, "onset": onset}
    cpu_first = {"inventory": {"svc_cpu_rise": sig(90), "svc_gc_rise": sig(120), "svc_thread_rise": sig(120)}}   # CPU saturation
    ll = {t: round(float(v), 6) for t, v in zip(types, b4.sigorder_loglik(H, cpu_first, m))}
    up, down = round(math.log(0.9 / 0.5), 6), round(math.log(0.1 / 0.5), 6)
    assert ll["F034-cpuFullLoad"] == up and ll["F012-cpuDeadLoop"] == up
    assert ll["F022-fullGC"] == down and ll["F007-memoryPressure"] == down and ll["F002-threadExhaustion"] == down
    assert ll["F010-slowSQL"] == 0.0                                                     # not a mechanism type: neutral
    leak = {"inventory": {"svc_cpu_rise": sig(60), "svc_gc_rise": sig(60), "svc_mem_rise": sig(None), "svc_thread_rise": sig(180)}}
    ll = {t: round(float(v), 6) for t, v in zip(types, b4.sigorder_loglik(H, leak, m))}
    assert ll["F007-memoryPressure"] == 0.0 and ll["F022-fullGC"] == up and ll["F002-threadExhaustion"] == down   # gradual: unknown
    none = {"inventory": {"svc_cpu_rise": sig(None)}}
    assert not any(b4.sigorder_loglik(H, none, m))                                       # no onset at all: neutral
    notheld = {"inventory": {"svc_cpu_rise": sig(30, p=0.5), "svc_gc_rise": sig(60)}}    # a non-holding signal does not count
    assert round(float(dict(zip(types, b4.sigorder_loglik(H, notheld, m)))["F022-fullGC"]), 6) == up


@test
def loadx2_load_causes_latency_and_heats_the_host():
    from pipeline import belief_v4 as b4
    obs = [{"entity": e, "signal": sg, "holds": True} for e, sg in
           (("frontend", "svc_request_rise"), ("recommendation", "svc_request_rise"), ("recommendation", "svc_cpu_rise"),
            ("recommendation", "svc_latency_rise"), ("inventory", "svc_cpu_rise"))]
    callees = {"frontend": ["recommendation", "cart"], "cart": ["inventory"]}
    hosts = {"recommendation": ["node-r"], "inventory": ["node-i"]}
    v1 = b4.load_explained(obs, callees)
    v2 = b4.load_explained(obs, callees, signals=b4.LOAD_SIGNALS + ("svc_latency_rise",), hosts=hosts)
    assert ("recommendation", "svc_latency_rise") not in v1 and ("recommendation", "svc_latency_rise") in v2
    assert ("node-r", "node_cpu_rise") in v2 and ("node-i", "node_cpu_rise") not in v2       # inventory's CPU is not load
    assert not any(e == "inventory" for e, _ in v2) and not any(e == "frontend" for e, _ in v2)   # no rising caller


@test
def msgtarget_relays_to_the_endpoint_named_in_the_message():
    from pipeline.trace_structure import analyse, named_target
    svcs = ["frontend", "checkout", "shipping", "quote"]
    m1 = "Request error: error sending request for url (http://quote:8080/getquote): error trying to connect: Connection refused"
    assert named_target(m1, svcs, "checkout") == "quote"
    assert named_target("dial tcp quote.default.svc.cluster.local:8080: connect: connection refused", svcs, "checkout") == "quote"
    assert named_target("dial tcp 192.168.132.109:8080: connect: connection refused", svcs, "checkout") == ""   # an IP names nothing
    assert named_target("http://checkout:8080/x refused", svcs, "checkout") == ""                               # its own name
    spans = []
    for i in range(12):                       # checkout -> shipping (reached, no error) ; shipping could not reach quote
        t = f"q{i}"
        a = _sp(t, f"{t}a", None, "checkout", 0, 1.0, True, "/oteldemo.ShippingService/GetQuote")
        a["msg"] = m1
        spans += [a, _sp(t, f"{t}b", f"{t}a", "shipping", 0.1, 0.5, False, "GET /getquote")]
    svcs_all = svcs
    plain = analyse(spans, 10, 5, services=svcs_all)
    named = analyse(spans, 10, 5, services=svcs_all, msgtarget=True)
    assert plain["err_origin"] == {"checkout": 12}                      # reached shipping: torigin3 keeps the caller
    assert named["err_origin"] == {"quote": 12} and named["relayed_to_callee"] == {"quote": 12}


@test
def rpchain_prefers_the_symptomatic_path():
    from pipeline.belief_rca import call_path
    callees = {"frontend": ["checkout", "cart"], "checkout": ["shipping"], "cart": ["shipping"], "shipping": ["quote"]}
    assert call_path(callees, {}, "frontend", "quote", held={"cart"}) == ["quote", "shipping", "cart", "frontend"]
    assert call_path(callees, {}, "frontend", "quote", held={"checkout"}) == ["quote", "shipping", "checkout", "frontend"]
    assert len(call_path(callees, {}, "frontend", "quote")) == 4                       # without the part: any shortest path


@test
def redchain_cites_step_values_and_step_anchored_rp():
    import datetime as dt
    from types import SimpleNamespace
    from schema import Observation
    from pipeline.belief_rca import chain_step_metrics, rule_chain
    from evaluation.scoring import step_checkpoint_rate
    w0 = dt.datetime(2026, 1, 1, 12, 0, 0)
    w1 = w0 + dt.timedelta(minutes=5)
    obs = []
    def m(eset, name, metric, base, win):
        for i in range(20):
            obs.append(Observation(None, w0 - dt.timedelta(seconds=30 * (i + 1)), "metrics", "",
                                   {"entity_set": eset, "entity_name": name, "metric": metric, "value": base}))
        for i in range(10):
            obs.append(Observation(None, w0 + dt.timedelta(seconds=30 * i), "metrics", "",
                                   {"entity_set": eset, "entity_name": name, "metric": metric, "value": win}))
    m("apm.service", "payment", "error_count", 2.0, 8900.0)
    m("apm.service", "payment", "request_count", 100.0, 100.0)
    m("apm.service", "checkout", "error_count", 1.0, 8850.0)
    m("apm.service", "checkout", "cpu_usage", 0.1, 0.9)              # not a golden signal of a service step
    m("apm.operation", "checkout::PlaceOrder", "request_count", 900.0, 20.0)   # a drop: the window MINIMUM is cited
    m("apm.service", "payment", "deployment_cpu_usage_vs_requests", 1.0, 30.0)      # a CPU ratio, not a request count
    case = SimpleNamespace(observations={"metrics": obs})
    sm = chain_step_metrics(case, w0, w1, ["payment", "payment", "checkout"], "checkout::PlaceOrder")
    assert dict(sm["payment"])["error_count"] == 8900.0 and "cpu_usage" not in dict(sm["checkout"])
    assert "deployment_cpu_usage_vs_requests" not in dict(sm["payment"])
    assert dict(sm["checkout::PlaceOrder"])["request_count"] == 20.0
    res = {"entity": "payment", "type": "F014-httpError5xx", "evidence": []}
    chain = rule_chain(res, ["payment", "checkout"], "checkout::PlaceOrder", sm)
    assert "error_count=8900" in chain[0] and "error_count=8850" in chain[1]
    gt = SimpleNamespace(checkpoints=[
        {"target": "payment", "signal": "error_count", "comparator": ">=", "value": 8829},
        {"target": "checkout", "signal": "error_count", "comparator": ">=", "value": 8828},
        {"target": "checkout :: PlaceOrder", "signal": "request_count", "comparator": "<=", "value": 50}])
    assert step_checkpoint_rate(chain, gt) == 1.0
    wrong = rule_chain({"entity": "cart", "type": "F014-httpError5xx", "evidence": []}, ["cart"], None,
                       {"cart": [("error_count", 9000.0)]})
    assert step_checkpoint_rate(wrong, gt) == 0.0                     # right values at the wrong step do not count


if __name__ == "__main__":
    ok = 0
    for fn in T:
        try:
            fn()
            ok += 1
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"{ok}/{len(T)} passed")
