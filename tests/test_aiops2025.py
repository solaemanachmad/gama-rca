"""Synthetic AIOps2025 release (same file layout and schemas as the real one; no real data): adapter + GAMA under the aiops2025
profile. Checks the A4 tier rule (a fault in one pod -> the pod; in every pod -> the service), A7 trace sampling, and that ground
truth is never opened.

    python tests/test_aiops2025.py
"""
import os
import sys

os.environ["DATASET_PROFILE"] = "aiops2025"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import builtins       # noqa: E402
import datetime as dt  # noqa: E402
import json           # noqa: E402
import random         # noqa: E402
import shutil         # noqa: E402
import tempfile       # noqa: E402

import pandas as pd   # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq   # noqa: E402

UTC = dt.timezone.utc
T_BEGIN = dt.datetime(2025, 6, 5, 10, 0, tzinfo=UTC)
T_END = dt.datetime(2025, 6, 5, 18, 0, tzinfo=UTC)
CASES = [("aaaa0001-1", dt.datetime(2025, 6, 5, 16, 10, 2, tzinfo=UTC), dt.datetime(2025, 6, 5, 16, 32, 2, tzinfo=UTC)),
         ("bbbb0002-2", dt.datetime(2025, 6, 5, 17, 10, 4, tzinfo=UTC), dt.datetime(2025, 6, 5, 17, 32, 4, tzinfo=UTC))]
SERVICES = ["frontend", "cartservice", "currencyservice", "shippingservice"]
NODES = ["aiops-k8s-01", "aiops-k8s-02", "aiops-k8s-03"]
PODS = {s: [f"{s}-{i}" for i in range(3)] for s in SERVICES}
HOST = {p: NODES[(j + k) % 3] for k, s in enumerate(SERVICES) for j, p in enumerate(PODS[s])}
CST = dt.timedelta(hours=8)


def cpu_factor(pod, t):
    """Case 1: cpu stress in ONE pod (cartservice-1). Case 2: cpu stress in EVERY pod of currencyservice."""
    if CASES[0][1] <= t <= CASES[0][2] and pod == "cartservice-1":
        return 4.0
    if CASES[1][1] <= t <= CASES[1][2] and pod.startswith("currencyservice-"):
        return 4.0
    return 1.0


def minutes():
    t = T_BEGIN
    while t < T_END:
        yield t
        t += dt.timedelta(minutes=1)


def cst_date(t):
    return (t + CST).strftime("%Y-%m-%d")


def write_release(root):
    rnd = random.Random(0)
    days = sorted({cst_date(t) for t in minutes()})
    for d in days:
        for sub in ("metric-parquet/apm/pod", "metric-parquet/apm/service", "metric-parquet/infra/infra_pod",
                    "metric-parquet/infra/infra_node", "log-parquet", "trace-parquet"):
            os.makedirs(os.path.join(root, d, d, sub), exist_ok=True)
        open(os.path.join(root, d, "._" + d), "wb").write(b"\x00\x05\x16\x07Mac OS X")      # macOS artefact
    apm, svc_apm, ipod, inode = [], [], [], []
    for t in minutes():
        ts = t.strftime("%Y-%m-%dT%H:%M:%SZ")
        for s in SERVICES:
            agg = {"request": 0, "rrt": []}
            for p in PODS[s]:
                f = cpu_factor(p, t)
                req = rnd.randint(90, 110)
                rrt = rnd.gauss(10.0, 0.5) * (1.0 + 0.6 * (f > 1))
                apm.append({"time": ts, "client_error": 0, "client_error_ratio": 0.0, "error": 0, "error_ratio": 0.0, "object_id": p,
                            "object_type": "pod", "request": req, "response": req, "rrt": rrt, "rrt_max": int(rrt * 3),
                            "server_error": 0, "server_error_ratio": 0.0, "timeout": 0})
                agg["request"] += req
                agg["rrt"].append(rrt)
                ipod.append({"time": ts, "instance": HOST[p], "kpi_key": "pod_cpu_usage", "namespace": "hipstershop", "object_type": "pod",
                             "pod": p, "pod_cpu_usage": max(0.01, rnd.gauss(0.2, 0.02) * f), "_kpi": "pod_cpu_usage"})
                ipod.append({"time": ts, "instance": HOST[p], "kpi_key": "pod_memory_working_set_bytes", "namespace": "hipstershop",
                             "object_type": "pod", "pod": p, "pod_memory_working_set_bytes": rnd.gauss(2e8, 2e6),
                             "_kpi": "pod_memory_working_set_bytes"})
            r = sum(agg["rrt"]) / 3
            svc_apm.append({"time": ts, "client_error": 0, "client_error_ratio": 0.0, "error": 0, "error_ratio": 0.0, "object_id": s,
                            "object_type": "service", "request": agg["request"], "response": agg["request"], "rrt": r,
                            "rrt_max": int(r * 3), "server_error": 0, "server_error_ratio": 0.0, "timeout": 0})
        for n in NODES:
            inode.append({"time": ts, "device": "null", "instance": "10.0.0.1", "kpi_key": "node_cpu_usage_rate", "kubernetes_node": n,
                          "mountpoint": "null", "object_type": "node", "node_cpu_usage_rate": rnd.gauss(25, 1.5)})
    apm, svc_apm, ipod, inode = map(pd.DataFrame, (apm, svc_apm, ipod, inode))
    for frame in (apm, svc_apm, ipod, inode):
        frame["_d"] = [cst_date(pd.Timestamp(x).to_pydatetime()) for x in frame["time"]]
    for d in days:
        base = os.path.join(root, d, d, "metric-parquet")
        for p, g in apm[apm["_d"] == d].groupby("object_id"):
            name = p + (" (deleted)" if p == "shippingservice-2" else "")              # renamed pods are the same entity
            g.drop(columns="_d").assign(object_id=name).to_parquet(os.path.join(base, "apm/pod", f"pod_{name}_{d}.parquet"), index=False)
        for s, g in svc_apm[svc_apm["_d"] == d].groupby("object_id"):
            g.drop(columns="_d").to_parquet(os.path.join(base, "apm/service", f"service_{s}_{d}.parquet"), index=False)
        for k, g in ipod[ipod["_d"] == d].groupby("_kpi"):
            g.drop(columns=["_d", "_kpi"]).dropna(axis=1, how="all").to_parquet(os.path.join(base, "infra/infra_pod", f"infra_pod_{k}_{d}.parquet"),
                                                                                 index=False)
        inode[inode["_d"] == d].drop(columns="_d").to_parquet(os.path.join(base, "infra/infra_node",
                                                                           f"infra_node_node_cpu_usage_rate_{d}.parquet"), index=False)
    # traces and logs, hourly files named in CST
    tag_t = pa.list_(pa.struct([("key", pa.string()), ("type", pa.string()), ("value", pa.string())]))
    schema = pa.schema([("traceID", pa.string()), ("spanID", pa.string()), ("operationName", pa.string()),
                        ("references", pa.list_(pa.struct([("refType", pa.string()), ("spanID", pa.string()), ("traceID", pa.string())]))),
                        ("startTime", pa.int64()), ("startTimeMillis", pa.int64()), ("duration", pa.int64()), ("tags", tag_t),
                        ("process", pa.struct([("serviceName", pa.string()), ("tags", tag_t)]))])
    tg = lambda **kw: [{"key": k.replace("_", "."), "type": "string", "value": str(v)} for k, v in kw.items()]
    hours = defaultdict_list()
    logs = defaultdict_list()
    nid = iter(range(10 ** 9))
    for t in minutes():
        for j in range(12):
            fe = PODS["frontend"][j % 3]
            callee = ["cartservice", "currencyservice", "shippingservice"][j % 3]
            pod = PODS[callee][(j // 3) % 3]
            tid = f"{next(nid):032x}"
            s0, s1, s2 = (f"{next(nid):016x}" for _ in range(3))
            st = int((t + dt.timedelta(seconds=j * 4)).timestamp() * 1e6)
            dur = int(rnd.gauss(10000, 500) * cpu_factor(pod, t))
            fail = rnd.random() < 0.002
            code = "13" if fail else "0"
            h = (t + CST).strftime("%Y-%m-%d_%H")
            proc_fe = {"serviceName": "frontend", "tags": tg(name=fe, node_name=HOST[fe])}
            proc_cs = {"serviceName": callee, "tags": tg(name=pod, node_name=HOST[pod])}
            hours[h] += [
                {"traceID": tid, "spanID": s0, "operationName": "hipstershop.Frontend/Recv.", "references": [], "startTime": st,
                 "startTimeMillis": st // 1000, "duration": dur + 2000, "tags": tg(span_kind="server", status_code=code, http_status_code=200),
                 "process": proc_fe},
                {"traceID": tid, "spanID": s1, "operationName": f"hipstershop.{callee}/Get",
                 "references": [{"refType": "CHILD_OF", "spanID": s0, "traceID": tid}], "startTime": st + 100, "startTimeMillis": st // 1000,
                 "duration": dur + 500, "tags": tg(span_kind="client", status_code=code, rpc_system="grpc"), "process": proc_fe},
                {"traceID": tid, "spanID": s2, "operationName": f"hipstershop.{callee}/Get",
                 "references": [{"refType": "CHILD_OF", "spanID": s1, "traceID": tid}], "startTime": st + 200, "startTimeMillis": st // 1000,
                 "duration": dur, "tags": tg(span_kind="server", status_code=code, rpc_system="grpc"), "process": proc_cs}]
        for p in [x for v in PODS.values() for x in v]:
            h = (t + CST).strftime("%Y-%m-%d_%H")
            logs[h].append({"k8_namespace": "hipstershop", "@timestamp": t.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "agent_name": "fb",
                            "k8_pod": p, "message": "request complete", "k8_node_name": HOST[p]})
    for h, rows in hours.items():
        d = h[:10]
        pq.write_table(pa.Table.from_pylist(rows, schema=schema),
                       os.path.join(root, d, d, "trace-parquet", f"trace_jaeger-span_{h}-00-00.parquet"))
        pd.DataFrame(logs[h]).to_parquet(os.path.join(root, d, d, "log-parquet", f"log_filebeat-server_{h}-00-00.parquet"), index=False)
    json.dump([{"Anomaly Description": f"A fault occurred from {a:%Y-%m-%dT%H:%M:%SZ} to {b:%Y-%m-%dT%H:%M:%SZ}. Please identify the "
                                       "root cause.", "uuid": u} for u, a, b in CASES], open(os.path.join(root, "input.json"), "w"))
    open(os.path.join(root, "groundtruth.jsonl"), "w").write("SECRET\n")


def defaultdict_list():
    from collections import defaultdict
    return defaultdict(list)


class Stub:
    """Scripted LLM (no model): names one entity in the Evidence->LLM prompt, a type in the typing prompt; records the prompts."""

    def __init__(self, entity):
        self.entity, self.prompts = entity, []
        self.reset_usage()

    def reset_usage(self):
        self.total_calls = 0

    def usage_stats(self):
        return {"total_calls": self.total_calls}

    def generate_json(self, prompt, system=None):
        self.prompts.append(prompt)
        self.total_calls += 1
        if '"root_cause_entity"' in prompt:
            return {"root_cause_entity": self.entity, "predicted_fault_type": "cpu_stress", "reasoning_chain": ["cause: x"]}
        return {}


def llm_arms(rel, specs, wins) -> int:
    """PREREGISTRATION_V6.md AIOps2025 arms on the synthetic release: the LLM prompts carry the AIOps2025 taxonomy (not RCA100's), the
    Evidence->LLM arm maps its answer to a topology entity, and v6 runs end to end with the gate forced (STOP_P = 1.01)."""
    import gama.run as gr
    from adapters.aiops2025 import AIOpsCase
    from gama.belief_llm import run_belief_llm
    from gama.config import Modules
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
    from aiops2025_run import answer
    case = AIOpsCase(rel, specs[1], all_windows=wins)
    st = Stub("currencyservice")
    r = run_belief_llm(case, st, with_belief=False)
    p = st.prompts[0]
    good = ("cpu_stress" in p and "node_network_loss" in p and "F034-cpuFullLoad" not in p and "Statistical diagnosis" not in p
            and r.predicted_entity_ids == ["currencyservice"])
    a = answer(case, r)
    good &= a["component"] == "currencyservice" and a["reason"].startswith("cpu stress at currencyservice") and len(a["reasoning_trace"]) > 0
    stop, gr.STOP_P = gr.STOP_P, 1.01
    try:
        st2 = Stub("frontend")
        r2 = gr.run_case(case, st2, Modules.parse("v6"))
    finally:
        gr.STOP_P = stop
    s2 = r2.retrieval_stats
    good &= bool(s2.get("v6_gate_called")) and s2.get("v6_gate_entity") == "frontend" and r2.predicted_entity_ids == ["frontend"]
    good &= all("F034-cpuFullLoad" not in q for q in st2.prompts)
    a2 = answer(case, r2)
    good &= a2["component"] == "frontend"
    print(f"{'ok  ' if good else 'FAIL'} LLM arms: evidence -> {r.predicted_entity_ids} {r.predicted_fault_type}; v6 gate "
          f"{s2.get('v6_entity_before')} -> {s2.get('v6_gate_entity')}, decided {r2.predicted_entity_ids} {r2.predicted_fault_type}, "
          f"{s2.get('total_calls')} calls")
    return int(good)


def main():
    root = tempfile.mkdtemp(prefix="aiops2025_synth_")
    try:
        write_release(root)
        real_open, real_rp, real_rt = builtins.open, pd.read_parquet, pq.read_table

        def guard(path):
            assert "groundtruth" not in str(path), f"ground truth opened: {path}"

        builtins.open = lambda f, *a, **k: (guard(f), real_open(f, *a, **k))[1]
        pd.read_parquet = lambda f, *a, **k: (guard(f), real_rp(f, *a, **k))[1]
        pq.read_table = lambda f, *a, **k: (guard(f), real_rt(f, *a, **k))[1]
        from data.dataset_profile import PROFILE
        from adapters.aiops2025 import Release, AIOpsCase
        from gama.config import Modules
        from gama.run import run_case
        assert PROFILE == "aiops2025"
        rel = Release(root)
        specs = rel.cases()
        assert [s["uuid"] for s in specs] == [c[0] for c in CASES]
        wins = [(s["start"], s["end"]) for s in specs]
        ok = 0
        for s, want_ent, want_kind in ((specs[0], "cartservice-1", "pod"), (specs[1], "currencyservice", "service")):
            case = AIOpsCase(rel, s, all_windows=wins)
            cov = case.coverage()
            g = case.topology
            assert cov["entities"]["service"] == 4 and cov["entities"]["pod"] == 12 and cov["entities"]["k8s.node"] == 3, cov
            assert g.has_edge("frontend", "cartservice") and g.has_edge("cartservice", "cartservice-1"), list(g.edges())[:10]
            assert "shippingservice-2" in g and not any("deleted" in n for n in g), "renamed pod not merged"
            assert g.has_edge(HOST["cartservice-1"], "cartservice-1")
            assert case.memberships["currencyservice-2"] == "currencyservice"
            n_tr = len(case.observations["traces"])
            assert 0 < n_tr < 0.5 * 3 * 12 * 60 * 5.4, n_tr                     # A7: ~20% of the traces kept
            r = run_case(case, None, Modules.parse("core"))
            ent, typ = r.predicted_entity_ids[0], r.predicted_fault_type
            good = ent == want_ent and typ in ("cpu_stress", "jvm-cpu-stress")
            ok += good
            print(f"{'ok  ' if good else 'FAIL'} {want_kind}-level cpu fault: decided {ent} {typ} (p={r.confidence:.2f}); "
                  f"top {r.retrieval_stats.get('v3_top')}")
        ok += llm_arms(rel, specs, wins)
        print(f"{ok}/3 passed")
        sys.exit(0 if ok == 3 else 1)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
