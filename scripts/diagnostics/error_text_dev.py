"""error_text_dev.py -- v5 diagnostic: what TEXT evidence exists at the root for types the core misses (NPE, HTTP 5xx, DB network,
cache, K8s events)? ODD CASES ONLY; no answer key is read (case:service pairs are given on the command line).

Per case and service, in the alert window:
  (1) failing spans: status code / message, HTTP and gRPC status attributes, exception attributes (exception.* / excep.*), top 8;
  (2) log lines of the service that look like errors (exception / error / fail / refused / timeout / sql / redis / null), top 10
      normalised templates, window vs one window length before;
  (3) Kubernetes events whose involved object belongs to the service (kind, reason, message), and the object names that did NOT
      map to any service (to check the event -> service mapping).

Usage (Kaggle): !RCA100_ROOT=... ALERT_T0_MODE=window python scripts/diagnostics/error_text_dev.py t029:inventory t035:inventory \
                 t083:inventory t001:payment t045:ad t043:cart t015:inventory t025:inventory t079:recommendation t039:shipping \
                 t095:shipping t087:shipping > /kaggle/working/error_text_odd.txt 2>&1
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ERR = re.compile(r"exception|error|fail|refused|timeout|timed out|sql|jdbc|redis|null|panic|traceback|denied|unavailable", re.I)
ATTR = re.compile(r'"((?:http\.(?:response\.)?status_code|rpc\.grpc\.status_code|status\.code|exception\.[a-z_]+|excep\.[a-z_]+|'
                  r'error\.type|otel\.status_description))":"?([^",}]{0,160})', re.I)


def norm(s):
    s = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", s)
    s = re.sub(r"\d+", "<n>", s)
    return re.sub(r"\s+", " ", s).strip()[:180]


def main(pairs):
    import config
    from data.loader import Case
    from data.loader import event_time_bounds, _parse_ts_iso
    from pipeline.fault_calibrated import compute_signal_db
    from pipeline.fault_fingerprint import _owner, _naive

    for pair in pairs:
        cid, svc = pair.split(":")
        if int(cid[1:]) % 2 == 0:
            print(f"{cid}: refused (even = validation half)")
            continue
        case = Case(cid, cases_dir=config.CASES_DIR)
        db = compute_signal_db(case, case.alert.alert_timestamp)
        services = list(db["services"])
        w0, w1 = db["w0"], db["w1"]
        h0 = w0 - (w1 - w0)
        print(f"################ {cid}  service={svc}  window {w0} .. {w1}")
        spans, attrs, n_err = Counter(), Counter(), 0
        for o in case.observations.get("traces", []):
            p = o.payload or {}
            ts = _naive(o.timestamp)
            if ts is None or ts < w0 or ts > w1 or _owner(p.get("serviceName"), services) != svc:
                continue
            if str(p.get("statusCode")) != "2":
                continue
            n_err += 1
            a = p.get("attributes") if isinstance(p.get("attributes"), str) else ""
            spans[(str(p.get("spanName"))[:60], norm(str(p.get("statusMessage") or ""))[:120])] += 1
            for k, v in ATTR.findall(a):
                attrs[(k, norm(v)[:100])] += 1
        print(f"  failing spans: {n_err}")
        for (nm, msg), n in spans.most_common(8):
            print(f"    {n:6d}  {nm} | msg: {msg}")
        print("  status / exception attributes on failing spans:")
        for (k, v), n in attrs.most_common(12):
            print(f"    {n:6d}  {k} = {v}")
        logs = Counter()
        before = Counter()
        for o in case.observations.get("logs", []):
            p = o.payload or {}
            ts = _naive(o.timestamp)
            if ts is None or ts < h0 or ts > w1 or _owner(p.get("_container_name_"), services) != svc:
                continue
            c = str(p.get("content") or "")
            if ERR.search(c):
                (logs if ts >= w0 else before)[norm(c)] += 1
        print(f"  error-like log lines: window {sum(logs.values())}, before {sum(before.values())}")
        for t, n in logs.most_common(10):
            print(f"    {n:6d} (before {before.get(t, 0)})  {t}")
        ev, unmapped = Counter(), Counter()
        for o in case.observations.get("events", []):
            p = o.payload or {}
            raw = p.get("eventId")
            try:
                e = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except Exception:
                e = {}
            f, l = event_time_bounds(e)
            first, last = _naive(_parse_ts_iso(f)), _naive(_parse_ts_iso(l))
            if first is None or last is None or last < h0 or first > w1:
                continue
            io = e.get("involvedObject") or {}
            name = io.get("name") or p.get("pod_name") or ""
            own = _owner(name, services)
            key = (io.get("kind", ""), str(e.get("reason", "")), norm(str(e.get("message", "")))[:110])
            if own == svc:
                ev[key] += int(e.get("count") or 1)
            elif not own and io.get("kind") != "Node":
                unmapped[(io.get("kind", ""), name[:50], str(e.get("reason", "")))] += 1
        print("  events of this service:")
        for (k, r, m), n in ev.most_common(10):
            print(f"    {n:6d}  {k} {r}: {m}")
        if unmapped:
            print("  events NOT mapped to any service (kind, name, reason):")
            for (k, nm, r), n in unmapped.most_common(8):
                print(f"    {n:6d}  {k} {nm} {r}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
