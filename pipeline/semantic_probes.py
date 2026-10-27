"""Semantic probes for v3: per-entity raw text from the alert window, and the LLM text sensor that maps it onto ONE category.

The sensor is the only place where the LLM reads telemetry. Its answer is accepted only if it quotes an exact excerpt of the text
(faithfulness check); otherwise the outcome is 'none'."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Dict, List

from pipeline.fault_fingerprint import _owner, _naive, _TEXT_ANY
from pipeline.belief_diagnosis import CATS

MAX_LINES, MAX_CHARS = 20, 200
_MASK = re.compile(r"\d{4}-\d\d-\d\dT[\d:.+Z-]+|\b[0-9a-f]{8,}\b|\b\d+(\.\d+)?\b", re.I)
CATEGORY_HELP = {
    "npe": "null pointer / null or nil dereference", "ratelimit": "throttling, rate limit, quota exceeded, HTTP 429",
    "dns": "name resolution failure (no such host, unknown host)", "refused": "connection refused or reset, service unavailable",
    "timeout": "timeouts, deadline exceeded", "redis": "Redis / Valkey cache errors", "oom": "out of memory, OOMKilled, heap space",
    "gc": "garbage-collection pauses or overhead", "thread": "thread pool exhausted or task rejected",
    "queue": "message queue / Kafka backlog or consumer errors", "lb": "load balancer: no healthy upstream, no endpoints",
    "netpol": "network policy, no route to host, i/o timeout between services", "sql": "database / SQL errors or slow queries",
    "cache": "cache misses or cache layer problems", "none": "no failure mechanism visible in the text"}
SENSOR_SYSTEM = "You classify failure messages from a microservice system. Respond ONLY with valid JSON."


def collect_texts(case, db) -> Dict[str, List[str]]:
    """Distinct (id/number-masked) failure texts per service and node in the alert window: error-span messages, log lines that match
    a failure pattern, K8s event messages. One pass over each modality."""
    from data.loader import event_time_bounds, _parse_ts_iso
    w0, w1 = db["w0"], db["w1"]
    services, nodes = list(db["services"]), set(db["nodes"])
    seen = defaultdict(dict)

    def add(ent, txt):
        t = re.sub(r"\s+", " ", str(txt)).strip()[:MAX_CHARS]
        k = _MASK.sub("#", t)
        if t and k not in seen[ent] and len(seen[ent]) < MAX_LINES:
            seen[ent][k] = t
    for o in case.observations.get("traces", []):
        p = o.payload or {}
        ts = _naive(o.timestamp)
        if ts is None or ts < w0 or ts > w1 or str(p.get("statusCode")) != "2" or not p.get("statusMessage"):
            continue
        s = _owner(p.get("serviceName"), services)
        if s:
            add(s, p["statusMessage"])
    for o in case.observations.get("logs", []):
        ts = _naive(o.timestamp)
        content = (o.payload or {}).get("content") or ""
        if ts is None or ts < w0 or ts > w1 or not _TEXT_ANY.search(content):
            continue
        s = _owner((o.payload or {}).get("_container_name_"), services)
        if s:
            add(s, content)
    for o in case.observations.get("events", []):
        p = o.payload or {}
        try:
            e = json.loads(p.get("eventId")) if isinstance(p.get("eventId"), str) else (p.get("eventId") or {})
        except Exception:
            e = {}
        _f, _l = event_time_bounds(e)
        first, last = _naive(_parse_ts_iso(_f)), _naive(_parse_ts_iso(_l))
        if first is None or last is None or last < w0 or first > w1:
            continue
        io = e.get("involvedObject") or {}
        name = io.get("name") or ""
        ent = name if (io.get("kind") == "Node" or name in nodes) else _owner(name, services)
        if ent:
            add(ent, f"{e.get('reason', '')}: {e.get('message', '')}")
    return {k: list(v.values()) for k, v in seen.items()}


def make_sensor(llm, log: list):
    """Returns sensor(entity, texts) -> category, recording every call (prompt size, raw answer, accepted category) in `log`."""
    cats = "\n".join(f"- {c}: {CATEGORY_HELP[c]}" for c in CATS + ["none"])

    def sensor(entity, texts):
        body = "\n".join(f"- {t}" for t in texts)
        prompt = (f"Messages observed for '{entity}' during an incident window (ids and numbers may be masked):\n{body}\n\n"
                  f"Which ONE category best describes the dominant failure mechanism these messages show?\n{cats}\n\n"
                  'Respond as JSON: {"category": "<one category name>", "quote": "<an exact excerpt of one message above that shows it>"}')
        res = llm.generate_json(prompt, system=SENSOR_SYSTEM)
        res = res if isinstance(res, dict) else {}
        cat = str(res.get("category", "")).strip().lower()
        quote = str(res.get("quote", "")).strip().lower()
        valid = cat in CATS and bool(quote) and any(quote in t.lower() for t in texts)
        accepted = cat if valid else "none"
        log.append({"entity": entity, "answer": cat, "quote_ok": bool(quote) and any(quote in t.lower() for t in texts),
                    "accepted": accepted, "n_lines": len(texts), "prompt_chars": len(prompt)})
        return accepted
    return sensor
