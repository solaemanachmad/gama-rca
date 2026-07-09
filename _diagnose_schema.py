"""
diagnose_schema.py
====================
Run this and paste me the FULL output.

Usage:
    export RCA100_ROOT=/path/to/RCA100
    python diagnose_schema.py --case_id t001
"""

import argparse
import json
import os
import pandas as pd

import config


def dump_topology_keys(path):
    print(f"\n=== topology.json structure ({path}) ===")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print("top-level keys:", list(raw.keys()))
    print("number of entities:", len(raw.get("entities", [])))
    for key in raw.keys():
        if key == "entities":
            continue
        val = raw[key]
        if isinstance(val, list):
            print(f"\n  key '{key}' is a list of {len(val)} items. First item:")
            print(json.dumps(val[0], indent=2, default=str)[:1000] if val else "  (empty)")
        else:
            print(f"\n  key '{key}':", str(val)[:300])

    # also show entity TYPES present (service? pod? node? operation?)
    types_seen = {}
    for e in raw.get("entities", []):
        t = e.get("type", "unknown")
        types_seen[t] = types_seen.get(t, 0) + 1
    print("\n  entity types present:", types_seen)


def dump_parquet(path, label):
    print(f"\n=== {label} ({path}) ===")
    df = pd.read_parquet(path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    if len(df) > 0:
        print("\nfirst row as dict:")
        print(json.dumps(df.iloc[0].to_dict(), indent=2, default=str)[:1500])


def dump_answer_key_dir():
    print(f"\n=== answer_key directory listing ({config.ANSWER_KEY_DIR}) ===")
    if not os.path.isdir(config.ANSWER_KEY_DIR):
        print(f"  DOES NOT EXIST at this path. Check the real folder name under {config.DATASET_ROOT}")
        return
    entries = sorted(os.listdir(config.ANSWER_KEY_DIR))
    print(f"  {len(entries)} entries. First 20:")
    for e in entries[:20]:
        print(" ", e)

    # if there's an obvious mapping-like file, show its shape
    for name in entries:
        if "map" in name.lower() or "index" in name.lower():
            path = os.path.join(config.ANSWER_KEY_DIR, name)
            print(f"\n  possible mapping file: {name}")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                print("  type:", type(content).__name__)
                if isinstance(content, dict):
                    print("  sample keys:", list(content.items())[:5])
                elif isinstance(content, list):
                    print("  sample items:", content[:3])
            except Exception as ex:
                print("  (could not parse as JSON:", ex, ")")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", default="t001")
    args = parser.parse_args()

    case_dir = os.path.join(config.CASES_DIR, args.case_id)
    print(f"Inspecting case: {case_dir}")

    dump_topology_keys(os.path.join(case_dir, config.FILE_TOPOLOGY))

    for fname, label in [
        (config.FILE_METRICS, "metrics.parquet"),
        (config.FILE_LOGS, "logs.parquet"),
        (config.FILE_TRACES, "traces.parquet"),
        (config.FILE_EVENTS, "events.parquet"),
        (config.FILE_ALERTS, "alerts.parquet"),
    ]:
        dump_parquet(os.path.join(case_dir, fname), label)

    dump_answer_key_dir()