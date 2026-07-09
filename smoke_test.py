"""
smoke_test.py
===============
Run this FIRST, locally, before touching run_experiment.py.

It exercises each module one stage at a time and stops with a clear error
at whichever stage breaks (usually: wrong dataset path, or a field-name
mismatch in data_loader.py's CANDIDATE_*_KEYS). Fix that one stage, re-run
this script, then move to the next stage — don't jump straight to the full
103-case x 5-system experiment.

Usage:
    export RCA100_ROOT=/path/to/RCA100         # folder containing cases/, answer-key/, manifest.txt
    python smoke_test.py                        # stages 1-3, no LLM/Ollama needed
    python smoke_test.py --with-llm             # also runs stage 4 (needs `ollama serve` running)
"""

import argparse
import sys

import config
from data_loader import inspect_case, Case, list_case_ids
from graph_retrieval import GraphRetriever
from vector_retrieval import build_case_index
from hybrid_retrieval import HybridRetriever
from evidence_summarizer import summarize_evidence, render_summary_text
from pipeline import parse_alert


def stage_1_inspect(case_id: str):
    print(f"\n[Stage 1] inspect_case({case_id!r}) — confirm real field names")
    inspect_case(case_id)


def stage_2_load(case_id: str) -> Case:
    print(f"\n[Stage 2] loading Case({case_id!r})")
    case = Case(case_id)
    print(case)
    print(case.alert)
    if case.unresolved_entities:
        print(f"  warning: {len(case.unresolved_entities)} entity IDs not found in topology "
              f"(expected to be small/rare per the paper's 98.06% match rate)")
    return case


def stage_3_retrieval(case: Case):
    print("\n[Stage 3] graph + vector + hybrid retrieval (no LLM)")
    parsed = parse_alert(case)
    print("  parsed alert:", parsed)

    graph_retriever = GraphRetriever(case.topology)
    graph_result = graph_retriever.retrieve(parsed["entry_entity_id"])
    print(f"  candidate subgraph size: {graph_result['subgraph'].number_of_nodes()} nodes")
    print(f"  top-5 ranked entities: {graph_result['ranked_entities'][:5]}")

    vector_index = build_case_index(case)
    print(f"  vector index size: {vector_index.index.ntotal} embedded observations")

    hybrid = HybridRetriever(graph_result, vector_index)
    evidence_items = hybrid.retrieve_multi([parsed["alert_text"]] + parsed["keywords"][:5])
    print(f"  hybrid evidence items retrieved: {len(evidence_items)}")
    if evidence_items:
        print(f"  top hit: {evidence_items[0].observation.text!r} "
              f"(hybrid_score={evidence_items[0].hybrid_score:.3f})")

    summary = summarize_evidence(evidence_items)
    print("\n  --- Evidence summary (this is what the LLM will see) ---")
    print(render_summary_text(summary))
    return evidence_items


def stage_4_full_pipeline(case_id: str):
    print(f"\n[Stage 4] full LLM pipeline on {case_id!r} (requires `ollama serve` running)")
    from llm_client import LLMClient
    from pipeline import GraphRAGPipeline

    llm = LLMClient()
    pipeline = GraphRAGPipeline(llm=llm)
    result = pipeline.run(case_id)
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", default=None, help="defaults to the first case in manifest.txt")
    parser.add_argument("--with-llm", action="store_true", help="also run stage 4 (needs Ollama)")
    args = parser.parse_args()

    try:
        case_id = args.case_id or list_case_ids()[0]
    except FileNotFoundError:
        print(f"ERROR: could not find manifest.txt under {config.DATASET_ROOT}\n"
              f"Set RCA100_ROOT to the folder containing cases/, answer-key/, manifest.txt.")
        sys.exit(1)

    stage_1_inspect(case_id)
    case = stage_2_load(case_id)
    stage_3_retrieval(case)

    if args.with_llm:
        stage_4_full_pipeline(case_id)
    else:
        print("\n[done] Stages 1-3 passed without touching the LLM. "
              "Run with --with-llm once `ollama serve` + `ollama pull <model>` are ready.")
