"""Stage 4 -- the bounded LLM (M7). The LLM reads text and chooses among options the statistics already ranked; it never
overrides calibrated evidence (P3):
  sensor  reads the log / span / event text of ONE entity chosen by expected information gain (gama/decide.py) and names its failure
          category; accepted only with a verbatim quote of the text; the answer enters the belief as a likelihood term
  typing  picks the decided root's fault type from the statistical shortlist (top 5), with signature cards (expected cross-modal
          evidence per type); accepted only if the type is in the shortlist, a quoted evidence line (>= 10 chars) is verbatim, and
          that line names a signal of the chosen type's fingerprint (pipeline/llm_typing.py)
  narrator writes the causal chain of the decided diagnosis; the scored chain stays the computed one (gama/chain.py)
Settings fixed a priori (config/__init__.py): temperature 0, seed 0, max new tokens 1024, thinking off."""
from __future__ import annotations

from data.taxonomy import FAULT_DEFINITIONS

NARRATOR_SYSTEM = ("You are a site-reliability engineer. The diagnosis has already been decided from the evidence; you write the causal "
                   "chain that the evidence supports. Respond ONLY with valid JSON.")


def texts_and_sensor(case, db, llm, sensor_log):
    """Preprocessed text per entity (log templates tested against the pre-window rate, failing-span messages, events) and the sensor."""
    from pipeline.log_preprocess import preprocess_texts
    from pipeline.semantic_probes import make_sensor
    texts, notes, pre_stats = preprocess_texts(case, db)
    return texts, make_sensor(llm, sensor_log, notes), pre_stats


def type_root(llm, db, res, texts, case, model):
    """Shortlist typing of the decided root with signature cards and the fingerprint check of the quoted line."""
    from pipeline.llm_typing import llm_type
    if not texts:
        from pipeline.log_preprocess import preprocess_texts
        texts, _n, _s = preprocess_texts(case, db)
    return llm_type(llm, db, res, texts, model=model, xcard=True, typecheck=True)


def narrator_prompt(case, alert, op_name, path, res, ev_lines, tables) -> str:
    """Narrator (M7): writes the chain of a DECIDED diagnosis along the topology path, citing window values. With M7 the scored
    chain stays the computed one; the narrative is kept (llm_chain) and scored separately."""
    return (f"Alert: {case.alert.alert_text}\nAlerted service: {alert or '(none given)'}\nAlerted operation: {op_name or '(none given)'}\n"
            f"Call path from the root cause to the alerted service (topology): {' -> '.join(path) if path else '(none)'}\n\n"
            f"Diagnosis (decided): root cause entity = {res['entity']} ({res['level']}), fault type = {res['type']}: "
            f"{FAULT_DEFINITIONS.get(res['type'], '')}\n\nEvidence that decided it:\n" + "\n".join(ev_lines)
            + ("\n\nText probes: " + "; ".join(f"{p['probe']} -> {p['category']}" for p in res["probes"]) if res["probes"] else "")
            + f"\n\nEvidence tables:\n{tables}\n\n"
            "Task: write the causal chain from the root cause to the alerted operation, one step per entity of the call path: "
            "'cause: <root cause entity> ...' first, then 'propagation: <service> ...' for each service between the root and the "
            "alerted service, and last 'impact: <alerted operation name, exactly as given> ...'. 'target' names are copied from "
            "the text above; write signals as metric_name=value using the window value the evidence shows (e.g. "
            "node_cpu_usage_rate=0.95, error_count=412, latency=2.8), copied exactly; do not write test statistics, p-values or "
            "ratios.\n"
            'Respond as JSON: {"reasoning_chain": ["cause: ...", "propagation: ...", "impact: ..."]}')
