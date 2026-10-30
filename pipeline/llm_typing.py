"""v4.3 LLM typing stage (part 'llmtype' of pipeline/belief_v4.py). The entity has been chosen by the Bayesian localisation; the LLM
chooses its fault type from ALL types of the entity's level, reading (1) the entity's evidence summary (calibrated signals with their
values, categorical observations, the hosting nodes' signals, preprocessed log / trace / event items) and (2) the statistical ranking of
types for that entity. Its answer is accepted only if the type is one of the candidates and it copies one evidence line verbatim
(grounding); otherwise the Bayesian type stands. Motivation (DEVLOG_V4.md, dev run 3): in 34 of 103 cases the entity was right and the
type wrong, and in 19 of them the right type was not even among the top five statistical candidates."""
from __future__ import annotations

from typing import Dict, List

from data.taxonomy import FAULT_DEFINITIONS
from pipeline.fault_calibrated import entity_signals

TYPING_SYSTEM = "You are a site-reliability engineer classifying the fault mechanism of one component. Respond ONLY with valid JSON."
MAX_LINES, MAX_TEXTS, P_SHOW = 40, 10, 0.2


def evidence_lines(db, entity: str, texts: Dict[str, List[str]]) -> List[str]:
    lines = []

    def add(ent, sig):
        for name, v in sorted(sig.items(), key=lambda kv: (kv[1]["p"] is None, kv[1]["p"] or 0)):
            if v["kind"] == "cat":
                lines.append(f"{ent} {name}: {v['ev']}")
            elif v["p"] is not None and v["p"] <= P_SHOW:
                lines.append(f"{ent} {name}: {v['ev']}")
    add(entity, entity_signals(db, entity))
    for n in db.get("hosts", {}).get(entity, []):
        add(n, {k: v for k, v in entity_signals(db, n).items() if v["kind"] == "cat" or (v["p"] is not None and v["p"] <= 0.05)})
    for t in (texts or {}).get(entity, [])[:MAX_TEXTS]:
        lines.append(f"{entity} message: {t}")
    return lines[:MAX_LINES]


def llm_type(llm, db, res: Dict, texts: Dict[str, List[str]]) -> Dict:
    ent, level = res["entity"], res["level"]
    cands = [t for t in res.get("candidate_types", []) if t]
    lines = evidence_lines(db, ent, texts)
    out = {"llm_type": None, "accepted": False, "changed": False, "n_lines": len(lines), "prompt_chars": 0}
    if not cands or not lines:
        return out
    rank = ", ".join(f"{t} ({p:.2f})" for t, p in res.get("type_rank", [])[:5]) or "(none)"
    cand_txt = "\n".join(f"- {t}: {FAULT_DEFINITIONS.get(t, '')}" for t in cands)
    prompt = (f"Component: {ent} ({level} level). It has been localised as the root cause of the incident.\n\n"
              "Evidence observed for it in the incident window (each line: entity, signal, values; calibrated p-values compare the window "
              "with the component's own history):\n" + "\n".join(f"- {l}" for l in lines)
              + f"\n\nStatistical ranking of fault types for this component (posterior probability; it can be wrong when two mechanisms "
              f"produce similar metrics): {rank}\n\nCandidate fault types:\n{cand_txt}\n\n"
              "Which ONE candidate fault type best explains the evidence? Prefer direct evidence of a mechanism (error messages, status "
              "codes, events, the specific resource that saturates) over generic symptoms (latency, request count).\n"
              'Respond as JSON: {"fault_type": "<one candidate id exactly as listed>", "evidence_line": "<one evidence line above, '
              'copied verbatim, that supports it>"}')
    out["prompt_chars"] = len(prompt)
    ans = llm.generate_json(prompt, system=TYPING_SYSTEM)
    ans = ans if isinstance(ans, dict) else {}
    t = str(ans.get("fault_type", "")).strip()
    q = str(ans.get("evidence_line", "")).strip().lstrip("- ").lower()
    out["llm_type"] = t
    grounded = len(q) >= 10 and any(q in l.lower() or l.lower() in q for l in lines)
    if t in cands and grounded:
        out["accepted"] = True
        out["changed"] = t != res["type"]
    out["quote_ok"] = grounded
    return out
