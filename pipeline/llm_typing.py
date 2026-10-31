"""v4 LLM typing stage (part 'llmtype' of pipeline/belief_v4.py). The entity has been chosen by the Bayesian localisation; the LLM
chooses its fault type from a SHORTLIST: the K most probable types for that entity under the statistical model. It reads the entity's
evidence summary (calibrated signals with their values and their ONSET relative to the window start, categorical observations, the
hosting nodes' signals, preprocessed log / trace / event items). Its answer is accepted only if the type is in the shortlist and it copies
one evidence line verbatim (grounding); otherwise the statistical type stands.

v2 (2026-10-30, after dev run 4): v1 offered all types of the level and told the LLM to prefer direct mechanism evidence (events,
messages). It then picked podRestartFlapping for CPU faults because of restart events that are consequences (12 of 37 changes; net -4
types). v2 restricts the choice to the statistical shortlist, gives every signal its onset, and states explicitly that events, status
codes and error messages are often consequences, so the mechanism that changes FIRST is the candidate root."""
from __future__ import annotations

from typing import Dict, List

from data.taxonomy import FAULT_DEFINITIONS
from pipeline.fault_calibrated import entity_signals

TYPING_SYSTEM = "You are a site-reliability engineer classifying the fault mechanism of one component. Respond ONLY with valid JSON."
MAX_LINES, MAX_TEXTS, P_SHOW, SHORTLIST = 40, 10, 0.2, 5


def _onset_txt(db, ent, name):
    r = (db.get("_cal") or {}).get(ent, {}).get(name) or {}
    t = r.get("onset")
    if t is None:
        return ""
    return f" [starts at {t - db['w0'].timestamp():+.0f} s from the window start]"


def evidence_lines(db, entity: str, texts: Dict[str, List[str]]) -> List[str]:
    lines = []

    def add(ent, sig):
        for name, v in sorted(sig.items(), key=lambda kv: (kv[1]["p"] is None, kv[1]["p"] or 0)):
            if v["kind"] == "cat":
                lines.append(f"{ent} {name}: {v['ev']}")
            elif v["p"] is not None and v["p"] <= P_SHOW:
                lines.append(f"{ent} {name}: {v['ev']}{_onset_txt(db, ent, name)}")
    add(entity, entity_signals(db, entity))
    for n in db.get("hosts", {}).get(entity, []):
        add(n, {k: v for k, v in entity_signals(db, n).items() if v["kind"] == "cat" or (v["p"] is not None and v["p"] <= 0.05)})
    for t in (texts or {}).get(entity, [])[:MAX_TEXTS]:
        lines.append(f"{entity} message: {t}")
    return lines[:MAX_LINES]


def shortlist(res: Dict, k: int = SHORTLIST) -> List[str]:
    cands = [t for t, _ in res.get("type_rank", [])][:k]
    if res.get("type") and res["type"] not in cands:
        cands = [res["type"]] + cands[:k - 1]
    return cands


def llm_type(llm, db, res: Dict, texts: Dict[str, List[str]]) -> Dict:
    ent, level = res["entity"], res["level"]
    cands = shortlist(res)
    lines = evidence_lines(db, ent, texts)
    out = {"llm_type": None, "accepted": False, "changed": False, "n_lines": len(lines), "prompt_chars": 0, "quote_ok": False,
           "candidates": cands, "evidence_line": None}
    if len(cands) < 2 or not lines:
        return out
    probs = dict(res.get("type_rank", []))
    cand_txt = "\n".join(f"- {t} (statistical probability {probs.get(t, 0.0):.2f}): {FAULT_DEFINITIONS.get(t, '')}" for t in cands)
    prompt = (f"Component: {ent} ({level} level). It has been localised as the root cause of the incident.\n\n"
              "Evidence observed for it in the incident window (each line: entity, signal, values; continuous signals also show when "
              "their change started relative to the window start):\n" + "\n".join(f"- {l}" for l in lines)
              + f"\n\nCandidate fault types (the most probable ones under a statistical model of the evidence):\n{cand_txt}\n\n"
              "Which ONE candidate is the root mechanism? Guidance:\n"
              "- Restarts, failed health probes, OOM kills, HTTP error codes and error messages are often CONSEQUENCES of a resource or "
              "dependency problem; they identify the mechanism only when nothing else changed before them.\n"
              "- The root mechanism is the one whose signal changes FIRST and that explains the other changes.\n"
              "- Keep the most probable candidate unless the evidence clearly favours another one.\n"
              'Respond as JSON: {"fault_type": "<one candidate id exactly as listed>", "evidence_line": "<one evidence line above, '
              'copied verbatim, that supports it>"}')
    out["prompt_chars"] = len(prompt)
    ans = llm.generate_json(prompt, system=TYPING_SYSTEM)
    ans = ans if isinstance(ans, dict) else {}
    t = str(ans.get("fault_type", "")).strip()
    q = str(ans.get("evidence_line", "")).strip().lstrip("- ").lower()
    out["llm_type"] = t
    out["evidence_line"] = q
    grounded = len(q) >= 10 and any(q in l.lower() or l.lower() in q for l in lines)
    out["quote_ok"] = grounded
    if t in cands and grounded:
        out["accepted"] = True
        out["changed"] = t != res["type"]
    return out
