"""Arm C -- candidate check of the fault type (VERIFY_FAULT_TYPE=1).

Why (claude/audit smoke v4 + P1, 2026-10-12): in the 22k-char Coordinator prompt the 7B model picks the fault type from surface words
(error_rate -> httpError5xx) although a short, direct question ("does this evidence satisfy this definition?") is answered correctly by 7B and 14B
(probe: 403 is not 5xx). This module asks exactly that short question once, about the type the Coordinator already chose.

Rules kept: the LLM decides (the check and the re-choice are LLM calls); code only extracts evidence rows by entity NAME (never by significance)
and verifies that a quoted row really occurs in the rows (diagnostic); nothing is filtered before the first Coordinator answer; no ground truth.
The hypothesis is shown as its public DEFINITION only, without the slug, so that the name 'httpError5xx' cannot anchor the answer.
The entity and the chain of the first answer are kept; only predicted_fault_type may change after a re-choice.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List

from agents.dossier import _name_re

BUDGET = 5000
CHECK_SYSTEM = (
    "You check one fault hypothesis against evidence rows of ONE entity. The hypothesis is satisfied only if the rows contain the observable "
    "its definition requires. A different status-code class, a different resource, a rate without the required breakdown, or a missing signal "
    "does not satisfy it. Copy the deciding row text verbatim. Respond ONLY with valid JSON."
)
CHECK_SCHEMA = (
    '{"required_observable": "<what the definition requires>", "quote": "<verbatim row text that shows it, or none>", '
    '"satisfied": "<yes|no>", "rows_show_instead": "<what the rows do show about the abnormal signals, one sentence>"}'
)


def entity_rows(names: List[str], blocks: List[str], budget: int = BUDGET) -> str:
    """Lines of every evidence block that mention one of `names` (whole name or name + pod-hash suffix); block order kept."""
    pats = [_name_re(n) for n in names if n]
    out, used = [], 0
    for blk in blocks:
        for ln in (blk or "").split("\n"):
            if not ln.strip() or ln.startswith("Onset table"):
                continue
            if any(p.search(ln) for p in pats):
                ln = ln[:700]
                if used + len(ln) + 1 > budget:
                    continue
                out.append(ln)
                used += len(ln) + 1
    return "\n".join(out)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def check_type(llm, alert_text: str, entity: str, rows: str, definition: str) -> Dict:
    """Returns {"satisfied": bool|None, "quote", "grounded": bool, "required", "instead", "chars", "parse_error"}."""
    prompt = (f"Alert: {alert_text}\nEntity: {entity}\n\nEvidence rows that mention {entity} (every table, everything shown):\n{rows}\n\n"
              f"Hypothesis: {definition}\n\nDoes the evidence satisfy the hypothesis? Respond as JSON: {CHECK_SCHEMA}")
    res = {"satisfied": None, "quote": None, "grounded": None, "required": None, "instead": None, "chars": len(prompt), "parse_error": False,
           "prompt": prompt}
    out = llm.generate_json(prompt, system=CHECK_SYSTEM)
    if not isinstance(out, dict) or out.get("_parse_error"):
        res["parse_error"] = True
        return res
    ans = _norm(out.get("satisfied"))
    res["satisfied"] = True if ans.startswith("yes") else (False if ans.startswith("no") else None)
    q = str(out.get("quote") or "")
    res["quote"], res["required"], res["instead"] = q, out.get("required_observable"), out.get("rows_show_instead")
    res["grounded"] = bool(q) and _norm(q) != "none" and _norm(q) in _norm(rows)
    res["raw"] = out
    return res


def rechoice_prompt(first_prompt: str, answer: dict, entity: str, chk: Dict) -> str:
    return (first_prompt + "\n\nYour first answer was:\n" + json.dumps(answer, ensure_ascii=False)
            + f"\n\nA direct check of your fault type against the rows of {entity} found it NOT satisfied. Required observable: "
              f"{chk.get('required')}. The rows show instead: {chk.get('instead')}.\n"
              "Keep the entity and the chain. Choose predicted_fault_type again from the same taxonomy: pick the type whose definition is "
              "satisfied by what those rows show. Answer with the same JSON schema.")
