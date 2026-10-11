"""Arm D -- per-entity dossier + constrained fault-type choice (USE_DOSSIER_FT=1).

Why: in the smoke runs the Coordinator prompt is 18-23k chars with ~8 tables; the 7B model picks the fault type from
surface words ("error" -> httpError5xx) and ignores numbers it quotes itself (cpu x670 -> "memoryPressure"). Here the
fault type is decided in a SEPARATE short call that sees only the rows about the already chosen entity (~3k chars) and
chooses among lettered options by next-token probability (no free generation, no parse errors). Two option orders are
averaged to cancel position bias.

Rules kept: the LLM decides; code only extracts (rows are selected by entity NAME, not by significance); nothing is
filtered by fault type; no ground truth, no training. The entity itself is unchanged (anchor / coordinator as before).
"""
from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Optional

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["a", "b", "c", "d"]
CHAR_BUDGET = 3600
SYSTEM = ("You are an SRE root-cause analyst. Decide the fault type from the numeric evidence of ONE entity. "
          "Judge by the numbers (levels, change, z), not by words. Reply with only the letter of the best option.")


def _name_re(name: str):
    n = re.escape(name)
    # whole name, or name followed by a pod-hash suffix (ad-5f845bc878-cxqs7); not 'ad' inside 'ad-admission' / 'frontend-proxy'
    return re.compile(rf"(?<![\w-]){n}(?:(?![\w-])|-[a-z0-9]{{8,10}}-[a-z0-9]{{5}}(?![\w-]))")


def build_dossier(names: List[str], texts: Dict[str, Optional[str]], budget: int = CHAR_BUDGET) -> str:
    """Lines of every evidence table that mention one of `names` (entity, and for a node its hosted services).
    Hosting nodes found in the selected lines are added so node / event rows of the entity's host are kept."""
    pats = [_name_re(n) for n in names if n]
    order = ("propagation_text", "infra_text", "colocation_text", "trace_text", "onset_text")
    lines_by_src = {k: [ln for ln in (texts.get(k) or "").split("\n") if ln.strip()] for k in order}
    sel: Dict[str, List[str]] = {k: [] for k in order}
    for k in order:
        for ln in lines_by_src[k]:
            if ln.startswith("Onset table"):
                continue
            if any(p.search(ln) for p in pats):
                sel[k].append(ln)
    nodes = set()
    for ln in sum(sel.values(), []):
        m = re.search(r"node (\S+) \(hosts:", ln)
        if m:
            nodes.add(m.group(1))
    if nodes:
        node_pats = [_name_re(n) for n in nodes]
        for k in order:
            for ln in lines_by_src[k]:
                if ln not in sel[k] and not ln.startswith("Onset table") and any(p.search(ln) for p in node_pats):
                    sel[k].append(ln)
    out, used = [], 0
    for k in order:
        for ln in sel[k]:
            ln = ln[:700]
            if used + len(ln) + 1 > budget:
                continue
            out.append(ln)
            used += len(ln) + 1
    return "\n".join(out)


def _options_block(slugs: List[str], defs: Dict[str, str]) -> str:
    return "\n".join(f"{LABELS[i]}. {s}: {defs.get(s, '')}".rstrip(": ") for i, s in enumerate(slugs))


def decide_fault_type(llm, alert_text: str, entity_name: str, dossier: str, slugs: List[str],
                      defs: Dict[str, str], knowledge: str, change_legend: str = "") -> Dict:
    """Returns {"slug", "p1", "top3", "mode", "chars"}; slug None if the call failed."""
    res = {"slug": None, "p1": None, "top3": "", "mode": "none", "chars": 0}
    if not dossier.strip() or len(slugs) > len(LABELS):
        return res
    logps = {s: [] for s in slugs}
    for order in (list(slugs), list(reversed(slugs))):
        prompt = (f"Alert: {alert_text}\nSuspected origin entity: {entity_name}\n\n"
                  f"Evidence rows about this entity and its host (everything shown; judge which signals are abnormal)."
                  f"{change_legend}\n{dossier}\n\n{knowledge}\n\n"
                  f"Which fault type best explains the abnormal signals of {entity_name}? Options:\n"
                  f"{_options_block(order, defs)}\n\nAnswer with the single letter of the best option.")
        res["chars"] = max(res["chars"], len(prompt))
        labels = LABELS[:len(order)]
        sc = llm.score_options(prompt, labels, system=SYSTEM) if hasattr(llm, "score_options") else None
        if sc is None:                                   # backend without logprobs: one sampled letter
            raw = llm.generate(prompt, system=SYSTEM)
            m = re.match(r"\s*([A-Za-z])", raw or "")
            if not m or m.group(1) not in labels:
                continue
            res["mode"] = "letter"
            logps[order[labels.index(m.group(1))]].append(0.0)
            continue
        res["mode"] = "logprob"
        for lab, lp in sc.items():
            logps[order[labels.index(lab)]].append(lp)
    scored = {s: sum(v) / len(v) for s, v in logps.items() if v}
    if not scored:
        return res
    if res["mode"] == "letter":
        best = max(scored, key=scored.get)
        res.update(slug=best, p1=None, top3=best)
        return res
    z = sum(math.exp(v) for v in scored.values())
    probs = sorted(((s, math.exp(v) / z) for s, v in scored.items()), key=lambda x: -x[1])
    res.update(slug=probs[0][0], p1=round(probs[0][1], 3),
               top3="|".join(f"{s}:{p:.2f}" for s, p in probs[:3]))
    return res
