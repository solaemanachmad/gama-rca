"""Number grounding of an LLM-written chain: the share of quoted values (signal=value) that occur in the evidence text. Used by the
GAMA narrator (gama/run.py; statistics v3_grounded / v3_numbers). Moved unchanged from agents/multi_agent.py (archived)."""
import re

_QUOTED_NUM = re.compile(r"=\s*[-+]?(\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.I)
_ANY_NUM = re.compile(r"(?<![\w.])[-+]?(\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.I)


def _evidence_pool(state):
    texts = [state.get("propagation_text") or "", state.get("infra_text") or "", state.get("trace_text") or ""]
    for bl in (state.get("evidence_summary") or {}).values():
        texts.extend(str(b) for b in bl)
    pool = set()
    for t in texts:
        for m in _ANY_NUM.finditer(t):
            try:
                pool.add(float(m.group(1)))
            except ValueError:
                pass
    return sorted(pool), "\n".join(texts)


def _is_grounded(x, pool):
    import bisect
    for v in (x, x / 1000.0, x * 1000.0, x / 100.0, x * 100.0):
        tol = max(1e-9, 0.01 * abs(v))
        i = bisect.bisect_left(pool, v - tol)
        if i < len(pool) and pool[i] <= v + tol:
            return True
    return False


def _grounding(chain, pool):
    nums, bad = 0, []
    for step in chain or []:
        for m in _QUOTED_NUM.finditer(str(step)):
            try:
                x = float(m.group(1))
            except ValueError:
                continue
            nums += 1
            if not _is_grounded(x, pool):
                bad.append(m.group(0).lstrip("= "))
    return (1.0 if nums == 0 else (nums - len(bad)) / nums), nums, bad
