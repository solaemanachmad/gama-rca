r"""Lenient JSON parsing shared by all LLM clients.

Why: small instruction-tuned models often emit JSON with invalid backslash
escapes (e.g. a quoted log line "React\\EventLoop\\Loop" written as
"React\EventLoop\Loop") or raw control characters inside strings.  The strict
parser rejects the whole object, the agent finding becomes {"_parse_error": ...}
and the coordinator receives a blob of raw text instead of the finding.

Valid JSON is returned exactly as json.loads would return it (so results for
well-formed outputs are unchanged); repairs are only attempted after the strict
parse has failed.  Purely syntactic -- no content-based decisions.
"""
import json
import re
from typing import Optional

_SIMPLE_ESC = set('"\\/bfnrt')
_HEX4 = re.compile(r"[0-9a-fA-F]{4}")


def fix_invalid_escapes(s: str) -> str:
    """Double every backslash that does not start a valid JSON escape."""
    out, i, n = [], 0, len(s)
    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = s[i + 1] if i + 1 < n else ""
        if nxt in _SIMPLE_ESC and nxt != "":
            out.append(s[i:i + 2])
            i += 2
        elif nxt == "u" and _HEX4.fullmatch(s[i + 2:i + 6] or ""):
            out.append(s[i:i + 6])
            i += 6
        else:
            out.append("\\\\")          # literal backslash; next char handled normally
            i += 1
    return "".join(out)


def _try(text: str) -> Optional[dict]:
    for t in (text, fix_invalid_escapes(text), re.sub(r",\s*([}\]])", r"\1", fix_invalid_escapes(text))):
        try:
            v = json.loads(t, strict=False)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(v, dict):
            return v
    return None


def parse_json_lenient(raw: str) -> dict:
    """Return the parsed object, or {"_parse_error": True, "_raw": raw}."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    raw = raw or ""
    start, end = raw.find("{"), raw.rfind("}")
    candidates = [raw] + ([raw[start:end + 1]] if start != -1 and end > start else [])
    for c in candidates:
        v = _try(c)
        if v is not None:
            return v
    return {"_parse_error": True, "_raw": raw}
