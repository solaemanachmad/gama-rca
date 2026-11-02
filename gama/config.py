"""Frozen configuration of GAMA-RCA (PREREGISTRATION_V5.md "Final configuration", git tag v5-frozen).

The method is a base model (M0) plus seven modules. Each module is ONE switch: it is the unit of the ablation table in the paper,
so the code has no other switches. Constants (thresholds, priors) are fixed a priori and live next to the code that uses them
(pipeline/belief_diagnosis.py: THETA_K, THETA_U, THETA_P, ALPHA, ALPHA_C, priors, sensor budget; gama/belief.py: THETA_M, THETA_T,
EM bounds); none is tuned per case.

Module spec (env GAMA_MODULES, or Modules.parse): comma-separated tokens, applied left to right
    core   = mm, trace, typing, load, text, chain        (GAMA_CORE, CPU)
    full   = core + llm                                   (GAMA_LLM, GPU)
    base   = none (M0 only)
    <name> adds a module, -<name> removes it              e.g. "core,-trace" = leave-one-out without the trace module
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields

MODULES = ("mm", "trace", "typing", "load", "text", "chain", "llm")
DESCRIPTION = {
    "mm":     "M1 cross-modal corroboration: mechanism evidence in >= 2 modalities (P1)",
    "trace":  "M2 trace structure: error / slow origin, origin messages, endpoint relay of unreachable callees (P2)",
    "typing": "M3 type stage: host CPU explained by the root, error-class gating, general-before-special tie-break, "
              "intra-entity signal order (P2)",
    "load":   "M4 directional load propagation: load from upstream (and its latency / host CPU) is explained away (P2)",
    "text":   "M5 mechanism-naming text: span exceptions, gRPC status, log text new in the window (P1)",
    "chain":  "M6 evidence-cited chain: every step cites metric=value of its own target (reasoning process only)",
    "llm":    "M7 bounded LLM: EIG log sensor, shortlist typing with verbatim quote + fingerprint check, narrator scored "
              "separately (P3)",
}
CORE = MODULES[:-1]

# Names of the same switches in the development code (pipeline/belief_v4.py V4_PARTS) -- logs and the equivalence test only.
BASE_PARTS = ("score", "open", "twostage", "fp", "peer", "prop", "events", "avail", "onset", "em", "selfnorm", "llmtype", "conseq",
              "retry")
LEGACY_PARTS = {"mm": ("mm",), "trace": ("torigin3", "omsg", "msgtarget"), "typing": ("hostcpu", "errclass", "tiebreak", "sigorder2"),
                "load": ("loadx2",), "text": ("textx2",), "chain": ("redchain",), "llm": ("xcard", "keepchain", "typecheck")}


@dataclass(frozen=True)
class Modules:
    mm: bool = True
    trace: bool = True
    typing: bool = True
    load: bool = True
    text: bool = True
    chain: bool = True
    llm: bool = False

    @classmethod
    def parse(cls, spec: str = "core") -> "Modules":
        on = set()
        for tok in (t.strip() for t in (spec or "core").split(",")):
            if not tok:
                continue
            if tok in ("core", "full", "all", "base", "none"):
                on |= set(CORE) if tok == "core" else set(MODULES) if tok in ("full", "all") else set()
            elif tok.startswith("-") and tok[1:] in MODULES:
                on.discard(tok[1:])
            elif tok in MODULES:
                on.add(tok)
            else:
                raise ValueError(f"unknown module '{tok}' (known: {', '.join(MODULES)}; core, full, base)")
        return cls(**{m: m in on for m in MODULES})

    @classmethod
    def from_env(cls) -> "Modules":
        return cls.parse(os.environ.get("GAMA_MODULES", "core"))

    def on(self):
        return tuple(f.name for f in fields(self) if getattr(self, f.name))

    def spec(self) -> str:
        return ",".join(self.on()) or "base"

    def legacy_parts(self):
        return BASE_PARTS + tuple(p for m in self.on() for p in LEGACY_PARTS[m])
