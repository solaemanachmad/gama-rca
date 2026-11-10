"""GAMA-RCA: Graph-aware, Multimodal, Abductive root-cause analysis for microservices.
Label-free: an abductive belief over (entity x fault type) hypotheses is COMPUTED from
calibrated multimodal evidence along the service / host topology; a bounded local LLM
reads text and selects among ranked options (optional).

Principles:
  P1 calibrate, do not threshold: every observation is scored against the case's own history and peers.
  P2 causes precede and propagate directionally: errors flow up the call graph, load flows down.
  P3 the LLM reads text and chooses among ranked options; it never overrides calibrated evidence.

Architecture:
  - gama.core: Reasoning modules (evidence, belief, chain, decide, llm)
  - gama.config: Hyperparameters, module switches, and ablation arm definitions
  - gama.run: Case execution and multi-case evaluation runner
"""

import sys
from gama.core import belief, chain, decide, evidence, llm
from gama.run import gama, run_case

# Backward compatibility module aliases
sys.modules.setdefault("gama.evidence", evidence)
sys.modules.setdefault("gama.belief", belief)
sys.modules.setdefault("gama.chain", chain)
sys.modules.setdefault("gama.decide", decide)
sys.modules.setdefault("gama.llm", llm)
try:
    import baselines.belief_llm as _b_llm
    sys.modules.setdefault("gama.belief_llm", _b_llm)
except ImportError:
    pass

__all__ = [
    "belief",
    "chain",
    "decide",
    "evidence",
    "llm",
    "gama",
    "run_case",
]
