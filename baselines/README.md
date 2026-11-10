# `baselines/` — Comparative Baseline Models

This directory contains the comparative baseline implementations evaluated alongside **GAMA-RCA** in the paper (Table 4, Section 5.2).

---

## 🔬 Systems Included

1. **`direct_llm.py` (Direct LLM)**:
   - Evaluates pure zero-shot LLM reasoning without any telemetry or evidence.
   - The LLM receives only the alert text and the 28 RCA100 fault definitions.
   - Serves as the lower baseline measuring the value of multimodal observability.

2. **`evidence_llm.py` (Evidence $\rightarrow$ LLM)**:
   - Evaluates unconstrained LLM reasoning when provided with the **exact same** statistically calibrated evidence tables as GAMA-RCA (call graphs, metric shifts, log bursts, trace errors).
   - Isolates the empirical value of GAMA's Bayesian abduction core: compares how the LLM decides freely versus how the calibrated Bayesian belief decides.

3. **`belief_llm.py` (Belief $\rightarrow$ LLM)**:
   - Evaluates unconstrained LLM reasoning when provided with both the calibrated evidence tables AND GAMA's computed Bayesian belief ranking.
   - Tests whether a capable LLM improves or damages the computed belief when allowed to override it freely.

4. **`__init__.py` (`BASELINE_REGISTRY`)**:
   - Central registration dictionary accessed by `main.py --systems <name>`.
   - Supports:
     - `gama` (Proposed GAMA-RCA framework)
     - `belief_llm` (Belief $\rightarrow$ LLM adjudication)
     - `evidence_llm` (Evidence $\rightarrow$ LLM)
     - `direct_llm` (Direct LLM)
