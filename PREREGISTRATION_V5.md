# Pre-registration v5 — frozen configuration, odd/even protocol and multi-backbone study

Written 2026-10-31, BEFORE any v5 run. Amendments are appended with a date; nothing above an amendment is edited.

## 1. Status of the data

RCA100 (103 cases) is DEVELOPMENT data: v3 and v4 were designed and evaluated on all 103 cases and their per-case errors were
inspected (DEVLOG_V4.md). Scores on RCA100 are therefore optimistic and are reported as development results.

## 2. Frozen configuration A

GAMA-RCA = `belief_v4` with `V4_PARTS=all` (= `DEFAULT_PARTS`: score, open, twostage, fp, peer, prop, events, avail, onset, em,
selfnorm, llmtype, conseq, retry), V3_LOG_MODE=template, ALERT_T0_MODE=window, hyperparameters as in the code at the commit that adds
this file. Reference results: V4 + Qwen3-8B (Final 50.4) and V4_RULE (Final 50.0), DEVLOG_V4.md dev run 6.

## 3. Label-leakage audit (2026-10-31)

- Agent-facing code reads only the case's telemetry, topology, alert and task window; `answer_key/` is read only by
  `evaluation/scoring.py` and by diagnostics. Enforced by `tests/test_v5.py::agent_code_never_reads_labels`.
- The fault-type list and definitions are static schema text (`data/taxonomy.py`), identical for every system.
- No component is fitted to labels. Fixed constants are listed in Appendix B of the paper.
- Design leakage remains: fingerprint additions (fp, conseq) and the choice of parts were informed by errors on RCA100. This is the
  reason for sections 4 and 6.
- LLM contamination: all backbones (Qwen2.5 2024, Llama 3.1 2024, Gemma 2 2024, Qwen3 2025) were released before the RCA100
  benchmark paper (2026) and answer keys are distributed separately; memorisation of answers is unlikely. The LLM never receives an
  answer, a label or a score.

## 4. Protocol for new changes (v5): odd / even split

- DEV_ODD = cases with an odd number (t001, t003, …; 52 cases). VAL_EVEN = even numbers (51 cases). Defined in scripts/kaggle_run.py.
- New changes are designed by looking ONLY at odd cases (scripts/diagnostics/entity_errors_dev.py refuses even cases).
- Each change is stated as a mechanism (not a case), with its constants fixed before the first run, and recorded as an amendment.
- After the design is frozen, VAL_EVEN is run ONCE for configuration A and A + changes, and both halves are reported.
- Acceptance (decided now): a change is kept only if Final on VAL_EVEN improves by >= 2.0 points over A and the one-sided paired
  Wilcoxon test gives p < 0.05. Otherwise A stays the final configuration and the change is reported as a negative result.
- Disclosure: per-case results of even cases from v4 runs exist and were seen; the split protects the design of the NEW changes only.

Planned changes (mechanisms; exact form fixed in an amendment after the odd-case diagnosis):
- C1 chronically anomalous entities: an entity whose signals are flagged in the baseline pseudo-windows of the same case is less
  informative when flagged in the window (label-free; measured on pseudo-windows only). Motivation: recommendation is predicted in
  22 cases and is never the true root.
- C2 node versus service: a node fault should show at several services hosted on that node at the same time; a single anomalous
  service on a node favours the service.

## 5. Multi-backbone study (after the configuration is final)

Arms on all 103 cases, each with the backbones qwen3_8b_local, llama31_8b, gemma2_9b, qwen25_7b_v5 (optional qwen3_4b):
- V4 (GAMA-RCA), EVIDENCE_LLM (the LLM decides from the same calibrated evidence, one call), DIRECT_LLM (alert only).
- Identical prompts, greedy decoding, bfloat16, thinking off; a template without a system role gets the system text prepended to the
  user message (agents/kaggle_client.render_chat), recorded per run.
Hypotheses: H1 for every backbone, Final(V4) > Final(EVIDENCE_LLM) (paired Wilcoxon, Holm-corrected over backbones);
H2 the Final scores of V4 across backbones lie within 3 points of each other (the statistical core carries the result).

## 6. Analysis plan

Metrics: EL, FI, RP (node-match), Final = 0.4 EL + 0.3 FI + 0.3 RP; cases with exact entity / exact type. 95% percentile bootstrap
intervals (5,000 resamples). Paired Wilcoxon signed-rank tests; Holm correction within each family of comparisons. All runs are
reported, including negative ones.
