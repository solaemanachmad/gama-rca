# Pre-registration v3: computed belief states with information-gain text probes, RCA100 (all 103 cases)

Written and committed before any final run. Tag `v3-freeze` marks the frozen code. After the tag, code, fingerprints, likelihood
parameters, prompts and this analysis plan do not change; only crash fixes are allowed, recorded below with their commit.
Supersedes PREREGISTRATION.md (v1 FP arms) and PREREGISTRATION_V2.md (draft, not executed).

## Data and exposure
- All 103 cases of RCA100 v1.1, each system run once. Scoring as in main.py / evaluation/scoring.py (answer_key read only for scoring).
- Disclosure: earlier versions of this project (v1, v2 rounds 0-1; DEVLOG_V2.md) looked at the labels of 30 cases (DEV30 in
  scripts/kaggle_run.py). v3 was developed without labels (DEVLOG_V3.md: synthetic acceptance tests and a real-LLM smoke test on
  synthetic cases only). Results are additionally reported separately for those 30 and the other 73 cases.

## Systems (Qwen2.5-7B-Instruct local, temperature 0, seed 0; Kaggle arms)
| Arm | System | Role |
|---|---|---|
| `V3` | belief_rca: computed belief + EIG text probes (budget 5, preprocessed logs, Amendment 2) + narrator | **proposed** |
| `V3_RAWLOG` | belief_rca with the frozen raw log input (Amendment 2) | does log preprocessing help? |
| `V3_B0` | belief_rca_b0: no text probes, narrator | does the text sensor help? |
| `V3_RULE` | belief_rule: no LLM | what the LLM adds |
| `V2` | hypothesis_rca: one-shot constrained selection | previous design |
| `DIRECT_LLM` | one LLM call on the alert | lower bound |
PoS (Luo et al., 2026) is reported from its paper only (large API backbones; joint accuracy on the same 103 cases), as context. A local
PoS run was attempted and did not finish one case in 2 h on 2xT4 (cost evidence, reported).

## Fixed parameters
THETA_K 0.8, THETA_U 0.6, THETA_P 0.6, ALPHA 0.05, ALPHA_C 0.05, ALPHA_ALERT 0.01, H0_PRIOR 0.1, RESIDUAL_PRIOR 0.5, BG_PRIOR_WEIGHT 10,
STOP_P 0.8, EIG_MIN 0.05 bit, PROBE_BUDGET 5; calibration alpha 0.05, 5 pseudo-windows, history 4 window lengths; ALERT_T0_MODE=window.

## Outcomes and tests (scripts/analyze_final.py, committed with this file)
- Primary outcome: **joint accuracy** = exact root-cause entity (rcl_exact) AND fault type (ft_binary), as in PoS. Error rows count as wrong.
- **H1:** joint accuracy V3 > V2. **H2:** joint accuracy V3 > V3_B0. One-sided exact McNemar, Holm correction over H1-H2, alpha 0.05;
  paired bootstrap 95% CI of the difference.
- Exploratory (two-sided, uncorrected): V3 vs V3_RULE, V3 vs DIRECT_LLM, V2 vs DIRECT_LLM, V3_B0 vs V3_RULE on joint, entity and type.
- Descriptive: entity / type / group accuracy, RP (chain-based) and Final, LLM calls and seconds per case, probes per case, posterior at
  stop, by level (node / service) and by semantic fault group; the 30 / 73 split above.
- Faithfulness: share of accepted sensor answers with a verified quote; grounding rate of quoted numbers in the chain.

## Amendment 1 (before any 103-case result was seen)
Backbone for all LLM arms (V3, V3_B0, V2, DIRECT_LLM) changed from Qwen2.5-7B-Instruct to **Qwen3-8B** (local Kaggle copy
qwen-lm/qwen-3/transformers/8b/1, MODEL=qwen3_8b_local), thinking mode OFF, temperature 0, seed 0. Reason: the user's choice of a newer
open small model of the same size class; no 103-case run had finished (the first V3 run was stopped at 60 min with no result read).
Method code, parameters, hypotheses and analysis are unchanged. All arms of the comparison use the same backbone.

## Amendment 2 (before any 103-case result was seen): log preprocessing for the text sensor
Reason: the frozen sensor input (semantic_probes.collect_texts: first 20 distinct keyword-matching lines per entity in the window) depends
on vocabulary, drops frequency and time order, and cannot separate chronic errors from new ones (no baseline). New input
(pipeline/log_preprocess.py): regex masking + Drain (drain3 defaults: similarity 0.4, depth 4), one tree per service, no keyword filter;
per template n_b (pre-window) and n_w (window); one-sided Poisson test with lambda = (n_b + 0.5) T_w / T_b; Benjamini-Hochberg over the
case's window templates at q = 0.05; per entity at most 20 items of at most 200 characters (the same budget as before): Kubernetes events,
error-span messages (window only; traces have no baseline), then significant templates by p-value, each with one raw example (the only
quotable text) and its counts shown to the sensor. No log test when the log baseline is shorter than one window. Belief model, fingerprints,
parameters, probe selection and narrator are unchanged; arms V3_B0, V3_RULE, V2, DIRECT_LLM are unaffected.
- **V3** = belief_rca with V3_LOG_MODE=template (proposed). **V3_RAWLOG** = belief_rca with V3_LOG_MODE=raw (the frozen input; the run
  started under Amendment 1 at commit 393d690 is this arm: identical code path).
- **H3:** joint accuracy V3 > V3_RAWLOG, one-sided exact McNemar; Holm over H1-H3. Exploratory: V3_RAWLOG vs V3_B0.
- Label-free check before the freeze (scripts/diagnostics/logpre_check.py, CPU): runtime per case and the false-alarm rate on a pseudo
  window inside the log baseline, reported as is; no parameter is changed from its result.
- Tests: tests/test_logpre.py (8 synthetic checks).

## Crash fixes after the freeze
| Commit | Reason | Effect |
|---|---|---|
| (this commit) | Runner display only: the Kaggle progress counter counted v1 debug dumps, so it showed 0/103 for v3 runs; it now counts main.py's per-case lines. | None (no method code changed). |
| (this commit) | Reporting only: the llm_model column showed KAGGLE_MODEL_HANDLE (default Qwen2.5-7B) even when a local copy (KAGGLE_MODEL_LOCAL_PATH) was loaded; it now shows the local path when set. | None (label column only). |
