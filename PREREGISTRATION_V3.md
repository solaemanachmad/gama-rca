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
| `V3` | belief_rca: computed belief + EIG text probes (budget 5) + narrator | **proposed** |
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

## Crash fixes after the freeze
| Commit | Reason | Effect |
|---|---|---|
