# Pre-registration: evidence-to-label layer, held-out evaluation

Written and committed BEFORE any run on the 73 held-out cases. Tag `fp-freeze-v1` marks the frozen code. After this tag, no rule,
fingerprint, threshold, prompt or parameter of the arms below may change; a code change is allowed only to fix a crash, and is then
recorded here with its commit.

## Frozen method
- Code: `pipeline/fault_calibrated.py` (Algorithms 1-3), fingerprint table `FINGERPRINTS` in that file.
- Parameters: alpha = 0.05, top-K = 3, history B = 4 window lengths, 5 pseudo-windows (logs: 1 / 1). Backbone Qwen2.5-7B-Instruct,
  deterministic decoding (temperature 0, seed 0).

## Label-free validity check on dev-30 (fp_oracle_coverage.py, commit bf541ba)
- Held-out pseudo-window false-alarm rate per signal: calibrated 147/4446 = 3.3%; uncalibrated rank test 1727/4446 = 38.8% (target 5%).
- Oracle-entity coverage on dev-30 (evaluation only; reported, not used to change anything): true type ranked first in 10/30,
  among the candidates in 10/30, no candidate in 5/30.

## Development history (disclosed)
- a56af13: first fingerprint version with fixed a-priori thresholds (kept as ablation `_FIX`).
- e14ddff: rule changes chosen after looking at dev-30 labels; reverted in 1d3a84b, not part of the method.
- f2e2bf4: calibrated layer (rank tests, categorical observations, (entity, type) hypotheses).
- bf541ba: empirical-null calibration, added because the label-free false-alarm check of f2e2bf4 failed (~30%).

## Held-out run (cases = HELDOUT, 73 cases, each arm run once)
| Arm | Role |
|---|---|
| `A1rT_fixed_ABS_KP` | reference: free fault typing by the Coordinator |
| `A1rT_fixed_ABS_KP_FPSL` | **primary**: constrained choice among the top-3 calibrated hypotheses |
| `A1rT_fixed_ABS_KP_FPRULE` | top hypothesis, no LLM typing call |
| `A1rT_fixed_ABS_KP_FPSHOW` | evidence profile shown, free choice |
| `A1rT_fixed_ABS_KP_FPRULE_FIX` | ablation: fixed thresholds instead of calibration |
Secondary (if GPU time allows, same protocol): `A1rT_fixed_ABS_KP_FPSL_VER`; backbone `qwen3_8b` for `ABS_KP` and `FPSL`.

## Hypotheses and analysis
- **H1 (primary):** FI exact (28 classes) of `FPSL` > `ABS_KP` on the 73 held-out cases. Test: one-sided exact McNemar on the paired
  binary FI column, alpha = 0.05. Effect reported as the difference in FI with a paired bootstrap 95% CI (10 000 resamples,
  `scripts/summarize_arms.py`).
- **H2:** `FPRULE` vs `FPSL` (does the LLM add to the rules?), two-sided McNemar.
- **H3:** `FPRULE` vs `FPRULE_FIX` (does calibration matter?), two-sided McNemar.
- Reported for every arm: EL, FI exact, FI group, RP (local approximation), Final; per semantic group and per level (node / service);
  confusion matrix of fault types; share of cases with no admitted hypothesis; oracle-entity coverage on held-out; false-alarm rate.
- Secondary comparisons are exploratory and labelled as such. Dev-30 numbers are reported separately as development results.
