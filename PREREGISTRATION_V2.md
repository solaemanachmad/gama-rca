# Pre-registration v2: hypothesis-constrained RCA on the RCA100 held-out cases

Status: DRAFT. Becomes binding when tagged `v2-freeze` (after the fingerprint review and the synthetic LLM smoke test). After the tag,
code, fingerprints, prompts and parameters do not change; only crash fixes are allowed, each recorded below with its commit.

## Data
- Test set: the 73 held-out cases (RCA100 v1.1 minus DEV30 in scripts/kaggle_run.py). Never used before the freeze, not even label-free.
- Dev-30: used during development (DEVLOG_V2.md: round 0/1 with labels, then label-free checks only). Results on dev-30 may be reported
  separately as "seen data", never as evidence for the hypotheses.
- No pre-freeze check uses RCA100 data after this draft: mechanics are checked on synthetic cases (tests/, scripts/smoke_synthetic_llm.py).

## Systems (one run each on the 73 cases; Qwen2.5-7B-Instruct, temperature 0, seed 0)
| Arm | System | Role |
|---|---|---|
| `A1rT_fixed_ABS_KP` | v1 GAMA-RCA (free LLM diagnosis, 4 specialists + Coordinator) | baseline |
| `DIRECT_LLM` | one LLM call on the alert | baseline |
| `V2_RULE` | v2 Stages 1-3, no LLM | what the LLM adds |
| `V2` | v2 Stages 1-4 | **proposed** |
| `V2_RULE_NOTOPO`, `V2_RULE_NOEXCL`, `V2_RULE_UNCAL`, `V2_RULE_NOLOCAL`, `V2_RULE_LEGACYWIN`, `V2_NOVERIFY` | ablations | exploratory |
| `V2` with `qwen3_14b_4bit` | backbone | exploratory (if GPU time allows) |

## Hypotheses (confirmatory; Holm correction over H1-H2)
- **H1:** fault identification, exact 28-class (`fault_identification_exact`): V2 > v1. One-sided exact McNemar on paired binary outcomes.
- **H2:** entity localization, exact (`rcl_exact`): V2 > v1. One-sided exact McNemar.
- **H3 (secondary):** V2 vs V2_RULE on both outcomes, two-sided McNemar.
- Effect sizes: paired differences with bootstrap 95% CIs (10 000 resamples).

## Also reported (descriptive)
EL (local partial credit), FI group, RP chain-based (`reasoning_process_chain`), Final; per level (node / service) and per semantic
group; error decomposition (wrong entity / wrong type / both); abstain rate; hypothesis counts; false-alarm rate of the calibrated
tests; grounding rate of quoted numbers and LLM choice validity (faithfulness).

## Crash fixes after the freeze
| Commit | Reason | Effect |
|---|---|---|
