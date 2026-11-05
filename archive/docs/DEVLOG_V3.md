# v3 development log — computed belief states + information-gain probes (design: claude/RCA_v3_design_belief_eig.md)

Protocol: evaluation on all 103 RCA100 cases, once, after a freeze. Development uses NO labels: synthetic acceptance tests
(tests/test_belief.py), a real-LLM smoke test on synthetic cases (scripts/smoke_synthetic_llm.py), and mechanics-only runs (crash, time)
on real inputs. v1 (dev-30) and v2 (dev-30 rounds 0-1) looked at labels of 30 cases; this exposure is disclosed in the paper.

## Changes made while writing the synthetic acceptance tests (no RCA100 data involved)
1. Key expressions are alternatives ("any key suffices"): with n testable keys each holds with q = 1-(1-THETA_K)^(1/n), so that
   P(at least one key | h) = THETA_K. Otherwise every unobserved alternative (e.g. HTTP 403 for a 429 rate limit) penalised the true type.
2. Testability: a hypothesis enters the space only if one of its key expressions is observable for its entity; otherwise unconfirmable
   types collected generic supporting evidence for free.
3. Background rate of categorical observations: Beta prior with mean ALPHA and 10 pseudo-entities (a Laplace estimate from the one or two
   services that have spans gave 0.67 for a 429).
4. The alert is an observation (P = THETA_P if the hypothesis can reach the alerted service, ALPHA_ALERT = 0.01 otherwise); the root's own
   latency/error symptoms are expected (THETA_P), but not a specific status code.
5. Residual types (httpError5xx, codeDefect) get half the prior weight (they are defined as fallbacks).
Synthetic robustness over 15 seeds: rate limit at checkout with an unrelated strong GC anomaly 15/15; node CPU 15/15 (all confident);
OOM at cart with upstream victims 15/15 (all confident); no fault 15/15 with h0 above every hypothesis.

## Amendment 2: log preprocessing for the sensor (no RCA100 labels or results involved)
Motivation (design review, not data): the raw sensor input is biased by vocabulary, truncation order and chronic errors. Templates tested
against each service's own pre-window rate (pipeline/log_preprocess.py). Synthetic checks (tests/test_logpre.py, 8/8): a chronic error at
an unchanged rate is not selected and its entity is not probe-able; a new error is; a keyword-free surge is; variable parts collapse to one
template with counts; a drop in frequency is not selected; events and span messages are kept with counts; no baseline -> no log test; the
sensor rejects quotes taken from the count notes. The first final V3 run (raw input, commit 393d690) continues and becomes V3_RAWLOG.
