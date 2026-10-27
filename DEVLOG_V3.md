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
