# Pre-registration v6: an LLM that contributes where the statistics are unsure (2026-11-09, before any v6 run)

## Why v6, and what is disclosed

The v5 study on RCA100 (PREREGISTRATION_V5.md) found that the bounded LLM (module M7) lowers the decision score of the statistical
core by 3.5-3.7 Final points with Qwen2.5-7B and gpt-oss-120b, while a larger model writes a much better explanation. An audit of
those runs (claude/GAMA-RCA_audit_LLM_role.md in the project) produced three observations. They were made AFTER seeing RCA100
results and are therefore hypotheses, not findings:

- O1 The posterior knows when it is unsure: P(entity) >= 0.95 -> exact entity in 91% of 46 cases; P(entity) < 0.8 -> 31% of 42 cases.
  On those 42 cases an LLM that decides from the evidence alone (Evidence->LLM, 7B) names the exact entity in 19 vs 13 for GAMA.
  Given the posterior (Belief->LLM) the same model copies it (no independent opinion).
- O2 Most harmful LLM type changes swap two types whose fingerprints share the quoted signal (cpuFullLoad -> cpuDeadLoop, both keyed
  on svc_cpu_rise; memoryPressure -> fullGC via svc_gc_rise, a support signal of memoryPressure).
- O3 The text sensor moved a correct entity away in the same four cases with both backbones: the sensor's likelihood term, not the
  model, moves the entity.

v6 turns O1-O3 into three rules with constants that already exist in the frozen code. RCA100 results of v6 are DEVELOPMENT results
(reported as exploratory). The confirmatory test is AIOps2025, which has not been run with any arm when this file is committed.

## The three rules (modules; all off in v5 "core" and "full", so every v5 result is unchanged)

G1 `gate` -- second opinion when unsure (requires `llm`).
   After the computed decision, if P(e*) < STOP_P (= 0.8, the existing sensor stop rule, pipeline/belief_diagnosis.py), ONE LLM call
   is made with the Evidence->LLM prompt (pipeline/evidence_llm.build_prompt, unchanged, WITHOUT the belief block, so the opinion is
   independent; evidence = the observations that hold in GAMA's own evidence, log templates as for the sensor). Only the entity of the
   answer is used. If it names an entity of the hypothesis space other than e*, the decision is recomputed with that entity fixed:
   the type is argmax P(type | entity, evidence) of the same model (two-stage decision, M3 re-weighting, fallback rule), computed
   without further text probes; evidence list and chain follow the new entity. Otherwise (same entity, unknown name, parse failure) the
   computed decision stands. P(e*) >= 0.8: no call.
G2 `defer` -- the LLM may change the type only with discriminating evidence (requires `llm`).
   Shortlist typing as in v5 (top 5, verbatim quote >= 10 characters, the quoted line must name a key / support / may signal of the
   chosen type). New: when the chosen type differs from the computed one, at least one quoted line must name a signal that is NOT a
   key or support signal of the computed type. Otherwise the computed type stands.
G3 `calsensor` -- the text sensor may re-weight types, not move the entity (requires `llm`).
   Probe selection, budget (5), stop rule (0.8) and the category likelihoods are unchanged. The update is renormalised within the
   probed entity: every hypothesis of that entity is multiplied by its likelihood divided by the posterior-weighted mean likelihood of
   the entity's hypotheses, so P(entity) is unchanged by a probe.

Spec: `v6` = full (M1-M7) + gate + defer + calsensor. Narrator unchanged (scored separately; the scored chain stays computed).
No constant is new: STOP_P = 0.8, shortlist 5, quote >= 10 characters, probe budget 5 are the v5 values.

## Arms and runs

AIOps2025 (confirmatory; 400 cases; scorer = re-implemented Cai et al. protocol, PREREGISTRATION_AIOPS2025.md):
- GAMA core (CPU; already pre-registered)
- GAMA v6, gpt-oss-120b (Cerebras API, reasoning medium, temperature 0, seed 0, settings of the V5 F4 amendment)
- Evidence->LLM, gpt-oss-120b (the LLM decides from the same evidence; reference for "LLM without the computed belief")
- optional, if budget allows: GAMA full (v5 M7) gpt-oss-120b; the same three arms with Qwen2.5-7B (Kaggle GPU)
RCA100 (exploratory, development data): GAMA v6 with gpt-oss-120b and with Qwen2.5-7B, all 103 cases.

LLM inputs on AIOps2025: the same reduced logs as the core (amendment I3: lines that match the core's text families); fault-type
definitions of the AIOps2025 taxonomy (data/aiops2025_profile.py, written from the published type list) replace the RCA100 ones.
The Evidence->LLM reference arm on AIOps2025 uses the prompt of the RCA100 baseline (pipeline/evidence_llm.build_prompt) with the
observations that hold in GAMA's own evidence (gama/belief_llm.run_belief_llm(with_belief=False), the same evidence as the G1 call);
the RCA100 EVIDENCE_LLM_FINAL arm is unchanged. Answers of every arm use the same submission format (scripts/aiops2025_run.answer:
taxonomy phrase, component, at most 5 trace steps from the holding evidence). Commands:
`python scripts/aiops2025_run.py [--arm gama|evidence_llm] [--modules core|v6] [--model gptoss120b_cerebras] --shard k/4`.

## Hypotheses and analysis (AIOps2025)

- V1 (confirmatory, one-sided): Final(GAMA v6) > Final(GAMA core). Paired Wilcoxon on per-case Final, 95% bootstrap CI (5,000, seed 0).
- V2 (confirmatory, one-sided): Final(GAMA v6) > Final(Evidence->LLM), same backbone.
- V3 (cost, descriptive): tokens per case of v6 vs GAMA full and vs Evidence->LLM; share of cases with a G1 call.
- V4 (mechanism, descriptive): on the gated cases, exact component of GAMA vs v6; G2 rejections; G3 type changes.
Holm correction over V1-V2. Reported whatever the outcome; v6 is not modified after the first AIOps2025 run.

## Budget

Cerebras free trial (USD 5, 1M tokens per day, 5 requests per minute): ~10-11K tokens per v6 case and ~5K per Evidence->LLM case
-> ~6.5M tokens for the 400-case AIOps2025 arms (~USD 2.5-3) over ~7 days, plus ~1.1M tokens for RCA100 v6. If credits or days
run out, the AIOps2025 arms are completed with the same model on a paid tier, never with a different model.
