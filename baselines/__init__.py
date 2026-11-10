"""baselines
=========
Comparative baseline systems evaluated alongside GAMA-RCA (Table 4 in paper).

  gama          GAMA-RCA (proposed framework)
  belief_llm    GAMA's computed belief given to an LLM that decides freely
  evidence_llm  the LLM decides from the same calibrated evidence, without the belief
  direct_llm    the LLM sees the alert only
"""
import config

BASELINE_REGISTRY = {
    "gama": lambda case_id, llm, cases_dir=config.CASES_DIR: __import__(
        "gama.run", fromlist=["gama"]).gama(case_id, llm, cases_dir),
    "belief_llm": lambda case_id, llm, cases_dir=config.CASES_DIR: __import__(
        "baselines.belief_llm", fromlist=["belief_llm"]).belief_llm(case_id, llm, cases_dir),
    "evidence_llm": lambda case_id, llm, cases_dir=config.CASES_DIR: __import__(
        "baselines.evidence_llm", fromlist=["evidence_llm"]).evidence_llm(case_id, llm, cases_dir),
    "direct_llm": lambda case_id, llm, cases_dir=config.CASES_DIR: __import__(
        "baselines.direct_llm", fromlist=["direct_llm"]).direct_llm(case_id, llm, cases_dir),
}
