"""Systems that main.py can run (--systems). The development systems (v1 GraphRAG / multi-agent, v2 hypotheses, v3-v4 belief arms)
are archived (archive/; git tag v5-frozen); `gama` reproduces the frozen v4/v5 arms exactly (PREREGISTRATION_V5.md, equivalence).

  gama          GAMA-RCA (modules from GAMA_MODULES: core | full | core,-<module> | base)
  belief_llm    GAMA's computed belief given to an LLM that decides freely
  evidence_llm  the LLM decides from the same calibrated evidence, without the belief
  direct_llm    the LLM sees the alert only
"""
import config

BASELINE_REGISTRY = {
    "gama": lambda case_id, llm, cases_dir=config.CASES_DIR: __import__(
        "gama.run", fromlist=["gama"]).gama(case_id, llm, cases_dir),
    "belief_llm": lambda case_id, llm, cases_dir=config.CASES_DIR: __import__(
        "gama.belief_llm", fromlist=["belief_llm"]).belief_llm(case_id, llm, cases_dir),
    "evidence_llm": lambda case_id, llm, cases_dir=config.CASES_DIR: __import__(
        "pipeline.evidence_llm", fromlist=["evidence_llm"]).evidence_llm(case_id, llm, cases_dir),
    "direct_llm": lambda case_id, llm, cases_dir=config.CASES_DIR: __import__(
        "pipeline.direct_llm", fromlist=["direct_llm"]).direct_llm(case_id, llm, cases_dir),
}
