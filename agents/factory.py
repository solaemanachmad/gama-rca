"""
agents/factory.py
===================
Single place that decides which LLM client backend to instantiate, based on
config.LLM_BACKEND ("ollama" default, or "gemini"). Every entry point
(run_experiment.py, smoke_test.py) and any pipeline default should call
get_llm_client() instead of hardcoding LLMClient() directly -- that's what
makes `--llm-backend gemini` on the CLI (config/args.py) actually take
effect without touching pipeline/pipeline.py or pipeline/baselines.py.
"""

import config


def get_llm_client():
    if config.LLM_BACKEND == "gemini":
        from agents.gemini_client import GeminiClient
        return GeminiClient(model_name=config.GEMINI_MODEL_NAME)
    if config.LLM_BACKEND == "kaggle":
        from agents.kaggle_client import KaggleTransformersClient
        return KaggleTransformersClient(model_handle=config.KAGGLE_MODEL_HANDLE,
                                         local_path=config.KAGGLE_MODEL_LOCAL_PATH)
    from agents.llm_client import LLMClient
    return LLMClient()


def get_coordinator_llm_client():
    """Returns a SEPARATE client for just the Coordinator, only if
    COORDINATOR_MODEL_HANDLE or COORDINATOR_MODEL_LOCAL_PATH is explicitly
    set. Returns None otherwise -- callers should fall back to the same
    client used everywhere else (the validated default configuration).
    Only meaningful with LLM_BACKEND=kaggle; the other backends don't
    support per-role model overrides."""
    if config.LLM_BACKEND != "kaggle":
        return None
    if not config.COORDINATOR_MODEL_HANDLE and not config.COORDINATOR_MODEL_LOCAL_PATH:
        return None
    from agents.kaggle_client import KaggleTransformersClient
    return KaggleTransformersClient(
        model_handle=config.COORDINATOR_MODEL_HANDLE,
        local_path=config.COORDINATOR_MODEL_LOCAL_PATH,
        use_4bit=config.COORDINATOR_USE_4BIT,
    )