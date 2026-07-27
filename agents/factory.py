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
        return KaggleTransformersClient(model_handle=config.KAGGLE_MODEL_HANDLE)
    from agents.llm_client import LLMClient
    return LLMClient()