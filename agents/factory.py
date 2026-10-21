"""agents/factory.py -- builds the LLM client(s).

Single backend: a local transformers model (agents/kaggle_client.py). No server,
no external API: RCA100 data never leaves the machine.
"""

import config


def get_llm_client():
    from agents.kaggle_client import KaggleTransformersClient
    return KaggleTransformersClient(model_handle=config.KAGGLE_MODEL_HANDLE,
                                     local_path=config.KAGGLE_MODEL_LOCAL_PATH,
                                     use_4bit=config.KAGGLE_USE_4BIT,
                                     enable_thinking=config.ENABLE_THINKING)


def get_coordinator_llm_client():
    """A SEPARATE client for just the Coordinator, only if COORDINATOR_MODEL_HANDLE or
    COORDINATOR_MODEL_LOCAL_PATH is set. None otherwise (callers reuse the main client)."""
    if not config.COORDINATOR_MODEL_HANDLE and not config.COORDINATOR_MODEL_LOCAL_PATH:
        return None
    from agents.kaggle_client import KaggleTransformersClient
    return KaggleTransformersClient(
        model_handle=config.COORDINATOR_MODEL_HANDLE,
        local_path=config.COORDINATOR_MODEL_LOCAL_PATH,
        use_4bit=config.COORDINATOR_USE_4BIT,
        enable_thinking=config.ENABLE_THINKING,
    )
