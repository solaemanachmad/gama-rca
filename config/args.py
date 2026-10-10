"""
config/args.py
==============
Centralized CLI argument parser for every config toggle that's realistically
worth switching between runs, without hand-editing config/__init__.py each
time:

    python run_experiment.py --n_cases 20 --embedding-backend hf_api \
        --llm-backend gemini --hybrid-alpha 0.7 --hybrid-beta 0.3

Only entry-point scripts (run_experiment.py, smoke_test.py) call this --
importing config.args does NOT parse anything by itself, so it's safe to
import from anywhere without side effects.

Usage in an entry-point script:
    import config
    from config.args import build_parser, apply_overrides

    parser = build_parser()               # or build_parser(parser=existing_parser)
    args = parser.parse_args()
    apply_overrides(args)                 # mutates config.* in place
"""

import argparse


def build_parser(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    """Adds every config-overriding flag to `parser` (creating one if not
    given) and returns it. Safe to call on a parser that already has its
    own script-specific flags (e.g. run_experiment.py's --n_cases)."""
    if parser is None:
        parser = argparse.ArgumentParser()

    backend = parser.add_argument_group("backends")
    backend.add_argument("--embedding-backend", choices=["local", "hf_api"], default=None,
                          help="Override config.EMBEDDING_BACKEND (default: config value / "
                               "EMBEDDING_BACKEND env var / 'local').")
    backend.add_argument("--llm-backend", choices=["ollama", "gemini", "kaggle"], default=None,
                          help="Override config.LLM_BACKEND (default: config value / "
                               "LLM_BACKEND env var / 'ollama').")
    backend.add_argument("--llm-model", default=None,
                          help="Override the model name for whichever --llm-backend is "
                               "active (LLM_MODEL_NAME for ollama, GEMINI_MODEL_NAME for "
                               "gemini).")
    backend.add_argument("--no-embedding-cache", dest="embedding_cache_enabled",
                          action="store_false", default=None,
                          help="Disable the persistent disk embedding cache "
                               "(config.EMBEDDING_CACHE_ENABLED). Enabled by default.")

    retrieval = parser.add_argument_group("retrieval")
    retrieval.add_argument("--graph-hop-limit", type=int, default=None,
                            help="Override config.GRAPH_HOP_LIMIT (Phase A BFS radius).")
    retrieval.add_argument("--infra-search-hop-cap", type=int, default=None,
                            help="Override config.INFRA_SEARCH_HOP_CAP (Phase B infra search).")
    retrieval.add_argument("--hybrid-alpha", type=float, default=None,
                            help="Override config.HYBRID_ALPHA (graph-score weight). "
                                 "RQ2 sweep knob.")
    retrieval.add_argument("--hybrid-beta", type=float, default=None,
                            help="Override config.HYBRID_BETA (vector-score weight). "
                                 "RQ2 sweep knob.")
    retrieval.add_argument("--vector-top-k", type=int, default=None,
                            help="Override config.VECTOR_TOP_K.")

    perf = parser.add_argument_group("performance / dev mode")
    perf.add_argument("--dev-quick-test", dest="dev_quick_test", action="store_true",
                       default=None, help="Force config.DEV_QUICK_TEST = True.")
    perf.add_argument("--no-dev-quick-test", dest="dev_quick_test", action="store_false",
                       default=None, help="Force config.DEV_QUICK_TEST = False "
                                          "(use for the real 103-case run).")
    perf.add_argument("--max-observations-per-index", type=int, default=None,
                       help="Override config.MAX_OBSERVATIONS_PER_INDEX.")
    perf.add_argument("--max-resolved-per-modality", type=int, default=None,
                       help="Override config.MAX_RESOLVED_PER_MODALITY.")

    return parser


def apply_overrides(args: argparse.Namespace) -> None:
    """Mutates config.* in place for every flag in `args` that was actually
    provided (i.e. not None / not left at its argparse default). Call this
    once, right after parser.parse_args(), before constructing any pipeline/
    client objects -- those read config.* at construction time."""
    import config  # local import: avoids a circular import at module load time

    if getattr(args, "embedding_backend", None) is not None:
        config.EMBEDDING_BACKEND = args.embedding_backend
    if getattr(args, "llm_backend", None) is not None:
        config.LLM_BACKEND = args.llm_backend
    if getattr(args, "llm_model", None) is not None:
        if config.LLM_BACKEND == "gemini":
            config.GEMINI_MODEL_NAME = args.llm_model
        else:
            config.LLM_MODEL_NAME = args.llm_model
    if getattr(args, "embedding_cache_enabled", None) is not None:
        config.EMBEDDING_CACHE_ENABLED = args.embedding_cache_enabled

    if getattr(args, "graph_hop_limit", None) is not None:
        config.GRAPH_HOP_LIMIT = args.graph_hop_limit
    if getattr(args, "infra_search_hop_cap", None) is not None:
        config.INFRA_SEARCH_HOP_CAP = args.infra_search_hop_cap
    if getattr(args, "hybrid_alpha", None) is not None:
        config.HYBRID_ALPHA = args.hybrid_alpha
    if getattr(args, "hybrid_beta", None) is not None:
        config.HYBRID_BETA = args.hybrid_beta
    if getattr(args, "vector_top_k", None) is not None:
        config.VECTOR_TOP_K = args.vector_top_k

    if getattr(args, "dev_quick_test", None) is not None:
        config.DEV_QUICK_TEST = args.dev_quick_test
    if getattr(args, "max_observations_per_index", None) is not None:
        config.MAX_OBSERVATIONS_PER_INDEX = args.max_observations_per_index
    if getattr(args, "max_resolved_per_modality", None) is not None:
        config.MAX_RESOLVED_PER_MODALITY = args.max_resolved_per_modality


EFFECTIVE_CONFIG_KEYS = [
    "EMBEDDING_BACKEND", "LLM_BACKEND", "LLM_MODEL_NAME", "GEMINI_MODEL_NAME",
    "KAGGLE_MODEL_HANDLE", "GRAPH_HOP_LIMIT", "INFRA_SEARCH_HOP_CAP", "HYBRID_ALPHA",
    "HYBRID_BETA", "VECTOR_TOP_K", "DEV_QUICK_TEST", "MAX_OBSERVATIONS_PER_INDEX",
    "MAX_RESOLVED_PER_MODALITY", "TEMPORAL_BOOST_ENABLED",
    "USE_PROPAGATION_EVIDENCE", "USE_INFRA_EVIDENCE", "CHAIN_FIRST",
    "USE_LOG_TEMPLATES", "ENTITY_SOURCE",
    "USE_COLOCATION_EVIDENCE", "LAYER_AGENT", "CHAIN_FIRST_ENTITY", "USE_CHANGE_FACTOR", "USE_KEYWORD_HINT", "USE_CHANGE_STATS",
]

# Console summary: hide settings that do not affect the run being started.
_HIDE_WHEN_LOG_TEMPLATES = ("MAX_OBSERVATIONS_PER_INDEX", "MAX_RESOLVED_PER_MODALITY")
_ALWAYS_HIDDEN = ("DEV_QUICK_TEST", "GRAPH_HOP_LIMIT", "INFRA_SEARCH_HOP_CAP", "TEMPORAL_BOOST_ENABLED")


def get_effective_config() -> dict:
    """Same fields as print_effective_config(), but as a dict -- reused for
    wandb.init(config=...) so a run's hyperparameters are traceable in the
    W&B dashboard, not just in console output."""
    import config
    return {k: getattr(config, k, None) for k in EFFECTIVE_CONFIG_KEYS}


def print_effective_config() -> None:
    """Prints the config values that actually matter for reproducing a run
    -- handy to log at the start of run_experiment.py so results.csv runs
    are traceable back to the settings that produced them.

    Backend-aware: only shows the model setting for whichever LLM_BACKEND
    is actually active, not all three possible backends' model names at
    once (confusing -- e.g. showing GEMINI_MODEL_NAME while running on
    ollama). get_effective_config() (used for W&B logging) stays complete/
    unfiltered, since a full record is useful there for reproducibility;
    this is just the human-readable console summary."""
    import config
    cfg = get_effective_config()
    backend = cfg.get("LLM_BACKEND")
    active_model_key = {"ollama": "LLM_MODEL_NAME", "gemini": "GEMINI_MODEL_NAME",
                         "kaggle": "KAGGLE_MODEL_HANDLE"}.get(backend)
    print("=== Effective config for this run ===")
    for k, v in cfg.items():
        if k in ("LLM_MODEL_NAME", "GEMINI_MODEL_NAME", "KAGGLE_MODEL_HANDLE") and k != active_model_key:
            continue  # skip inactive backends' model settings
        if k in _ALWAYS_HIDDEN:
            continue
        if cfg.get("USE_LOG_TEMPLATES") and k in _HIDE_WHEN_LOG_TEMPLATES:
            continue
        print(f"  {k} = {v}")
    print()