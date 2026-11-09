"""
config/args.py
==============
Centralized CLI argument parser for every config toggle that's realistically
worth switching between runs, without hand-editing config/__init__.py each
time:

    python main.py --n_cases 20 --embedding-backend hf_api \
        --hybrid-alpha 0.7 --hybrid-beta 0.3

Only entry-point scripts (main.py, smoke_test.py) call this --
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
    own script-specific flags (e.g. main.py's --n_cases)."""
    if parser is None:
        parser = argparse.ArgumentParser()

    backend = parser.add_argument_group("backends")
    backend.add_argument("--embedding-backend", choices=["local", "hf_api"], default=None,
                          help="Override config.EMBEDDING_BACKEND (default: config value / "
                               "EMBEDDING_BACKEND env var / 'local').")
    backend.add_argument("--llm-backend", choices=["kaggle"], default=None,
                          help="Override config.LLM_BACKEND (default: config value / "
                               "LLM_BACKEND env var / 'kaggle').")
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


def get_effective_config() -> dict:
    """Returns only the active hyperparameters and configuration for the current run."""
    import config
    backend = getattr(config, "LLM_BACKEND", "kaggle")
    cfg = {
        "LLM_BACKEND": backend,
        "GAMA_MODULES": os.environ.get("GAMA_MODULES", "hybrid"),
        "ALERT_T0_MODE": getattr(config, "ALERT_T0_MODE", "window"),
        "FP_SCORING": getattr(config, "FP_SCORING", "calibrated"),
        "FP_ALPHA": getattr(config, "FP_ALPHA", 0.05),
        "FP_TOP_K": getattr(config, "FP_TOP_K", 3),
        "LLM_TEMPERATURE": getattr(config, "LLM_TEMPERATURE", 0.0),
        "LLM_SEED": getattr(config, "LLM_SEED", 0),
        "LLM_MAX_TOKENS": getattr(config, "LLM_MAX_TOKENS", 1024),
    }
    if backend == "api":
        cfg["API_MODEL"] = os.environ.get("API_MODEL", "")
        cfg["API_BASE_URL"] = os.environ.get("API_BASE_URL", "")
        if os.environ.get("API_REASONING_EFFORT"):
            cfg["API_REASONING_EFFORT"] = os.environ.get("API_REASONING_EFFORT")
    elif backend == "kaggle":
        cfg["KAGGLE_MODEL_HANDLE"] = getattr(config, "KAGGLE_MODEL_HANDLE", "")
        cfg["KAGGLE_DTYPE"] = getattr(config, "KAGGLE_DTYPE", "bfloat16")
        cfg["KAGGLE_USE_4BIT"] = getattr(config, "KAGGLE_USE_4BIT", False)
    return cfg


EFFECTIVE_CONFIG_KEYS = list(get_effective_config().keys())


def print_effective_config() -> None:
    """Prints a concise summary of the active configuration."""
    cfg = get_effective_config()
    print("=== Active Configuration ===")
    for k, v in cfg.items():
        print(f"  {k} = {v}")
    print()