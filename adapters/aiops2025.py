"""adapters.aiops2025
==================
Backward-compatibility adapter forwarding to data.aiops2025.loader.
For AIOps2025 cases, prefer importing from data.aiops2025.loader directly.
"""

from data.aiops2025.loader import (
    AIOpsCase,
    DELETED,
    HISTORY,
    POD_RE,
    PROC_TAGS,
    Release,
    SAMPLE_MOD,
    SPAN_TAGS,
    TIDB,
    TS_RE,
    service_of,
)

__all__ = [
    "AIOpsCase",
    "DELETED",
    "HISTORY",
    "POD_RE",
    "PROC_TAGS",
    "Release",
    "SAMPLE_MOD",
    "SPAN_TAGS",
    "TIDB",
    "TS_RE",
    "service_of",
]
