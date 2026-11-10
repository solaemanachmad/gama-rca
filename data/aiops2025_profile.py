"""data.aiops2025_profile
======================
Backward-compatibility profile forwarding to data.aiops2025.taxonomy.
For AIOps2025 taxonomy and fingerprints, prefer importing from data.aiops2025.taxonomy directly.
"""

from data.aiops2025.taxonomy import (
    DEFINITIONS,
    FALLBACK,
    FINGERPRINTS,
    LEVEL,
    NODE_EXTRA,
    RESIDUAL,
    SVC_EXTRA,
    phrase,
)

__all__ = [
    "DEFINITIONS",
    "FALLBACK",
    "FINGERPRINTS",
    "LEVEL",
    "NODE_EXTRA",
    "RESIDUAL",
    "SVC_EXTRA",
    "phrase",
]
