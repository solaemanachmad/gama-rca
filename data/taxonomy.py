"""data.taxonomy
=============
Backward-compatibility taxonomy forwarding to data.rca100.taxonomy.
For RCA100 taxonomy and definitions, prefer importing from data.rca100.taxonomy directly.
"""

from data.rca100.taxonomy import (
    FAULT_DEFINITIONS,
    FAULT_GROUPS,
    FAULT_TYPE_LABELS,
    _TAXONOMY_CACHE,
    build_fault_taxonomy,
    fault_group,
    knowledge_prompt_block_pos,
    taxonomy_prompt_block,
)

__all__ = [
    "FAULT_DEFINITIONS",
    "FAULT_GROUPS",
    "FAULT_TYPE_LABELS",
    "_TAXONOMY_CACHE",
    "build_fault_taxonomy",
    "fault_group",
    "knowledge_prompt_block_pos",
    "taxonomy_prompt_block",
]
