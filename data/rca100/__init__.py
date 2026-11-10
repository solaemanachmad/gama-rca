"""data.rca100
============
RCA100 benchmark dataset loader and fault taxonomy.
"""

from data.rca100.loader import Case, list_case_ids, resolve_entity_by_name, normalize_entity_ids
from data.rca100.taxonomy import (
    FAULT_GROUPS,
    FAULT_DEFINITIONS,
    taxonomy_prompt_block,
    fault_group,
)

__all__ = [
    "Case",
    "list_case_ids",
    "resolve_entity_by_name",
    "normalize_entity_ids",
    "FAULT_GROUPS",
    "FAULT_DEFINITIONS",
    "taxonomy_prompt_block",
    "fault_group",
]
