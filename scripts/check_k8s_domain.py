"""
check_k8s_domain.py
=====================
Follow-up to check_values.py: entity_id in metrics.parquet is empty string
("") for every row, not NaN, so the previous isna()-based query returned
nothing useful. This checks entity_name/service directly for domain=="k8s"
vs domain=="apm" rows, to confirm why k8s-domain rows fail to resolve.

Usage:
    python check_k8s_domain.py t003
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import pandas as pd
from data.loader import Case, build_name_index

case_id = sys.argv[1] if len(sys.argv) > 1 else "t003"
case = Case(case_id)

df = pd.read_parquet(f"{case.case_dir}/metrics.parquet",
                      columns=["entity_id", "entity_name", "service", "domain", "entity_set"])

print("=== domain == 'k8s' rows: entity_name / service / entity_set sample ===")
print(df[df["domain"] == "k8s"][["entity_name", "service", "entity_set"]].drop_duplicates().head(15))

print("\n=== domain == 'apm' rows: entity_name / service / entity_set sample ===")
print(df[df["domain"] == "apm"][["entity_name", "service", "entity_set"]].drop_duplicates().head(15))

print("\n=== topology: what k8s.node / k8s.pod entities look like (name + attributes) ===")
for node_id, data in case.topology.nodes(data=True):
    entity = data.get("entity")
    if entity and entity.entity_type in ("k8s.node", "k8s.pod"):
        print(f"  id={node_id} type={entity.entity_type} name={entity.name!r} "
              f"attrs={dict(list(entity.attributes.items())[:5])}")