"""
check_dependency_edges.py
============================
Checks whether topology.json encodes an explicit dependency edge from a
given entity (default: cart, t002's target) to a Redis/cache/middleware
node -- if so, that's a structural signal we could feed to the LLM/agent
as a hint ("cart depends on X"), instead of relying on lucky retrieval of
a specific log line to reveal a middleware dependency.

Usage:
    python scripts/check_dependency_edges.py t002 469f8e313055adba13ca3f4e76c65505
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import Case

case_id = sys.argv[1] if len(sys.argv) > 1 else "t002"
entity_id = sys.argv[2] if len(sys.argv) > 2 else "469f8e313055adba13ca3f4e76c65505"

case = Case(case_id)
g = case.topology

print(f"=== {case_id}: entity_id={entity_id} ===")
if entity_id not in g:
    print("entity not found in topology!")
    sys.exit(1)

entity = g.nodes[entity_id].get("entity")
print(f"entity: name={entity.name!r} type={entity.entity_type}")

print("\n=== ALL nodes with name/type suggesting cache/redis/middleware ===")
KEYWORDS = ("redis", "cache", "valkey", "middleware", "queue", "mq", "kafka", "rabbitmq")
found_any = False
for node_id, data in g.nodes(data=True):
    e = data.get("entity")
    if e is None:
        continue
    haystack = f"{e.name} {e.entity_type}".lower()
    if any(k in haystack for k in KEYWORDS):
        found_any = True
        print(f"  id={node_id} name={e.name!r} type={e.entity_type}")
if not found_any:
    print("  (none found -- no redis/cache/middleware-labeled node exists in this case's topology)")

print(f"\n=== Direct neighbors of {entity_id} ({entity.name}), both directions ===")
undirected = g.to_undirected(as_view=True)
for neighbor_id in undirected.neighbors(entity_id):
    ne = g.nodes[neighbor_id].get("entity")
    edge_data_fwd = g.get_edge_data(entity_id, neighbor_id) or {}
    edge_data_bwd = g.get_edge_data(neighbor_id, entity_id) or {}
    edge_type = edge_data_fwd.get("edge_type") or edge_data_bwd.get("edge_type") or "?"
    name = ne.name if ne else "?"
    etype = ne.entity_type if ne else "?"
    print(f"  -> id={neighbor_id} name={name!r} type={etype} edge_type={edge_type}")

print("\n=== ALL distinct edge_type values in this case's topology (for reference) ===")
edge_types = set()
for _, _, data in g.edges(data=True):
    edge_types.add(data.get("edge_type", data.get("type", "?")))
print(edge_types)
