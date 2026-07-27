"""
check_node_reachability.py
============================
For t003 and t008, the GT k8s.node was NOT in the BFS subgraph
(hop_limit=3). This checks the TRUE shortest-path distance (undirected,
unbounded) from the alert entry entity to that node, and prints the edge
types along the path -- so we know whether to raise GRAPH_HOP_LIMIT or
whether the topology graph is simply missing a connecting edge type.

Usage:
    python check_node_reachability.py t003 afa894cb32aaf3beb1e86bf5a1f31649 54822d2959721bfc645f24ac2f5a3217
    python check_node_reachability.py t008 e433b3f842dc4e525a834701cc603c28 b3b2ffabc65e72425b34ea6a0caa18da
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import networkx as nx
from data.loader import Case

case_id, entry_id, target_id = sys.argv[1], sys.argv[2], sys.argv[3]
case = Case(case_id)
graph = case.topology
undirected = graph.to_undirected(as_view=True)

print(f"=== {case_id}: entry={entry_id} -> target={target_id} ===")
print(f"entry in graph: {entry_id in graph}, target in graph: {target_id in graph}")
print(f"total nodes in full topology: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")

if entry_id in undirected and target_id in undirected:
    if nx.has_path(undirected, entry_id, target_id):
        path = nx.shortest_path(undirected, entry_id, target_id)
        print(f"\nTRUE shortest path length (undirected, unbounded): {len(path) - 1} hops")
        print("Path (entity_id : entity_type : edge_type_to_next):")
        for i, node_id in enumerate(path):
            data = graph.nodes[node_id]
            entity = data.get("entity")
            etype = entity.entity_type if entity else "?"
            ename = entity.name if entity else "?"
            if i < len(path) - 1:
                nxt = path[i + 1]
                # look up edge type in either direction (original directed graph)
                edge_data = graph.get_edge_data(node_id, nxt) or graph.get_edge_data(nxt, node_id) or {}
                edge_type = edge_data.get("edge_type", edge_data.get("type", "?"))
                print(f"  [{i}] {node_id} ({etype}, {ename!r}) --[{edge_type}]-->")
            else:
                print(f"  [{i}] {node_id} ({etype}, {ename!r})  <-- TARGET")
    else:
        print("\nNO PATH EXISTS at all between entry and target, even unbounded. "
              "The two are in different connected components of the topology graph.")
else:
    print("entry or target missing from graph entirely -- check ID typos.")

print("\n=== all edge_type values present in this case's topology (for reference) ===")
edge_types = set()
for _, _, data in graph.edges(data=True):
    edge_types.add(data.get("edge_type", data.get("type", "?")))
print(edge_types)