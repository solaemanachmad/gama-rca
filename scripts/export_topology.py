"""
export_topology.py
=====================
Exports the topology graph for one or more cases, both as structured JSON
(nodes + edges, for flexible reuse in any diagramming tool) and as a
rendered PNG diagram (networkx + matplotlib, immediately usable in the
paper), with the alert entity and ground-truth root-cause entity
highlighted.

Usage:
    python scripts/export_topology.py t001 t002 t003
    python scripts/export_topology.py t001 --max-nodes 40   # trim large graphs for readability
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import networkx as nx
import matplotlib.pyplot as plt

from data.loader import Case
from evaluation.scoring import load_ground_truth
from pipeline.pipeline import parse_alert

parser = argparse.ArgumentParser()
parser.add_argument("case_ids", nargs="+")
parser.add_argument("--out-dir", default="topology_exports")
parser.add_argument("--max-nodes", type=int, default=60,
                     help="If the topology has more nodes than this, trim to the "
                          "alert/GT entities and their immediate neighbors only, "
                          "for a readable diagram (full JSON export is unaffected).")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)


def export_case(case_id: str):
    case = Case(case_id)
    g = case.topology
    gt = load_ground_truth(case_id, name_index=case.name_index)
    parsed_alert = parse_alert(case)
    alert_entity = parsed_alert.get("entry_entity_id")
    gt_entities = set(gt.target_entity_ids)

    # --- Full structured export (JSON) --------------------------------------
    nodes_export = []
    for node_id, data in g.nodes(data=True):
        entity = data.get("entity")
        nodes_export.append({
            "id": node_id,
            "name": entity.name if entity else None,
            "type": entity.entity_type if entity else None,
            "is_alert_entity": node_id == alert_entity,
            "is_gt_target": node_id in gt_entities,
        })
    edges_export = []
    for u, v, data in g.edges(data=True):
        edges_export.append({"source": u, "target": v, "attributes": {k: str(v) for k, v in data.items()}})

    export = {
        "case_id": case_id,
        "gt_fault_type": gt.fault_type,
        "alert_entity": alert_entity,
        "gt_target_entities": list(gt_entities),
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "nodes": nodes_export,
        "edges": edges_export,
    }
    json_path = os.path.join(args.out_dir, f"{case_id}_topology.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)
    print(f"{case_id}: saved full topology ({g.number_of_nodes()} nodes, "
          f"{g.number_of_edges()} edges) to {json_path}")

    # --- Trimmed PNG diagram --------------------------------------------------
    draw_graph = g
    if g.number_of_nodes() > args.max_nodes:
        keep = set()
        if alert_entity and alert_entity in g:
            keep.add(alert_entity)
            keep.update(g.to_undirected(as_view=True).neighbors(alert_entity))
        for e in gt_entities:
            if e in g:
                keep.add(e)
                keep.update(g.to_undirected(as_view=True).neighbors(e))
        draw_graph = g.subgraph(keep).copy()
        print(f"  (trimmed to {draw_graph.number_of_nodes()} nodes around alert/GT entities for the PNG)")

    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    pos = nx.spring_layout(draw_graph, seed=42, k=0.8)

    node_colors = []
    for n in draw_graph.nodes():
        if n == alert_entity:
            node_colors.append("#e8622c")   # alert entity: orange
        elif n in gt_entities:
            node_colors.append("#2ca02c")   # ground-truth root cause: green
        else:
            node_colors.append("#9fb8d9")   # other: neutral blue-gray

    labels = {}
    for n, data in draw_graph.nodes(data=True):
        entity = data.get("entity")
        labels[n] = entity.name if entity else n[:8]

    nx.draw_networkx_edges(draw_graph, pos, ax=ax, alpha=0.4, arrows=True, arrowsize=8)
    nx.draw_networkx_nodes(draw_graph, pos, ax=ax, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(draw_graph, pos, labels=labels, ax=ax, font_size=7)

    ax.set_title(f"{case_id} -- {gt.fault_type}\n"
                 f"orange = alert entity, green = ground-truth root cause",
                 fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    png_path = os.path.join(args.out_dir, f"{case_id}_topology.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved diagram to {png_path}")


for cid in args.case_ids:
    try:
        export_case(cid)
    except Exception as e:
        print(f"[skip] {cid}: {type(e).__name__}: {e}")