"""
audit_fault_groups.py
========================
Systematic audit across the 6 RCA100 fault groups (Application logic, JVM
runtime, Cloud resource, Middleware&DB, K8s lifecycle, Resource&perf.):
for up to N cases per group (default 5), checks whether each of the 6
modalities carries ANY observation directly attributable to the ground-
truth root-cause entity (or its apm.service ancestor). Prints full detail
per case AND an aggregated summary (fraction of cases per group where each
modality is usable) -- the summary is what actually answers "is this a
one-off unlucky case or a structural pattern for this whole group",
BEFORE spending more effort on case-by-case retrieval/prompt tuning.

Cheap: only loads ground truth + raw parquet ingestion, no embedding model,
no LLM calls.

Usage:
    python scripts/audit_fault_groups.py                # auto-picks up to 5 cases/group
    python scripts/audit_fault_groups.py t002 t003 t017  # audit specific cases instead
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict

from data.loader import Case, list_case_ids, find_service_ancestor
from data.taxonomy import fault_group
from evaluation.scoring import load_ground_truth


def pick_cases_per_group(n_per_group: int = 5, max_scan: int = None):
    """Scans manifest.txt and returns {group_name: [case_id, ...]}, up to
    n_per_group cases per group."""
    case_ids = list_case_ids()
    if max_scan:
        case_ids = case_ids[:max_scan]
    picked = defaultdict(list)
    for cid in case_ids:
        try:
            gt = load_ground_truth(cid)
        except Exception as e:
            print(f"  [skip] {cid}: {type(e).__name__}: {e}")
            continue
        g = fault_group(gt.fault_type)
        if len(picked[g]) < n_per_group:
            picked[g].append(cid)
    return dict(picked)


def audit_case(case_id: str, verbose: bool = True):
    """Returns {modality: rolled_hits} for this case. Prints full detail if
    verbose=True."""
    case = Case(case_id)
    gt = load_ground_truth(case_id, name_index=case.name_index)
    targets = set(gt.target_entity_ids)
    target_rolled = {find_service_ancestor(t, case.topology) or t for t in targets}
    all_target_ids = targets | target_rolled

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"CASE {case_id}  |  fault_type={gt.fault_type}  |  group={fault_group(gt.fault_type)}")
        print(f"{'=' * 70}")
        for t in targets:
            in_topo = t in case.topology
            etype = case.topology.nodes[t]["entity"].entity_type if in_topo else "NOT IN TOPOLOGY"
            print(f"  target entity: {t}  (in_topology={in_topo}, type={etype})")
        print(f"\n  {'modality':10s} {'total_rows':>12s} {'resolved':>10s} {'direct_hits':>12s} {'rolled_hits':>12s} {'note'}")

    modality_hits = {}
    for modality, obs_list in case.observations.items():
        total = len(obs_list)
        resolved = sum(1 for o in obs_list if o.entity_id)
        direct_hits = sum(1 for o in obs_list if o.entity_id in all_target_ids)
        # Roll up the OBSERVATION side too: raw telemetry is tagged at
        # instance/operation granularity (e.g. "payment::grpc.../Charge"),
        # while GT targets are usually already service-level -- comparing
        # raw entity_id against an already-service-level target never
        # matches unless the observation itself IS bare-service-tagged,
        # which is rare. This is the same bug found and fixed in
        # retrieval_precision_recall() previously, mistakenly reintroduced
        # in this script's first version.
        rolled_hits = sum(1 for o in obs_list
                           if o.entity_id and (find_service_ancestor(o.entity_id, case.topology) or o.entity_id) in all_target_ids)
        modality_hits[modality] = rolled_hits
        if verbose:
            note = ""
            if total == 0:
                note = "(no data for this case)"
            elif rolled_hits == 0:
                note = "*** ZERO evidence tied to root-cause entity (even after roll-up) ***"
            print(f"  {modality:10s} {total:12d} {resolved:10d} {direct_hits:12d} {rolled_hits:12d}  {note}")

    return modality_hits


if __name__ == "__main__":
    N_PER_GROUP = 5
    if len(sys.argv) > 1:
        case_ids = sys.argv[1:]
        picked = {"(manual selection)": case_ids}
    else:
        print(f"Scanning manifest.txt for up to {N_PER_GROUP} cases per fault group...")
        picked = pick_cases_per_group(n_per_group=N_PER_GROUP)
        for g, ids in picked.items():
            print(f"  {g}: {ids}")

    all_modalities = ["metrics", "logs", "traces", "events", "alerts"]
    # {group: {modality: [n_usable, n_total]}}
    tally = {g: {m: [0, 0] for m in all_modalities} for g in picked}

    for group, ids in picked.items():
        for cid in ids:
            hits = audit_case(cid, verbose=True)
            for m in all_modalities:
                tally[group][m][1] += 1
                if hits.get(m, 0) > 0:
                    tally[group][m][0] += 1

    print(f"\n\n{'#' * 90}")
    print("SUMMARY: fraction of sampled cases per group where each modality has")
    print("ANY evidence tied to the root-cause entity (after service-level roll-up)")
    print(f"{'#' * 90}\n")
    header = f"{'group':20s}" + "".join(f"{m:>12s}" for m in all_modalities)
    print(header)
    print("-" * len(header))
    for group, mods in tally.items():
        row = f"{group:20s}"
        for m in all_modalities:
            usable, total = mods[m]
            row += f"{f'{usable}/{total}':>12s}"
        print(row)

    print(f"\n{'=' * 70}")
    print("Read the 'rolled_hits' column in each case's detail above (or the "
          "SUMMARY fractions): 0 means that modality can NEVER contribute "
          "evidence about the root cause for that case (even after rolling "
          "telemetry up to its apm.service ancestor) -- a structural data "
          "gap, not a retrieval bug. A LOW fraction in the summary (e.g. "
          "1/5) means it's usually a gap for that group, not just one "
          "unlucky case. Modalities with high fractions are where fixing "
          "retrieval/summarization can actually help.")