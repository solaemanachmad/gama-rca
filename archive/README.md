# archive/

Development code kept for provenance only. Nothing in the active code imports it.

- `pipeline/`, `retrieval/`, `agents/multi_agent.py`: v1–v4 arms (GraphRAG hybrid retrieval, multi-agent, hypothesis RCA, belief v3/v4).
- `scripts/equivalence.py`: the check that `gama/` reproduces the frozen v4 code (206/206 identical on RCA100 before archiving).
- `docs/`: earlier preregistrations and development logs.
- `tests/`: tests of the archived arms and prompt snapshots.

To run any of it, check out git tag `v5-frozen`, where these files are at their original paths.
