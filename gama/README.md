# `gama/` — GAMA-RCA Reasoning Orchestrator

This directory contains the main **GAMA-RCA** framework for abductive root-cause analysis in microservices.

---

## 🏛️ Directory Structure

```
gama/
├── core/                     # Core Bayesian reasoning & abductive inference
│   ├── evidence.py           # Stage 1: Multi-modal evidence aggregation
│   ├── belief.py             # Stage 2: Bayesian belief computation & EM propagation
│   ├── chain.py              # Stage 3: Causal propagation walk along the call graph
│   ├── decide.py             # Stage 4: Decision engine (Three Rules: Gate, Defer, CalSensor)
│   └── llm.py                # Stage 5: Bounded LLM sensor & typing arbitration
├── config.py                 # Module switches (M1-M7) and ablation configurations
├── run.py                    # Benchmark case runner & pipeline coordinator
├── __init__.py               # Re-exports core modules and run entrypoints
└── README.md
```

---

## 🔄 The 5 Sequential Reasoning Stages

```
             ┌─────────────────────────────────────────────────────────┐
             │                1. gama.core.evidence                    │
             │ Extracts calibrated multi-modal observations (pipeline) │
             └───────────────────────────┬─────────────────────────────┘
                                         │
                                         ▼
             ┌─────────────────────────────────────────────────────────┐
             │                2. gama.core.belief                      │
             │ Updates Bayesian prior/posterior P(e*, f* | E)          │
             └───────────────────────────┬─────────────────────────────┘
                                         │
                                         ▼
             ┌─────────────────────────────────────────────────────────┐
             │                3. gama.core.chain                       │
             │ Traces directional causal chain along the topology      │
             └───────────────────────────┬─────────────────────────────┘
                                         │
                                         ▼
             ┌─────────────────────────────────────────────────────────┐
             │                4. gama.core.llm                         │
             │ Evaluates text sensor quotes & arbitrates typing        │
             └───────────────────────────┬─────────────────────────────┘
                                         │
                                         ▼
             ┌─────────────────────────────────────────────────────────┐
             │                5. gama.core.decide                      │
             │ Applies Gate, Defer, and CalSensor decision rules       │
             └─────────────────────────────────────────────────────────┘
```
