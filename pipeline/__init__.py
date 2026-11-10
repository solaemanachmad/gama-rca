"""pipeline
========
Multi-modal telemetry extraction and statistical calibration engines.

Four Functional Subpackages:
  1. pipeline.metric: Infrastructure and network propagation metrics
  2. pipeline.log: Log pre-processing, template mining, and text probes
  3. pipeline.trace: Distributed trace span aggregations and deep origin analysis
  4. pipeline.calibration: Statistical calibration, shift scoring, and fault fingerprints
"""

import sys
from pipeline.metric import infra, propagation
from pipeline.log import preprocess, probes, templates
from pipeline.trace import evidence as trace_evidence, structure as trace_structure
from pipeline.calibration import belief, calibrated, fault, fingerprint, typing

# Backward-compatibility module aliases for legacy flat imports
sys.modules.setdefault("pipeline.infra_evidence", infra)
sys.modules.setdefault("pipeline.propagation_evidence", propagation)
sys.modules.setdefault("pipeline.log_preprocess", preprocess)
sys.modules.setdefault("pipeline.log_templates", templates)
sys.modules.setdefault("pipeline.semantic_probes", probes)
sys.modules.setdefault("pipeline.trace_evidence", trace_evidence)
sys.modules.setdefault("pipeline.trace_structure", trace_structure)
sys.modules.setdefault("pipeline.fault", fault)
sys.modules.setdefault("pipeline.fault_calibrated", calibrated)
sys.modules.setdefault("pipeline.fault_fingerprint", fingerprint)
sys.modules.setdefault("pipeline.belief_diagnosis", belief)
sys.modules.setdefault("pipeline.llm_typing", typing)

__all__ = [
    "metric",
    "log",
    "trace",
    "calibration",
    "infra",
    "propagation",
    "preprocess",
    "templates",
    "probes",
    "trace_evidence",
    "trace_structure",
    "fault",
    "calibrated",
    "fingerprint",
    "belief",
    "typing",
]
