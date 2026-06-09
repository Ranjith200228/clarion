"""Evaluation framework — consolidated metric rollup per the Phase 13 contract.

Module layout (locked):

* ``runner.py``   — load personas + run harness + return EvaluationReport
* ``metrics.py``  — pure metric computation
* ``reporter.py`` — HarnessResult + scenarios -> wire-shape EvaluationReport

The wire shape itself lives in ``clarion.schemas.evaluation``.
"""

from clarion.evaluation.metrics import (
    compute_evaluation_metrics,
    compute_metric_subset,
    load_trace_summaries,
)
from clarion.evaluation.reporter import build_report, write_report
from clarion.evaluation.runner import run_evaluation

__all__ = [
    "build_report",
    "compute_evaluation_metrics",
    "compute_metric_subset",
    "load_trace_summaries",
    "run_evaluation",
    "write_report",
]
