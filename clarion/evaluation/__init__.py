"""Phase 12 evaluation framework — consolidated metric rollup."""

from clarion.evaluation.metrics import (
    compute_evaluation_metrics,
    compute_metric_subset,
    load_trace_summaries,
)
from clarion.evaluation.report import build_report, run_evaluation, write_report

__all__ = [
    "build_report",
    "compute_evaluation_metrics",
    "compute_metric_subset",
    "load_trace_summaries",
    "run_evaluation",
    "write_report",
]
