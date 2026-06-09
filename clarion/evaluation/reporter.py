"""Assemble + write the consolidated ``EvaluationReport``.

End-to-end entrypoints (the Phase 13 contract):

* ``build_report(customer_id, scenarios, report, traces_path=None)`` —
  pure function: HarnessReport in, EvaluationReport out. Used by the
  unit tests and by ``clarion.evaluation.runner.run_evaluation``.
* ``write_report(report, out_path)`` — JSON file writer.

Orchestration (load personas + run harness) lives in
``clarion.evaluation.runner`` so this module stays focused on
"HarnessResult + scenarios -> wire-shape EvaluationReport" and the
Phase 14 Gradio UI can re-read the report without depending on
anything in this file.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from clarion.evaluation.metrics import compute_metric_subset
from clarion.schemas import (
    EvaluationCategoryBreakdown,
    EvaluationMetrics,
    EvaluationReport,
    HarnessReport,
    Scenario,
)

log = logging.getLogger(__name__)


# ---------- top-level entrypoint ----------


def build_report(
    customer_id: str,
    scenarios: list[Scenario],
    report: HarnessReport,
    *,
    traces_path: Path | None = None,
) -> EvaluationReport:
    """Assemble the EvaluationReport from a HarnessReport + scenarios.

    Computes the overall metrics, then per-difficulty and per-intent
    rollups using the same metric function so the columns are guaranteed
    consistent.
    """
    overall = compute_metric_subset(scenarios, list(report.results), traces_path=traces_path)

    by_difficulty: dict[str, EvaluationCategoryBreakdown] = {}
    for difficulty in sorted({r.difficulty for r in report.results}):
        subset = [r for r in report.results if r.difficulty == difficulty]
        metrics = compute_metric_subset(scenarios, subset, traces_path=traces_path)
        by_difficulty[difficulty] = EvaluationCategoryBreakdown(total=len(subset), metrics=metrics)

    by_intent: dict[str, EvaluationCategoryBreakdown] = {}
    for intent in sorted({r.intent for r in report.results}):
        subset = [r for r in report.results if r.intent == intent]
        metrics = compute_metric_subset(scenarios, subset, traces_path=traces_path)
        by_intent[intent] = EvaluationCategoryBreakdown(total=len(subset), metrics=metrics)

    return EvaluationReport(
        customer_id=customer_id,
        generated_at=datetime.now(UTC),
        scenario_count=len(report.results),
        pass_rate=overall.pass_rate,
        metrics=overall,
        by_difficulty=by_difficulty,
        by_intent=by_intent,
        headline=_headline(overall),
    )


def write_report(report: EvaluationReport, out_path: Path) -> Path:
    """Persist the report as JSON. Returns the path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return out_path


# ---------- helpers ----------


def _headline(metrics: EvaluationMetrics) -> dict[str, float]:
    """Six numbers the Phase 13 dashboard top strip renders."""
    return {
        "containment_rate": metrics.containment_rate,
        "booking_accuracy": metrics.booking_accuracy,
        "hallucination_rate": (
            metrics.hallucination_rate if metrics.hallucination_rate is not None else 0.0
        ),
        "escalation_precision": metrics.escalation_precision,
        "escalation_recall": metrics.escalation_recall,
        "safety_catch_rate": metrics.safety_catch_rate,
    }


# Re-exported helper for tests that want to inspect the discrete counts.
def category_counts(report: HarnessReport) -> dict[str, Counter[str]]:
    """{difficulty / intent} -> Counter for sanity-checking a build_report."""
    return {
        "difficulty": Counter(r.difficulty for r in report.results),
        "intent": Counter(r.intent for r in report.results),
    }
