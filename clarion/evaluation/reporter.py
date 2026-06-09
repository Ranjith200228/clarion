"""Assemble + write the Phase 12 consolidated ``EvaluationReport``.

End-to-end entrypoints:

* ``build_report(customer_id, scenarios, report, traces_path=None)`` —
  pure function: HarnessReport in, EvaluationReport out. Used by the
  unit tests and by ``run_evaluation``.
* ``run_evaluation(customer_id, settings, mode='scripted')`` — full
  pipeline: load personas, run the scripted harness, build the report,
  return it. Used by the CLI.
* ``write_report(report, out_path)`` — JSON file writer.

The report file lands at ``<data_dir>/<customer_id>/evaluation_report.json``
by default; Phase 13's Streamlit dashboard reads it from there.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from clarion.config import Settings, load_customer
from clarion.evaluation.metrics import compute_metric_subset
from clarion.pipelines.structured import StructuredStore
from clarion.rag.builder import load_customer_retriever
from clarion.rag.retriever import Retriever
from clarion.schemas import (
    EvaluationCategoryBreakdown,
    EvaluationMetrics,
    EvaluationReport,
    HarnessReport,
    Scenario,
)
from clarion.simulator.harness import load_scenarios, run_scripted

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


def run_evaluation(
    customer_id: str,
    *,
    settings: Settings,
    mode: str = "scripted",
) -> EvaluationReport:
    """Load personas + structured store + retriever, run the harness,
    build the report. Pure scripted mode for now (live mode wires through
    the same way; the harness already supports it via ``run_live``)."""
    if mode != "scripted":
        raise NotImplementedError(
            f"mode={mode!r} not yet wired into run_evaluation; "
            f"the CLI exposes only scripted at the moment. Live mode "
            f"is available via clarion.simulator.harness.run_live() + "
            f"build_report() directly."
        )

    customer = load_customer(customer_id, settings=settings)
    structured = StructuredStore.for_customer(customer.customer_id, settings.data_dir)
    try:
        retriever: Retriever | None = load_customer_retriever(customer, data_dir=settings.data_dir)
    except FileNotFoundError:
        log.warning(
            "no prebuilt RAG index for %r — evaluation will run without retrieval",
            customer_id,
        )
        retriever = None

    personas_path = settings.data_dir / "personas" / f"{customer_id}.json"
    scenarios = load_scenarios(personas_path)

    harness_report = run_scripted(
        scenarios,
        customer_config=customer,
        structured=structured,
        retriever=retriever,
    )

    traces_path = settings.data_dir / customer_id / "traces.jsonl"
    return build_report(
        customer_id,
        scenarios,
        harness_report,
        traces_path=traces_path if traces_path.exists() else None,
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
