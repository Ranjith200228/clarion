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
    writeback_dir: Path | None = None,
    no_show_model_path: Path | None = None,
    no_show_eval_seed: int | None = None,
) -> EvaluationReport:
    """Assemble the EvaluationReport from a HarnessReport + scenarios.

    Computes the overall metrics, then per-difficulty and per-intent
    rollups using the same metric function so the columns are guaranteed
    consistent.
    """
    overall = compute_metric_subset(scenarios, list(report.results), traces_path=traces_path)

    # Module M1 — fold field_extraction_accuracy into the overall metrics
    # when the writer ran. None signals the module wasn't enabled.
    if writeback_dir is not None:
        from clarion.modules.pms_writeback import (
            compute_field_extraction_accuracy,
        )

        fae = compute_field_extraction_accuracy(
            scenarios, report, writeback_dir=writeback_dir
        )
        if fae is not None:
            overall = overall.model_copy(
                update={"field_extraction_accuracy": fae.accuracy}
            )

    # Module M3 — fold no_show_roc_auc + no_show_top_decile_lift when the
    # trained model artifact exists. Held-out eval uses a seed offset
    # from the training seed so it's a real out-of-fold measurement.
    if no_show_model_path is not None:
        from clarion.modules.no_show_prediction import compute_no_show_metrics

        seed = no_show_eval_seed if no_show_eval_seed is not None else 4242
        ns_result = compute_no_show_metrics(no_show_model_path, seed=seed)
        if ns_result is not None:
            overall = overall.model_copy(
                update={
                    "no_show_roc_auc": ns_result.roc_auc,
                    "no_show_top_decile_lift": ns_result.top_decile_lift,
                }
            )

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
        outcome_distribution=_outcome_distribution(report),
        escalation_reason_frequency=_escalation_reason_frequency(report),
        escalated_scenario_ids=_escalated_scenario_ids(report),
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


# ---------- Phase 14 UI-feed pre-aggregates ----------


def _outcome_distribution(report: HarnessReport) -> dict[str, int]:
    """Count of each actual_outcome label across all results.

    The Phase 14 Quality tab renders this as the outcome breakdown bar.
    Sorted so JSON output is deterministic between runs.
    """
    counts = Counter(r.actual_outcome for r in report.results)
    return dict(sorted(counts.items()))


def _escalation_reason_frequency(report: HarnessReport) -> dict[str, int]:
    """Frequency of each escalation reason label across results that
    actually fired one. Pulled from ``result.escalation['reasons']``.

    A reason label looks like ``"low_confidence=0.80"`` or
    ``"frustration=0.62"``; the histogram strips the score suffix so the
    bins are stable across runs. Sorted for deterministic output.
    """
    counts: Counter[str] = Counter()
    for r in report.results:
        escalation = r.escalation
        if not isinstance(escalation, dict):
            continue
        for label in escalation.get("reasons") or []:
            # "low_confidence=0.80" -> "low_confidence"
            key = str(label).split("=", 1)[0].strip()
            if key:
                counts[key] += 1
    return dict(sorted(counts.items()))


def _escalated_scenario_ids(report: HarnessReport) -> list[str]:
    """Ordered scenario_ids whose escalation.should_escalate is True."""
    out: list[str] = []
    for r in report.results:
        escalation = r.escalation
        if isinstance(escalation, dict) and bool(escalation.get("should_escalate")):
            out.append(r.scenario_id)
    return out
