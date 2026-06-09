"""Wire shape for the Phase 12 consolidated evaluation report.

``EvaluationReport`` is what gets written to
``<data_dir>/<customer_id>/evaluation_report.json`` per harness run.
Phase 13's Streamlit dashboard reads this file directly — every number
it renders is in here, no recomputation.

Three layers:

* ``LatencyStats``    — pure numeric summary; null when no traces.
* ``EvaluationMetrics`` — the headline numbers from the spec, per scope.
* ``EvaluationCategoryBreakdown`` — wraps metrics with the category total
  so by_difficulty / by_intent renderers don't re-count.
* ``EvaluationReport`` — top level: overall metrics + per-category
  rollups + a small ``headline`` dict the dashboard top strip reads.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class LatencyStats(BaseModel):
    """Per-turn latency summary in milliseconds.

    ``count`` is the number of ``agent.chat`` spans that contributed to
    the avg / p50 / p95. None when no traces were available for the run.
    """

    model_config = ConfigDict(extra="forbid")

    avg: float = Field(ge=0.0)
    p50: float = Field(ge=0.0)
    p95: float = Field(ge=0.0)
    count: int = Field(ge=0)


class EvaluationMetrics(BaseModel):
    """All Phase 12 spec metrics for one scope (overall or a category)."""

    model_config = ConfigDict(extra="forbid")

    scenario_count: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)

    # Containment — % of scenarios resolved without a human handoff.
    containment_rate: float = Field(ge=0.0, le=1.0)

    # Booking accuracy — for scenarios whose ground_truth.expected_outcome
    # is "booked", how many actually booked correctly.
    booking_accuracy: float = Field(ge=0.0, le=1.0)
    booking_total: int = Field(ge=0)
    booking_correct: int = Field(ge=0)

    # Hallucination — average judge.hallucination across scenarios that
    # had a judge attached. None when no judge ran on this scope.
    hallucination_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    hallucination_with_judge: int = Field(ge=0, default=0)

    # Escalation — from Phase 11's stats_from_run. Pulled in here too so
    # this single object is the wire contract for the dashboard.
    escalation_precision: float = Field(ge=0.0, le=1.0)
    escalation_recall: float = Field(ge=0.0, le=1.0)
    escalation_f1: float = Field(ge=0.0, le=1.0)
    escalation_accuracy: float = Field(ge=0.0, le=1.0)

    # Safety — recall on emergency + clinical_advice intents.
    safety_catch_rate: float = Field(ge=0.0, le=1.0)
    safety_total: int = Field(ge=0)
    safety_caught: int = Field(ge=0)

    # Performance.
    avg_turns_to_resolution: float = Field(ge=0.0)
    cost_per_request_usd: float = Field(ge=0.0)
    latency_ms: LatencyStats | None = None


class EvaluationCategoryBreakdown(BaseModel):
    """Per-category metrics ready for the by_difficulty / by_intent dicts."""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0)
    metrics: EvaluationMetrics


class EvaluationReport(BaseModel):
    """One customer's full evaluation rollup.

    Written to ``<data_dir>/<customer_id>/evaluation_report.json``;
    re-read by the Phase 13 dashboard and the Phase 15 release notes.
    """

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    generated_at: datetime
    scenario_count: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)

    # Overall metrics.
    metrics: EvaluationMetrics

    # Per-category breakdowns. Keys are scenario difficulty / intent.
    by_difficulty: dict[str, EvaluationCategoryBreakdown] = Field(default_factory=dict)
    by_intent: dict[str, EvaluationCategoryBreakdown] = Field(default_factory=dict)

    # Tiny dict the dashboard top strip renders. Six numbers, by name.
    headline: dict[str, float] = Field(default_factory=dict)
