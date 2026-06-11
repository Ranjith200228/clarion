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
    tokens_per_call: float = Field(ge=0.0, default=0.0)
    latency_ms: LatencyStats | None = None

    # Module M1 (PMS Writeback) — fraction of extractor-produced fields
    # that match scenario ground truth. None when the module is
    # disabled for the customer.
    field_extraction_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)

    # Module M3 (No-Show Prediction) — held-out ROC-AUC and top-decile
    # lift from scoring a freshly generated synthetic test set through
    # the persisted booster. None when the module is disabled or no
    # model artifact exists yet for the customer.
    no_show_roc_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    no_show_top_decile_lift: float | None = Field(default=None, ge=0.0)


class EvaluationCategoryBreakdown(BaseModel):
    """Per-category metrics ready for the by_difficulty / by_intent dicts."""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0)
    metrics: EvaluationMetrics


class EvaluationReport(BaseModel):
    """One customer's full evaluation rollup.

    Written to ``<data_dir>/<customer_id>/report_<customer_id>.json``;
    re-read by the Phase 14 Gradio UI.

    LOCKED CONTRACT
    ---------------
    Per the Phase 13 spec: ``schema_version`` keys this file. Any
    breaking change (renamed / removed field, narrower domain on an
    existing field, changed semantics) must bump ``REPORT_SCHEMA_VERSION``
    below. The UI keys on ``schema_version`` so it can reject reports
    it doesn't know how to read.

    Additive changes (new optional field, looser bounds) keep the
    version stable. Removal or rename = bump.
    """

    model_config = ConfigDict(extra="forbid")

    # Schema lock — the version of the EvaluationReport wire contract
    # this object conforms to. Bump on any breaking change.
    schema_version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")

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

    # Pre-aggregated UI feeds (Phase 14 additive, schema stays at 1.0.0).
    # These are NOT metrics — they are denormalized counts the Quality
    # and Escalation tabs render directly. Phase 14 spec rule:
    # "No metric computation inside UI."

    # outcome_distribution: actual_outcome -> count across all scenarios.
    # Renders as the Quality tab's outcome breakdown bar.
    outcome_distribution: dict[str, int] = Field(default_factory=dict)

    # escalation_reason_frequency: reason_label -> count across all
    # results where the scorer fired at least one reason. Renders as
    # the Escalations tab's reason histogram.
    escalation_reason_frequency: dict[str, int] = Field(default_factory=dict)

    # escalated_scenario_ids: ordered list of scenario_ids whose
    # escalation.should_escalate was True. Renders as the Escalations
    # tab's "Escalated Calls" list (count + drilldown ids).
    escalated_scenario_ids: list[str] = Field(default_factory=list)


# Public constant the runner stamps on every EvaluationReport it builds.
# Exported here so the Phase 14 UI can compare without importing the
# entire schema module.
REPORT_SCHEMA_VERSION = "1.0.0"


# ---------- Trace sidecar (Phase 13 spec: trace_<customer>.json) ----------


class TraceEntry(BaseModel):
    """One scenario's worth of trace data, flattened for the UI.

    The Trace Explorer tab in the Phase 14 Gradio app reads a list of
    these per customer. Every field is denormalized so the UI does
    zero math: it just renders.
    """

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    customer_id: str
    trace_id: str
    difficulty: str
    intent: str

    # What the agent did.
    agent_replies: list[str]
    tools_called: list[str]
    actual_outcome: str
    passed: bool

    # Sentinel decisions.
    escalation_score: float | None = Field(default=None, ge=0.0, le=1.0)
    escalation_reasons: list[str] = Field(default_factory=list)
    judge_hallucination: float | None = Field(default=None, ge=0.0, le=1.0)
    judge_booking_correct: float | None = Field(default=None, ge=0.0, le=1.0)
    judge_violations: list[str] = Field(default_factory=list)

    # Cost / latency / token totals across the scenario's turns.
    duration_ms: float | None = Field(default=None, ge=0.0)
    cost_usd: float | None = Field(default=None, ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    step_count: int = Field(default=0, ge=0)


class TraceReport(BaseModel):
    """The sidecar JSON written next to ``report_<customer>.json``.

    Phase 14 Trace Explorer tab reads this and renders each TraceEntry
    as one row in a table. No business logic; pure render.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    customer_id: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    generated_at: datetime
    entries: list[TraceEntry] = Field(default_factory=list)


TRACE_SCHEMA_VERSION = "1.0.0"
