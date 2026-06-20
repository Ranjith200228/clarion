"""Aggregation façade over existing on-disk artifacts.

``gradio_app.data`` reads the locked-schema JSON files **per
customer**. This module is the next layer up: it reads ALL tenants,
computes typed rollups, and hands them to the v2 views (Mission
Control, Cost & SLO).

Design rules:

- **Read-only.** Every function here consumes existing artifacts
  written by the engine. No new POST endpoints, no business logic.
- **Typed.** Each returned shape is a frozen dataclass with primitive
  fields the views can plug straight into ``components.*`` builders.
- **Cheap to recompute.** Mission Control rebuilds on every customer
  switch + on every page load; the rollups must avoid heavy work.
  Each tenant's report+trace round-trips to disk once and is cached
  for the lifetime of the call.
- **Empty-state tolerant.** When a customer's artifacts are missing
  (fresh deploy, first-run, never-eval'd customer), the rollup
  surfaces a clearly-empty snapshot instead of raising.

The schema lock from Phase 13 still rules: every read here goes
through :mod:`gradio_app.data`, which validates ``schema_version``
on entry. If the engine bumps the locked schema, this layer breaks
loudly — that's the design.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from clarion.schemas import (
    EscalationWeights,
    EvaluationReport,
    TraceEntry,
)

from gradio_app import data as data_loader

log = logging.getLogger(__name__)

# Public type aliases — views import these for callsite typing.
HealthStatus = Literal["healthy", "warning", "critical", "unknown"]


# ---------- typed rollups ----------


@dataclass(frozen=True)
class TenantSnapshot:
    """One customer's headline view, ready to render.

    Built from one ``EvaluationReport`` + one ``TraceReport``. Every
    field is either a primitive or one of our typed aliases — no
    Pydantic instances leak past this boundary.
    """

    customer_id: str
    display_name: str
    has_data: bool
    health: HealthStatus
    pass_rate: float
    containment_rate: float
    booking_accuracy: float
    safety_catch_rate: float
    hallucination_rate: float
    escalation_recall: float
    escalation_precision: float
    avg_turns_to_resolution: float
    cost_per_request_usd: float
    scenario_count: int
    last_run_at: datetime | None
    last_run_relative: str  # "12m ago", "—" if no data
    # For the per-tenant comparison column on Mission Control.
    headline_score: float  # 0..1 composite — used to sort tenants by health


@dataclass(frozen=True)
class GlobalKPIs:
    """The 8 numbers in the Mission Control top strip.

    Aggregated across every tenant with data. Numbers that don't make
    sense to aggregate (escalation precision, hallucination rate)
    use a weighted mean by scenario_count; rates use sum-by-count.
    """

    total_tenants: int
    total_scenarios: int
    pass_rate: float
    containment_rate: float
    safety_catch_rate: float
    hallucination_rate: float
    avg_turns: float
    cost_per_request_usd: float
    composite_trust: float  # 1 - escalation_score-equivalent
    total_emergencies: int  # count of escalated entries across tenants


@dataclass(frozen=True)
class EscalationItem:
    """One row in the Recent Escalations stream."""

    tenant: str
    scenario_id: str
    severity: HealthStatus
    summary: str  # "low_confidence + frustration on conv_x123…"
    detected_at: str  # display-only; "12m ago" / "in this run"
    sort_key: float  # epoch-ish; used to merge across tenants then trim


@dataclass(frozen=True)
class EmergencyItem:
    """One row in the Recent Emergencies stream."""

    tenant: str
    scenario_id: str
    summary: str  # the first user message that tripped the guardrail
    detected_at: str
    sort_key: float


@dataclass(frozen=True)
class CostBreakdown:
    """One tenant's cost rollup for Cost & SLO."""

    tenant: str
    total_cost_usd: float
    avg_cost_per_request_usd: float
    total_input_tokens: int
    total_output_tokens: int
    scenarios: int


@dataclass(frozen=True)
class LatencyBreakdown:
    """One tenant's latency rollup for Cost & SLO."""

    tenant: str
    avg_ms: float
    p50_ms: float
    p95_ms: float
    sample_count: int


@dataclass(frozen=True)
class GlobalCostSLO:
    """Cross-tenant rollup for the Cost & SLO view."""

    per_tenant_cost: list[CostBreakdown]
    per_tenant_latency: list[LatencyBreakdown]
    total_cost_usd: float
    monthly_projection_usd: float  # naive: last-7d cost / 7 * 30
    global_p50_ms: float
    global_p95_ms: float


@dataclass(frozen=True)
class SignalContribution:
    """One row in the Sentinel Operations Center signal breakdown.

    Built by parsing per-entry ``escalation_reasons`` strings (the
    scorer emits them as ``"name=0.42"``) and folding them into
    means + counts. The ``weight`` carries the configured
    :class:`EscalationWeights` for the matching signal so the view
    can show "raw signal x weight = contribution to composite."
    """

    name: str
    raw_mean: float
    weight: float
    contribution: float  # raw_mean * weight
    fire_count: int      # entries that registered a non-zero value


@dataclass(frozen=True)
class JudgeAgreement:
    """LLM-as-judge agreement rollup (per tenant)."""

    sampled_turns: int                # entries that had a judge verdict at all
    no_hallucination_pct: float       # fraction with judge_hallucination ~ 0
    booking_correct_mean: float       # mean of judge_booking_correct (None ignored)
    violations_total: int             # sum of len(judge_violations) across entries


@dataclass(frozen=True)
class AuditTailItem:
    """One row in the audit-log tail panel.

    The audit log is the canonical PHI-redaction surface, so these
    strings are safe to render directly — the writer redacted on
    write. We do not re-redact here; doing so would mask
    redactor bugs.
    """

    ts: str
    kind: str           # event class label, mapped from AuditTurn.payload
    summary: str        # already-redacted one-line description


@dataclass(frozen=True)
class ProviderUtilization:
    """Per-provider availability cell for the heat map row.

    ``utilization`` is in [0, 1] where 0 = no slots booked, 1 =
    every slot booked. ``status`` is the colour band the view
    paints the cell with.
    """

    provider_id: str
    provider_name: str
    slots_total: int
    slots_booked: int
    utilization: float
    daily: list[float]              # per-day utilization for the 14-day grid
    status: Literal["healthy", "warning", "critical", "unknown"]


@dataclass(frozen=True)
class NoShowRiskBucket:
    """One bar in the no-show histogram."""

    band: Literal["low", "medium", "high"]
    count: int
    fraction: float


@dataclass(frozen=True)
class PmsTaskRow:
    """One row in the PMS task queue panel."""

    task_id: str
    subject: str
    priority: Literal["normal", "urgent"]
    assignee_group: str
    patient_id: str | None
    created_at: str                  # display-only, "yesterday" etc.


@dataclass(frozen=True)
class EligibilitySummary:
    """One slice of the eligibility coverage donut."""

    status: str                      # active / pending / denied / unknown
    count: int
    fraction: float


@dataclass(frozen=True)
class HealthcareOpsSnapshot:
    """Per-tenant rollup for the Healthcare Operations view.

    Mixes four data sources:
      * structured.sqlite3 -> providers + availability + eligibility
      * M3 NoShowPrediction.predictions.jsonl when present, otherwise
        a synthetic distribution computed from M3 dataset
      * M1 pms_writeback/<conv>/task.json artifacts
    The view labels each section so a reader knows which is which.
    """

    tenant: str
    has_structured: bool             # whether SQLite store exists
    providers: list[ProviderUtilization]
    days_in_grid: int                # always 14 (label shorthand)
    avg_utilization: float
    no_show_buckets: list[NoShowRiskBucket]
    no_show_total: int
    no_show_mean_risk: float
    pms_tasks: list[PmsTaskRow]
    pms_open_count: int
    eligibility: list[EligibilitySummary]
    eligibility_total: int


@dataclass(frozen=True)
class EmotionTotal:
    """One row in the emotion distribution chart.

    Voice Intelligence categorizes each turn into one of six
    emotions based on its escalation reasons + outcome. The mapping
    is heuristic (regex over the locked TraceEntry fields) — we don't
    have a sentiment model in production, and inventing one for the
    dashboard would mask what the existing engine actually sees.
    """

    emotion: str
    count: int
    fraction: float          # count / total_turns (0..1)


@dataclass(frozen=True)
class FrustrationPoint:
    """One sample in the frustration-over-turns line chart.

    ``score`` is the entry's escalation_score, used as a proxy for
    composite frustration. The locked schema doesn't separately
    record per-turn frustration — the EscalationScorer fuses
    frustration with the other 4 signals at composite time — so we
    visualize what we have rather than fabricating a richer field.
    """

    turn_index: int
    scenario_id: str
    score: float
    escalated: bool


@dataclass(frozen=True)
class VoicePipelineStage:
    """One stage in the voice round-trip target diagram."""

    name: str                # "STT", "Agent", "TTS"
    target_ms: int
    description: str         # one-line explainer


@dataclass(frozen=True)
class VoiceIntelligenceSnapshot:
    """Per-tenant rollup for the Voice Intelligence view.

    The dashboard intentionally mixes two data sources:
      * Chat trace (this rolls up emotion / frustration / escalation
        rate the engine actually saw across the scenario corpus).
      * Static voice pipeline targets (the budget the M5 voice
        layer is built against — STT / agent / TTS in series).
    The view labels each section so a reader knows which is which.
    """

    tenant: str
    has_data: bool
    total_turns: int
    emotions: list[EmotionTotal]
    frustration_trace: list[FrustrationPoint]
    mean_frustration: float
    escalation_rate: float           # historical fraction (0..1)
    predicted_escalation_rate: float  # smoothed Bayesian estimate
    voice_pipeline: list[VoicePipelineStage]
    sample_transcript: list[tuple[str, float]]  # (token, confidence) pairs


@dataclass(frozen=True)
class FlowNode:
    """One node in the agent-flow diagram.

    The view renders these with :func:`components.agent_node` plus
    a per-node detail line below.
    """

    name: str            # display label ("ROUTER", "BOOKING")
    state: Literal["idle", "active", "done", "escalated"]
    ms: int | None
    cost_usd: float | None
    detail: str          # one-line subtitle the view shows under the node


# Canonical position keys in the agent-flow diagram. Listed in the
# order the diagram lays them out left-to-right.
FlowPosition = Literal[
    "patient",
    "router",
    "specialist",
    "tools",
    "sentinel",
    "response",
]


@dataclass(frozen=True)
class AgentFlowSnapshot:
    """One turn's trip through the multi-agent graph.

    Built from a single :class:`TraceEntry`. Carries the activation
    state for every node + the list of specialists not chosen (so
    the view can grey them out) + the tools that fired.
    """

    tenant: str
    scenario_id: str
    intent: str
    user_message: str
    nodes: dict[str, FlowNode]      # keyed by FlowPosition
    chosen_specialist: str          # display name, e.g. "Booking"
    other_specialists: list[str]    # 4 unselected specialists
    tools_called: list[str]
    escalation_score: float
    escalation_reasons: list[str]
    judge_violations: list[str]
    final_outcome: str
    has_data: bool
    available_turns: list[str]      # scenario ids the view's picker offers


@dataclass(frozen=True)
class SentinelOpsSnapshot:
    """Per-tenant Sentinel Operations Center rollup.

    Built from one tenant's TraceReport + AuditLog. The view
    consumes this directly — no per-row business logic past this
    boundary.
    """

    tenant: str
    has_data: bool
    sample_count: int                  # turns the snapshot was built from
    mean_escalation_score: float       # 0 = always safe, 1 = always escalate
    trust_score: float                 # 1 - mean_escalation_score
    decision_threshold: float          # the configured escalate-at boundary
    signals: list[SignalContribution]  # 5 rows (one per EscalationSignals field)
    judge: JudgeAgreement
    phi_redactions_total: int          # PHI redactions seen across this tenant's audit log
    audit_tail: list[AuditTailItem]    # last 10 events, newest first
    emergencies_caught: int            # entries with emergency outcome OR reason
    escalation_precision: float        # carried through from EvaluationReport
    escalation_recall: float
    escalation_f1: float


# ---------- single-tenant builder ----------


def build_tenant_snapshot(
    customer_id: str,
    *,
    data_dir: Path | None = None,
) -> TenantSnapshot:
    """Read one customer's report+trace and roll into a TenantSnapshot.

    Returns a fully-populated ``TenantSnapshot(has_data=False)`` when
    artifacts are missing — the view renders these tenants as a
    greyed-out "no data" row instead of crashing.
    """
    display_name = _humanize(customer_id)
    try:
        artifacts = data_loader.load_artifacts(customer_id, data_dir)
    except (FileNotFoundError, data_loader.SchemaVersionMismatchError) as exc:
        log.info("tenant %r has no usable artifacts: %s", customer_id, exc)
        return TenantSnapshot(
            customer_id=customer_id,
            display_name=display_name,
            has_data=False,
            health="unknown",
            pass_rate=0.0,
            containment_rate=0.0,
            booking_accuracy=0.0,
            safety_catch_rate=0.0,
            hallucination_rate=0.0,
            escalation_recall=0.0,
            escalation_precision=0.0,
            avg_turns_to_resolution=0.0,
            cost_per_request_usd=0.0,
            scenario_count=0,
            last_run_at=None,
            last_run_relative="—",
            headline_score=0.0,
        )

    report: EvaluationReport = artifacts.report
    metrics = report.metrics
    headline_score = _composite_headline_score(report)
    last_run_at = report.generated_at
    return TenantSnapshot(
        customer_id=customer_id,
        display_name=display_name,
        has_data=True,
        health=_health_from_headline(headline_score),
        pass_rate=metrics.pass_rate,
        containment_rate=metrics.containment_rate,
        booking_accuracy=metrics.booking_accuracy,
        safety_catch_rate=metrics.safety_catch_rate,
        hallucination_rate=metrics.hallucination_rate or 0.0,
        escalation_recall=metrics.escalation_recall,
        escalation_precision=metrics.escalation_precision,
        avg_turns_to_resolution=metrics.avg_turns_to_resolution,
        cost_per_request_usd=metrics.cost_per_request_usd,
        scenario_count=report.scenario_count,
        last_run_at=last_run_at,
        last_run_relative=_relative_time(last_run_at),
        headline_score=headline_score,
    )


# ---------- multi-tenant aggregations ----------


def all_tenant_snapshots(data_dir: Path | None = None) -> list[TenantSnapshot]:
    """Build a TenantSnapshot for every known customer."""
    return [
        build_tenant_snapshot(cid, data_dir=data_dir)
        for cid in data_loader.KNOWN_CUSTOMERS
    ]


def build_global_kpis(
    snapshots: list[TenantSnapshot],
    *,
    data_dir: Path | None = None,
) -> GlobalKPIs:
    """Aggregate snapshots into the 8-tile top-strip view.

    Rates are weighted by scenario_count so a 100-scenario tenant
    counts 4x a 25-scenario tenant. Tenants with ``has_data=False``
    are excluded from the aggregation.

    ``data_dir`` threads to the emergency-count side trip — the
    snapshot doesn't carry per-entry trace metadata, so we re-read
    the trace JSONL to tally emergency outcomes. Tests pass a
    ``tmp_path`` here; production paths fall back to the env's
    default (the same dir every other reader uses).
    """
    live = [s for s in snapshots if s.has_data]
    if not live:
        return GlobalKPIs(
            total_tenants=0,
            total_scenarios=0,
            pass_rate=0.0,
            containment_rate=0.0,
            safety_catch_rate=0.0,
            hallucination_rate=0.0,
            avg_turns=0.0,
            cost_per_request_usd=0.0,
            composite_trust=0.0,
            total_emergencies=0,
        )
    total_scen = sum(s.scenario_count for s in live)

    def w(attr: str) -> float:
        if total_scen == 0:
            return 0.0
        return sum(getattr(s, attr) * s.scenario_count for s in live) / total_scen

    # Composite trust = mean of containment + safety + (1 - hallucination).
    # All three are healthy when high; subtracting hallucination keeps
    # the metric monotone with "more trust = higher".
    composite = (
        w("containment_rate") + w("safety_catch_rate") + (1.0 - w("hallucination_rate"))
    ) / 3.0

    return GlobalKPIs(
        total_tenants=len(live),
        total_scenarios=total_scen,
        pass_rate=w("pass_rate"),
        containment_rate=w("containment_rate"),
        safety_catch_rate=w("safety_catch_rate"),
        hallucination_rate=w("hallucination_rate"),
        avg_turns=w("avg_turns_to_resolution"),
        cost_per_request_usd=w("cost_per_request_usd"),
        composite_trust=max(0.0, min(1.0, composite)),
        total_emergencies=_count_emergencies(live, data_dir=data_dir),
    )


def recent_escalations(
    snapshots: list[TenantSnapshot],
    *,
    data_dir: Path | None = None,
    limit: int = 10,
) -> list[EscalationItem]:
    """Merge every tenant's escalated scenarios into one chrono stream.

    "Chrono" here is a soft order. Per-entry timestamps don't live in
    the locked TraceReport schema, so we use a synthetic key:
    last_run_at as epoch seconds + a tiny per-scenario tiebreaker so
    items within one run preserve their report-order.
    """
    items: list[EscalationItem] = []
    for snap in snapshots:
        if not snap.has_data:
            continue
        try:
            trace = data_loader.load_trace_report(snap.customer_id, data_dir)
        except (FileNotFoundError, data_loader.SchemaVersionMismatchError):
            continue
        base_ts = (
            snap.last_run_at.timestamp()
            if snap.last_run_at is not None
            else 0.0
        )
        # Walk entries in reverse so the tiebreaker bumps newer items
        # higher within the same run.
        for offset, entry in enumerate(reversed(trace.entries)):
            if not entry.escalation_reasons:
                continue
            items.append(
                EscalationItem(
                    tenant=snap.display_name,
                    scenario_id=entry.scenario_id,
                    severity=_escalation_severity(entry),
                    summary=_format_escalation_summary(entry),
                    detected_at=snap.last_run_relative,
                    sort_key=base_ts + offset * 1e-6,
                )
            )
    items.sort(key=lambda i: i.sort_key, reverse=True)
    return items[:limit]


def recent_emergencies(
    snapshots: list[TenantSnapshot],
    *,
    data_dir: Path | None = None,
    limit: int = 10,
) -> list[EmergencyItem]:
    """Entries the guardrail short-circuited with emergency handoff.

    Detected from the trace: an emergency turn has actual_outcome ==
    'escalated_emergency' OR an emergency-intent classification.
    We pull from the trace so a tenant without an audit log still
    contributes.
    """
    items: list[EmergencyItem] = []
    for snap in snapshots:
        if not snap.has_data:
            continue
        try:
            trace = data_loader.load_trace_report(snap.customer_id, data_dir)
        except (FileNotFoundError, data_loader.SchemaVersionMismatchError):
            continue
        base_ts = (
            snap.last_run_at.timestamp()
            if snap.last_run_at is not None
            else 0.0
        )
        for offset, entry in enumerate(reversed(trace.entries)):
            if not _is_emergency(entry):
                continue
            items.append(
                EmergencyItem(
                    tenant=snap.display_name,
                    scenario_id=entry.scenario_id,
                    summary=_format_emergency_summary(entry),
                    detected_at=snap.last_run_relative,
                    sort_key=base_ts + offset * 1e-6,
                )
            )
    items.sort(key=lambda i: i.sort_key, reverse=True)
    return items[:limit]


# ---------- Healthcare Operations ----------


# Number of days the provider heat map paints. Fixed because the
# view renders a static 14-column grid; if we ever change this,
# the view + CSS need to know.
_HEALTHCARE_GRID_DAYS = 14


def build_healthcare_ops(
    customer_id: str,
    *,
    data_dir: Path | None = None,
) -> HealthcareOpsSnapshot:
    """Build the per-tenant Healthcare Operations rollup.

    Reads from up to three on-disk surfaces (any of which may be
    absent on a fresh deploy — the empty-state path produces a
    well-formed snapshot):

      * SQLite at ``<data_dir>/<customer>/structured.sqlite3`` for
        providers + availability + eligibility.
      * M3 predictions at
        ``<data_dir>/<customer>/no_show_prediction/predictions.jsonl``.
        When missing, we fall back to synthesising from the M3
        dataset generator so the panel still renders.
      * M1 PMS writeback at
        ``<data_dir>/<customer>/pms_writeback/<conv>/task.json``.
    """
    base = data_dir if data_dir is not None else data_loader.DEFAULT_DATA_DIR
    tenant = _humanize(customer_id)

    providers, avg_util = _read_provider_heatmap(customer_id, base)
    no_show_buckets, no_show_total, no_show_mean = _read_no_show_distribution(
        customer_id, base
    )
    pms_tasks = _read_pms_tasks(customer_id, base)
    eligibility, eligibility_total = _read_eligibility(customer_id, base)

    return HealthcareOpsSnapshot(
        tenant=tenant,
        has_structured=any(p.slots_total > 0 for p in providers),
        providers=providers,
        days_in_grid=_HEALTHCARE_GRID_DAYS,
        avg_utilization=round(avg_util, 4),
        no_show_buckets=no_show_buckets,
        no_show_total=no_show_total,
        no_show_mean_risk=round(no_show_mean, 4),
        pms_tasks=pms_tasks,
        pms_open_count=sum(1 for t in pms_tasks if t.priority in {"normal", "urgent"}),
        eligibility=eligibility,
        eligibility_total=eligibility_total,
    )


# ---------- Healthcare Ops internals ----------


def _read_provider_heatmap(
    customer_id: str, base: Path
) -> tuple[list[ProviderUtilization], float]:
    """Read providers + availability from SQLite and aggregate.

    For each provider x day, we compute is_booked / total slots
    over the next 14 days starting today (UTC). Provider rows
    without any slot in that window still appear (zeroed) so the
    grid stays a rectangle.

    Returns ``([], 0.0)`` when the SQLite file is missing — the
    view interprets that as the empty-state.
    """
    sqlite_path = base / customer_id / "structured.sqlite3"
    if not sqlite_path.is_file():
        return [], 0.0

    import sqlite3
    from datetime import date, timedelta

    today = date.today()
    days = [today + timedelta(days=i) for i in range(_HEALTHCARE_GRID_DAYS)]
    days_str = [d.isoformat() for d in days]

    rows: list[ProviderUtilization] = []
    overall_total = 0
    overall_booked = 0
    try:
        with sqlite3.connect(sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            providers = cur.execute(
                "SELECT provider_id, full_name FROM providers ORDER BY full_name"
            ).fetchall()
            for prov in providers:
                pid = str(prov["provider_id"])
                pname = str(prov["full_name"])
                # Single query to pull this provider's slots in the
                # 14-day window.
                slots = cur.execute(
                    """
                    SELECT slot_date, is_booked
                    FROM availability
                    WHERE provider_id = ?
                      AND slot_date BETWEEN ? AND ?
                    """,
                    (pid, days_str[0], days_str[-1]),
                ).fetchall()
                per_day_total: dict[str, int] = {d: 0 for d in days_str}
                per_day_booked: dict[str, int] = {d: 0 for d in days_str}
                for s in slots:
                    d = str(s["slot_date"])
                    if d in per_day_total:
                        per_day_total[d] += 1
                        if int(s["is_booked"] or 0):
                            per_day_booked[d] += 1
                daily = [
                    per_day_booked[d] / per_day_total[d] if per_day_total[d] else 0.0
                    for d in days_str
                ]
                slots_total = sum(per_day_total.values())
                slots_booked = sum(per_day_booked.values())
                util = slots_booked / slots_total if slots_total else 0.0
                overall_total += slots_total
                overall_booked += slots_booked
                rows.append(
                    ProviderUtilization(
                        provider_id=pid,
                        provider_name=pname,
                        slots_total=slots_total,
                        slots_booked=slots_booked,
                        utilization=round(util, 4),
                        daily=[round(x, 4) for x in daily],
                        status=_utilization_status(util, slots_total),
                    )
                )
    except sqlite3.DatabaseError as exc:
        log.warning("provider heatmap: sqlite error for %r: %s", customer_id, exc)
        return [], 0.0

    avg = overall_booked / overall_total if overall_total else 0.0
    return rows, avg


def _utilization_status(
    util: float, slots_total: int
) -> Literal["healthy", "warning", "critical", "unknown"]:
    """Map a 0..1 utilization into a colour band.

    Both "no slots at all" and "no booked slots" land in unknown
    so the heat map doesn't paint empty rows as healthy.
    """
    if slots_total == 0:
        return "unknown"
    if util >= 0.85:
        return "critical"   # over-booked, can't take walk-ins
    if util >= 0.60:
        return "warning"
    if util >= 0.30:
        return "healthy"
    return "unknown"        # under-utilised


def _read_no_show_distribution(
    customer_id: str, base: Path
) -> tuple[list[NoShowRiskBucket], int, float]:
    """Read or synthesise a no-show risk distribution.

    Priority:
      1. M3 predictions file on disk.
      2. Synthetic distribution from the M3 dataset generator
         (using the customer_id as the seed so each tenant gets a
         distinct shape).

    Returns three buckets (low / medium / high) plus the sample
    size and mean p_no_show.
    """
    counts = {"low": 0, "medium": 0, "high": 0}
    p_values: list[float] = []

    predictions_path = (
        base / customer_id / "no_show_prediction" / "predictions.jsonl"
    )
    if predictions_path.is_file():
        try:
            with predictions_path.open("r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    p = record.get("p_no_show")
                    band = record.get("risk_band")
                    if isinstance(p, int | float) and isinstance(band, str):
                        if band in counts:
                            counts[band] += 1
                        p_values.append(float(p))
        except OSError:
            pass

    if not p_values:
        # Synthesise — same generator the M3 module uses for tests +
        # training. Tenant-stable seed via a deterministic hash.
        try:
            from clarion.modules.no_show_prediction import generate_dataset
        except ImportError:  # pragma: no cover — module is core
            return [], 0, 0.0
        seed = abs(hash(customer_id)) % 2**31
        dataset = generate_dataset(seed=seed, n=400)
        # raw_rows[i]["prior_no_show_rate"] is in [0, 1] from the
        # generator — use that as the proxy for p_no_show.
        for row in dataset.raw_rows:
            p = float(row.get("prior_no_show_rate", 0.0))
            p_values.append(p)
            if p < 0.25:
                counts["low"] += 1
            elif p < 0.50:
                counts["medium"] += 1
            else:
                counts["high"] += 1

    total = len(p_values)
    if total == 0:
        return [], 0, 0.0

    buckets = [
        NoShowRiskBucket(
            band="low",  # type: ignore[arg-type]
            count=counts["low"],
            fraction=counts["low"] / total,
        ),
        NoShowRiskBucket(
            band="medium",  # type: ignore[arg-type]
            count=counts["medium"],
            fraction=counts["medium"] / total,
        ),
        NoShowRiskBucket(
            band="high",  # type: ignore[arg-type]
            count=counts["high"],
            fraction=counts["high"] / total,
        ),
    ]
    return buckets, total, statistics.fmean(p_values)


def _read_pms_tasks(customer_id: str, base: Path) -> list[PmsTaskRow]:
    """Walk M1's PMS writeback directory and roll task.json files
    into typed rows. Returns [] when the directory doesn't exist."""
    tasks_dir = base / customer_id / "pms_writeback"
    if not tasks_dir.is_dir():
        return []

    rows: list[PmsTaskRow] = []
    for conv_dir in tasks_dir.iterdir():
        if not conv_dir.is_dir():
            continue
        task_path = conv_dir / "task.json"
        if not task_path.is_file():
            continue
        try:
            payload = json.loads(task_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        priority = payload.get("priority")
        if priority not in {"normal", "urgent"}:
            priority = "normal"
        rows.append(
            PmsTaskRow(
                task_id=str(payload.get("task_id", "—")),
                subject=str(payload.get("subject", "(no subject)")),
                priority=priority,  # type: ignore[arg-type]
                assignee_group=str(payload.get("assignee_group", "front_desk")),
                patient_id=(
                    str(payload["patient_id"])
                    if payload.get("patient_id") is not None
                    else None
                ),
                created_at=str(payload.get("generated_at", ""))[:10] or "—",
            )
        )

    # Urgent first, then by task_id for a stable display.
    rows.sort(key=lambda r: (0 if r.priority == "urgent" else 1, r.task_id))
    return rows


def _read_eligibility(
    customer_id: str, base: Path
) -> tuple[list[EligibilitySummary], int]:
    """Bucket eligibility records by status from SQLite."""
    sqlite_path = base / customer_id / "structured.sqlite3"
    if not sqlite_path.is_file():
        return [], 0

    import sqlite3

    counts: dict[str, int] = {}
    try:
        with sqlite3.connect(sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT status, COUNT(*) AS n FROM eligibility GROUP BY status"
            ).fetchall()
            for r in rows:
                counts[str(r["status"]) or "unknown"] = int(r["n"])
    except sqlite3.DatabaseError as exc:
        log.warning("eligibility: sqlite error for %r: %s", customer_id, exc)
        return [], 0

    total = sum(counts.values())
    if total == 0:
        return [], 0

    # Preserve a canonical order so colours don't flip between
    # tenants.
    order = ("active", "pending", "denied", "unknown")
    summaries: list[EligibilitySummary] = []
    for status in order:
        if status in counts:
            summaries.append(
                EligibilitySummary(
                    status=status,
                    count=counts[status],
                    fraction=counts[status] / total,
                )
            )
    # Tail — any statuses we didn't anticipate (e.g. "terminated").
    extras = sorted(k for k in counts if k not in order)
    for status in extras:
        summaries.append(
            EligibilitySummary(
                status=status,
                count=counts[status],
                fraction=counts[status] / total,
            )
        )
    return summaries, total


# ---------- Voice Intelligence ----------


# Six emotions matching what the vision brief calls out. Order
# matters — the view renders them in this sequence and stable
# ordering helps a viewer scan across customers.
_EMOTION_ORDER: tuple[str, ...] = (
    "calm",
    "anxious",
    "confused",
    "frustrated",
    "urgent",
    "distressed",
)


def _classify_emotion(entry: TraceEntry) -> str:
    """Heuristic emotion bucket for one turn.

    Maps the escalation reasons + outcome into one of six emotions.
    Priority order (highest wins):

      distressed   emergency outcome OR emergency reason
      urgent       difficulty == emergency OR intent emergency
      frustrated   any "frustration" reason
      confused     any "repeated_clarification" reason
      anxious      any "low_confidence" reason
      calm         default

    Heuristic on purpose — the engine doesn't currently surface a
    per-turn sentiment score, and inventing one in the dashboard
    would mask what Sentinel actually sees.
    """
    if _is_emergency(entry):
        return "distressed"
    if (entry.intent or "").lower() == "emergency" or (
        entry.difficulty or ""
    ).lower() == "emergency":
        return "urgent"
    reasons = " ".join(entry.escalation_reasons).lower()
    if "frustration" in reasons:
        return "frustrated"
    if "clarification" in reasons:
        return "confused"
    if "low_confidence" in reasons:
        return "anxious"
    return "calm"


def _predicted_escalation_rate(
    n_escalated: int, n_total: int, *, alpha: float = 2.0, beta: float = 8.0
) -> float:
    """Smoothed Beta-binomial estimate of the next-turn escalation rate.

    Pure historical rate is brittle for tiny samples (one
    escalation in two turns is not 50%). The Beta prior pulls a
    tiny sample back toward the prior mean (alpha / (alpha + beta) =
    0.20). Default prior strength = 10 turns, which matches the
    intuition that "we need ~10 turns before the data dominates."
    """
    if n_total <= 0:
        return alpha / (alpha + beta)
    return (n_escalated + alpha) / (n_total + alpha + beta)


# Pinned voice-pipeline budgets from the M5 Voice Layer plan. Used
# by the view to show a target-vs-budget bar chart even when the
# tenant has no live voice traces yet (which is most of the time).
_VOICE_PIPELINE: tuple[VoicePipelineStage, ...] = (
    VoicePipelineStage(
        name="STT",
        target_ms=500,
        description="OpenAI Whisper-1 transcribes the patient utterance.",
    ),
    VoicePipelineStage(
        name="Agent",
        target_ms=1500,
        description="Router -> specialist -> tools -> sentinel pipeline.",
    ),
    VoicePipelineStage(
        name="TTS",
        target_ms=600,
        description="OpenAI TTS streams the response audio back.",
    ),
)


# A small synthetic transcript used by the view's "Live Transcript"
# panel when no live turn is available. The view labels this clearly
# so the viewer knows it's an illustration, not a recorded turn.
# Confidence shading mimics what an STT engine surfaces — high on
# common words, lower on proper nouns + clinical terms.
_SAMPLE_TRANSCRIPT: tuple[tuple[str, float], ...] = (
    ("Hi,", 0.99),
    ("this", 0.99),
    ("is", 0.99),
    ("Jane", 0.91),
    ("Smith.", 0.88),
    ("I'd", 0.96),
    ("like", 0.99),
    ("to", 0.99),
    ("book", 0.99),
    ("a", 0.99),
    ("cataract", 0.78),
    ("pre-op", 0.74),
    ("consult", 0.85),
    ("with", 0.99),
    ("Dr.", 0.95),
    ("Patel,", 0.69),
    ("any", 0.97),
    ("morning", 0.94),
    ("next", 0.99),
    ("Tuesday.", 0.89),
)


def build_voice_intelligence(
    customer_id: str,
    *,
    data_dir: Path | None = None,
) -> VoiceIntelligenceSnapshot:
    """Build the Voice Intelligence rollup for one customer.

    Sources:
      * TraceReport.entries — classified into 6 emotions, plus the
        escalation_score time series for the frustration trace.
      * Static voice-pipeline targets (the M5 budget).
      * Static illustrative transcript (the live-mic path doesn't
        leave persisted artifacts; the view labels this clearly).

    Returns has_data=False shape when no trace on disk for the
    tenant.
    """
    display_name = _humanize(customer_id)
    try:
        trace = data_loader.load_trace_report(customer_id, data_dir)
    except (FileNotFoundError, data_loader.SchemaVersionMismatchError) as exc:
        log.info(
            "voice intelligence: tenant %r has no trace: %s",
            customer_id,
            exc,
        )
        return _empty_voice_intelligence(display_name)

    entries = trace.entries
    if not entries:
        return _empty_voice_intelligence(display_name)

    # Emotion distribution.
    emotion_counts: dict[str, int] = dict.fromkeys(_EMOTION_ORDER, 0)
    for e in entries:
        bucket = _classify_emotion(e)
        emotion_counts[bucket] += 1
    total = len(entries)
    emotions = [
        EmotionTotal(
            emotion=name,
            count=emotion_counts[name],
            fraction=emotion_counts[name] / total if total else 0.0,
        )
        for name in _EMOTION_ORDER
    ]

    # Frustration trace — chrono order across entries.
    frustration_trace: list[FrustrationPoint] = []
    for idx, e in enumerate(entries):
        frustration_trace.append(
            FrustrationPoint(
                turn_index=idx,
                scenario_id=e.scenario_id,
                score=e.escalation_score or 0.0,
                escalated=(e.escalation_score or 0.0) >= 0.5,
            )
        )
    mean_frustration = (
        statistics.fmean(p.score for p in frustration_trace)
        if frustration_trace
        else 0.0
    )
    escalated = sum(1 for p in frustration_trace if p.escalated)
    escalation_rate = escalated / total if total else 0.0
    predicted = _predicted_escalation_rate(escalated, total)

    return VoiceIntelligenceSnapshot(
        tenant=display_name,
        has_data=True,
        total_turns=total,
        emotions=emotions,
        frustration_trace=frustration_trace,
        mean_frustration=round(mean_frustration, 4),
        escalation_rate=round(escalation_rate, 4),
        predicted_escalation_rate=round(predicted, 4),
        voice_pipeline=list(_VOICE_PIPELINE),
        sample_transcript=list(_SAMPLE_TRANSCRIPT),
    )


def _empty_voice_intelligence(display_name: str) -> VoiceIntelligenceSnapshot:
    """has_data=False shape — still ships the voice pipeline targets
    + sample transcript so the view's lower panels render even on a
    fresh deploy with zero scored turns."""
    return VoiceIntelligenceSnapshot(
        tenant=display_name,
        has_data=False,
        total_turns=0,
        emotions=[
            EmotionTotal(emotion=name, count=0, fraction=0.0)
            for name in _EMOTION_ORDER
        ],
        frustration_trace=[],
        mean_frustration=0.0,
        escalation_rate=0.0,
        predicted_escalation_rate=_predicted_escalation_rate(0, 0),
        voice_pipeline=list(_VOICE_PIPELINE),
        sample_transcript=list(_SAMPLE_TRANSCRIPT),
    )


# ---------- Agent Flow ----------


# Map a tool name to the specialist that owns it. Mirrors the
# clarion.multiagent.specialists.* allowed_tools sets. The Info
# specialist is the fallback when no booking / eligibility /
# cancel tool fired.
_TOOL_TO_SPECIALIST: dict[str, str] = {
    "search_slots":       "Booking",
    "book_appointment":   "Booking",
    "check_eligibility":  "Eligibility",
    "cancel_appointment": "Cancel",
    "create_pms_task":    "Info",
}

# Canonical ordered specialist list — drives the "other specialists"
# grey-out panel.
_ALL_SPECIALISTS: tuple[str, ...] = (
    "Booking",
    "Eligibility",
    "Info",
    "Cancel",
    "Emergency",
)


def build_agent_flow(
    customer_id: str,
    *,
    scenario_id: str | None = None,
    data_dir: Path | None = None,
) -> AgentFlowSnapshot:
    """Reconstruct one turn's path through the multi-agent graph.

    ``scenario_id`` picks which entry to inspect. ``None`` selects
    the first entry in the trace (sensible default for the view's
    initial load).

    The trace doesn't directly record which specialist handled the
    turn — the locked TraceEntry schema predates the multi-agent
    refactor. We infer it from the tools fired:

      booking tools fired       -> BookingSpecialist
      eligibility tool fired    -> EligibilitySpecialist
      cancel tool fired         -> CancelSpecialist
      emergency outcome OR
      "emergency" reason fired  -> EmergencySpecialist
      otherwise                 -> InfoSpecialist (the read-only
                                   fallback)

    Returns an empty snapshot when no trace data is on disk for
    the customer. The view interprets that as an empty-state
    message rather than crashing.
    """
    display_name = _humanize(customer_id)
    try:
        trace = data_loader.load_trace_report(customer_id, data_dir)
    except (FileNotFoundError, data_loader.SchemaVersionMismatchError) as exc:
        log.info("agent flow: tenant %r has no trace: %s", customer_id, exc)
        return _empty_agent_flow(display_name)

    if not trace.entries:
        return _empty_agent_flow(display_name)

    available_turns = [e.scenario_id for e in trace.entries]
    if scenario_id is None:
        entry = trace.entries[0]
    else:
        entry = next(
            (e for e in trace.entries if e.scenario_id == scenario_id),
            trace.entries[0],
        )

    chosen, others = _infer_specialist(entry)
    nodes = _build_flow_nodes(entry, chosen=chosen)

    return AgentFlowSnapshot(
        tenant=display_name,
        scenario_id=entry.scenario_id,
        intent=entry.intent or "—",
        user_message=_first_user_message(entry),
        nodes=nodes,
        chosen_specialist=chosen,
        other_specialists=others,
        tools_called=list(entry.tools_called),
        escalation_score=entry.escalation_score or 0.0,
        escalation_reasons=list(entry.escalation_reasons),
        judge_violations=list(entry.judge_violations),
        final_outcome=entry.actual_outcome,
        has_data=True,
        available_turns=available_turns,
    )


# ---------- Agent Flow internals ----------


def _empty_agent_flow(display_name: str) -> AgentFlowSnapshot:
    return AgentFlowSnapshot(
        tenant=display_name,
        scenario_id="—",
        intent="—",
        user_message="",
        nodes={},
        chosen_specialist="—",
        other_specialists=list(_ALL_SPECIALISTS),
        tools_called=[],
        escalation_score=0.0,
        escalation_reasons=[],
        judge_violations=[],
        final_outcome="—",
        has_data=False,
        available_turns=[],
    )


def _infer_specialist(entry: TraceEntry) -> tuple[str, list[str]]:
    """Pick the specialist that most likely handled this turn.

    Priority order: emergency outcome > emergency reason >
    tool-to-specialist mapping > Info fallback. Emergency wins
    over everything because EmergencySpecialist short-circuits in
    the runner before any tool fires.
    """
    if _is_emergency(entry):
        chosen = "Emergency"
    else:
        chosen = "Info"  # default if no tool maps anywhere
        for tool in entry.tools_called:
            if tool in _TOOL_TO_SPECIALIST:
                chosen = _TOOL_TO_SPECIALIST[tool]
                break
    others = [name for name in _ALL_SPECIALISTS if name != chosen]
    return chosen, others


def _first_user_message(entry: TraceEntry) -> str:
    """The TraceEntry doesn't carry the user message directly —
    we infer a short label from intent + difficulty. That's enough
    for the view's "Turn driving this flow" header line."""
    return f"{entry.intent or 'turn'} · {entry.difficulty or 'unknown'}"


def _build_flow_nodes(
    entry: TraceEntry, *, chosen: str
) -> dict[str, FlowNode]:
    """Assemble per-position FlowNode dicts for the diagram."""
    # The trace doesn't time individual graph nodes. We split
    # duration_ms across the path proportionally — Router + Specialist
    # take the bulk, Sentinel a slice, Response a small tail.
    total_ms = int(entry.duration_ms or 0)
    cost_usd = entry.cost_usd or 0.0
    tools_count = len(entry.tools_called)
    escalated = entry.escalation_score is not None and entry.escalation_score >= 0.5
    emergency = _is_emergency(entry)

    router_ms = int(total_ms * 0.10)
    specialist_ms = int(total_ms * 0.55)
    tools_ms = int(total_ms * 0.15)
    sentinel_ms = int(total_ms * 0.10)
    # Response slice is whatever rounding left.
    response_ms = max(0, total_ms - router_ms - specialist_ms - tools_ms - sentinel_ms)

    return {
        "patient": FlowNode(
            name="PATIENT",
            state="done",
            ms=None,
            cost_usd=None,
            detail=f"{entry.intent or 'turn'} · {entry.difficulty or 'unknown'}",
        ),
        "router": FlowNode(
            name="ROUTER",
            state="done",
            ms=router_ms or None,
            cost_usd=cost_usd * 0.10 if cost_usd else None,
            detail=f"chose {chosen}",
        ),
        "specialist": FlowNode(
            name=chosen.upper(),
            state="escalated" if emergency else ("done" if not escalated else "active"),
            ms=specialist_ms or None,
            cost_usd=cost_usd * 0.55 if cost_usd else None,
            detail=f"{tools_count} tool{'s' if tools_count != 1 else ''} fired",
        ),
        "tools": FlowNode(
            name="TOOLS",
            state="done" if tools_count > 0 else "idle",
            ms=tools_ms or None if tools_count else None,
            cost_usd=None,
            detail=", ".join(entry.tools_called[:3]) or "none",
        ),
        "sentinel": FlowNode(
            name="SENTINEL",
            state="escalated" if escalated else "done",
            ms=sentinel_ms or None,
            cost_usd=None,
            detail=_sentinel_detail(entry),
        ),
        "response": FlowNode(
            name="RESPONSE",
            state="escalated" if emergency else "done",
            ms=response_ms or None,
            cost_usd=cost_usd * 0.10 if cost_usd else None,
            detail=entry.actual_outcome,
        ),
    }


def _sentinel_detail(entry: TraceEntry) -> str:
    """Single-line summary of what Sentinel decided on this turn."""
    score = entry.escalation_score
    if score is None:
        return "score n/a"
    parts: list[str] = [f"score {score:.2f}"]
    if entry.judge_hallucination is not None:
        parts.append(f"halluc {entry.judge_hallucination:.2f}")
    if entry.judge_violations:
        parts.append(f"{len(entry.judge_violations)} violation(s)")
    return " · ".join(parts)


# ---------- Sentinel Operations Center ----------


# Signal name → human-readable label rendered in the UI. Pinning
# the list here keeps the view from inventing display names.
_SIGNAL_LABELS: dict[str, str] = {
    "low_confidence":         "Low confidence",
    "repeated_clarification": "Repeated clarification",
    "rule_conflict":          "Rule conflict",
    "frustration":            "Frustration",
    "unsupported_request":    "Unsupported request",
}

# Match "name=0.42" reason strings the scorer emits. Anchored at the
# start so a stray "frustration_x=0.5" never collides with the
# frustration signal. The value is optional — older traces sometimes
# carry bare reasons like "already_escalated".
_REASON_RE = re.compile(r"^([a-z_]+)(?:=([0-9.]+))?$")


def build_sentinel_ops(
    customer_id: str,
    *,
    data_dir: Path | None = None,
) -> SentinelOpsSnapshot:
    """Build the Sentinel Operations Center rollup for one customer.

    Aggregates over:
      * ``TraceReport.entries`` for escalation score + signal
        contributions + judge agreement
      * ``EvaluationReport.metrics`` for headline precision/recall/F1
        (the scoreboard tile)
      * ``data_dir/<customer_id>/audit.jsonl`` for PHI redaction
        totals + the audit tail

    Returns ``has_data=False`` snapshot when no trace report is on
    disk for the customer. The view interprets that as the
    empty-state message rather than crashing.
    """
    display_name = _humanize(customer_id)

    try:
        report = data_loader.load_report(customer_id, data_dir)
        trace = data_loader.load_trace_report(customer_id, data_dir)
    except (FileNotFoundError, data_loader.SchemaVersionMismatchError) as exc:
        log.info("sentinel ops: tenant %r has no usable artifacts: %s", customer_id, exc)
        return _empty_sentinel_ops(display_name)

    entries = trace.entries
    if not entries:
        return _empty_sentinel_ops(display_name)

    # Threshold reflects the locked scorer convention — anything at
    # or above 0.5 escalates. We expose the raw value so the view's
    # gauge can draw the hairline at the right place.
    decision_threshold = 0.5

    # Composite — mean across entries that have any score at all.
    scored_entries = [e for e in entries if e.escalation_score is not None]
    mean_score = (
        statistics.fmean(e.escalation_score or 0.0 for e in scored_entries)
        if scored_entries
        else 0.0
    )

    signals = _signal_contributions(entries)
    judge = _judge_agreement(entries)
    audit_total, audit_tail = _audit_view(customer_id, data_dir=data_dir)
    emergencies = sum(1 for e in entries if _is_emergency(e))

    return SentinelOpsSnapshot(
        tenant=display_name,
        has_data=True,
        sample_count=len(entries),
        mean_escalation_score=round(mean_score, 4),
        trust_score=max(0.0, 1.0 - mean_score),
        decision_threshold=decision_threshold,
        signals=signals,
        judge=judge,
        phi_redactions_total=audit_total,
        audit_tail=audit_tail,
        emergencies_caught=emergencies,
        escalation_precision=report.metrics.escalation_precision,
        escalation_recall=report.metrics.escalation_recall,
        escalation_f1=report.metrics.escalation_f1,
    )


# ---------- Sentinel internals ----------


def _empty_sentinel_ops(display_name: str) -> SentinelOpsSnapshot:
    """The no-data shape — every field zeroed but in the right shape
    so the view's renderer doesn't need a separate code path."""
    return SentinelOpsSnapshot(
        tenant=display_name,
        has_data=False,
        sample_count=0,
        mean_escalation_score=0.0,
        trust_score=0.0,
        decision_threshold=0.5,
        signals=[],
        judge=JudgeAgreement(
            sampled_turns=0,
            no_hallucination_pct=0.0,
            booking_correct_mean=0.0,
            violations_total=0,
        ),
        phi_redactions_total=0,
        audit_tail=[],
        emergencies_caught=0,
        escalation_precision=0.0,
        escalation_recall=0.0,
        escalation_f1=0.0,
    )


def _signal_contributions(entries: list[TraceEntry]) -> list[SignalContribution]:
    """Mean per-signal value x configured weight, across all entries.

    The scorer logs each fired signal as a reason string like
    ``"low_confidence=0.80"``. We parse those strings, average per
    signal name, multiply by the schema-default weight, and return
    one row per signal in the order the schema defines them.
    """
    weights = EscalationWeights()  # locked defaults from the schema
    weight_by_name = {
        "low_confidence":         weights.low_confidence,
        "repeated_clarification": weights.repeated_clarification,
        "rule_conflict":          weights.rule_conflict,
        "frustration":            weights.frustration,
        "unsupported_request":    weights.unsupported_request,
    }

    # Bucket parsed values per signal.
    per_signal_values: dict[str, list[float]] = {name: [] for name in weight_by_name}
    fire_count: dict[str, int] = {name: 0 for name in weight_by_name}

    for entry in entries:
        for reason in entry.escalation_reasons:
            match = _REASON_RE.match(reason)
            if not match:
                continue
            name = match.group(1)
            if name not in per_signal_values:
                continue
            value_str = match.group(2)
            if value_str is None:
                # Reasons without a value still mean the signal fired;
                # treat as 1.0 so the count rises but the mean stays
                # bounded by entries that DID carry a numeric value.
                fire_count[name] += 1
                continue
            try:
                value = float(value_str)
            except ValueError:
                continue
            per_signal_values[name].append(value)
            fire_count[name] += 1

    out: list[SignalContribution] = []
    for name, label in _SIGNAL_LABELS.items():
        values = per_signal_values[name]
        raw_mean = statistics.fmean(values) if values else 0.0
        weight = weight_by_name[name]
        out.append(
            SignalContribution(
                name=label,
                raw_mean=round(raw_mean, 4),
                weight=weight,
                contribution=round(raw_mean * weight, 4),
                fire_count=fire_count[name],
            )
        )
    return out


def _judge_agreement(entries: list[TraceEntry]) -> JudgeAgreement:
    """LLM-as-judge rollup. Entries without a verdict are excluded
    from the rates so a small judge sample doesn't get diluted by
    judge-less entries."""
    judged = [e for e in entries if e.judge_hallucination is not None]
    if not judged:
        return JudgeAgreement(
            sampled_turns=0,
            no_hallucination_pct=0.0,
            booking_correct_mean=0.0,
            violations_total=0,
        )
    no_halluc = sum(1 for e in judged if (e.judge_hallucination or 0.0) < 0.05)
    booking_values = [
        e.judge_booking_correct for e in judged if e.judge_booking_correct is not None
    ]
    booking_mean = statistics.fmean(booking_values) if booking_values else 0.0
    violations = sum(len(e.judge_violations) for e in judged)
    return JudgeAgreement(
        sampled_turns=len(judged),
        no_hallucination_pct=no_halluc / len(judged),
        booking_correct_mean=round(booking_mean, 4),
        violations_total=violations,
    )


def _audit_view(
    customer_id: str,
    *,
    data_dir: Path | None,
) -> tuple[int, list[AuditTailItem]]:
    """Read ``<data_dir>/<customer>/audit.jsonl``.

    Returns (total_phi_redactions_across_log, last_10_events_newest_first).
    When the audit file doesn't exist (fresh deploy / never written),
    surfaces (0, []). The view explicitly handles that as "no audit
    on disk yet — agent never spoke."
    """
    base = data_dir if data_dir is not None else data_loader.DEFAULT_DATA_DIR
    path = base / customer_id / "audit.jsonl"
    if not path.is_file():
        return 0, []

    items: list[AuditTailItem] = []
    total_redactions = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                redactions = record.get("redactions") or {}
                if isinstance(redactions, dict):
                    total_redactions += sum(
                        int(v) for v in redactions.values() if isinstance(v, int)
                    )
                items.append(_audit_record_to_item(record))
    except OSError:
        return 0, []

    # Newest first, capped at 10. The audit log is append-only so
    # the last lines on disk are the most recent.
    items.reverse()
    return total_redactions, items[:10]


def _audit_record_to_item(record: dict) -> AuditTailItem:
    """Map one audit JSON line to the typed tail-row dataclass."""
    ts_raw = str(record.get("timestamp", ""))
    # Show "HH:MM:SS" in UTC for a tidy row. Falls back to the raw
    # value when the format isn't ISO.
    ts = ts_raw[11:19] if "T" in ts_raw and len(ts_raw) >= 19 else ts_raw[:8]
    guardrail = str(record.get("guardrail", "safe"))
    kind = _kind_from_guardrail(guardrail)
    # Truncate the redacted reply so the row stays one line. The
    # writer already PHI-redacted the content — we never re-redact.
    reply = str(record.get("agent_reply", ""))
    summary = (reply[:100] + "…") if len(reply) > 100 else reply
    return AuditTailItem(ts=ts, kind=kind, summary=summary or "(no reply)")


def _kind_from_guardrail(guardrail: str) -> str:
    """Map a guardrail outcome string to a short UI label."""
    if guardrail == "safe":
        return "chat"
    if guardrail.startswith("emergency"):
        return "emergency"
    if guardrail.startswith("clinical"):
        return "clinical"
    return guardrail


# ---------- Cost & SLO ----------


def build_cost_slo(
    snapshots: list[TenantSnapshot],
    *,
    data_dir: Path | None = None,
) -> GlobalCostSLO:
    """Roll cost + latency from every tenant's TraceReport."""
    per_cost: list[CostBreakdown] = []
    per_lat: list[LatencyBreakdown] = []
    all_lats: list[float] = []
    total_cost = 0.0

    for snap in snapshots:
        if not snap.has_data:
            continue
        try:
            trace = data_loader.load_trace_report(snap.customer_id, data_dir)
        except (FileNotFoundError, data_loader.SchemaVersionMismatchError):
            continue
        costs = [e.cost_usd for e in trace.entries if e.cost_usd is not None]
        latencies = [e.duration_ms for e in trace.entries if e.duration_ms is not None]
        in_tokens = sum(e.input_tokens for e in trace.entries)
        out_tokens = sum(e.output_tokens for e in trace.entries)
        cost_total = sum(costs) if costs else 0.0
        cost_avg = cost_total / max(1, len(costs)) if costs else 0.0

        per_cost.append(
            CostBreakdown(
                tenant=snap.display_name,
                total_cost_usd=cost_total,
                avg_cost_per_request_usd=cost_avg,
                total_input_tokens=in_tokens,
                total_output_tokens=out_tokens,
                scenarios=len(trace.entries),
            )
        )
        total_cost += cost_total

        if latencies:
            per_lat.append(
                LatencyBreakdown(
                    tenant=snap.display_name,
                    avg_ms=statistics.fmean(latencies),
                    p50_ms=_pct(latencies, 50.0),
                    p95_ms=_pct(latencies, 95.0),
                    sample_count=len(latencies),
                )
            )
            all_lats.extend(latencies)

    global_p50 = _pct(all_lats, 50.0) if all_lats else 0.0
    global_p95 = _pct(all_lats, 95.0) if all_lats else 0.0
    # Naive projection: today's total / N tenants extrapolated to a month.
    # This is intentionally simple — the snapshot is "what the dashboard
    # has on disk right now", not a real billing forecast.
    monthly_projection = total_cost * 30.0

    return GlobalCostSLO(
        per_tenant_cost=per_cost,
        per_tenant_latency=per_lat,
        total_cost_usd=total_cost,
        monthly_projection_usd=monthly_projection,
        global_p50_ms=global_p50,
        global_p95_ms=global_p95,
    )


# ---------- low-level helpers ----------


def _humanize(customer_id: str) -> str:
    """customer_id -> Title Case display name."""
    return " ".join(part.capitalize() for part in customer_id.split("_"))


def _composite_headline_score(report: EvaluationReport) -> float:
    """0..1 composite used to bucket a tenant into a health band.

    A weighted mean of containment + safety + booking + (1 -
    hallucination). Weights tuned so a single-axis failure still
    drags the bucket down hard (FDE intuition: a healthcare front
    desk that hits 100% safety but 50% containment is NOT healthy).
    """
    m = report.metrics
    weights = {
        "containment_rate": 0.35,
        "safety_catch_rate": 0.30,
        "booking_accuracy": 0.20,
        "halluc_inv": 0.15,
    }
    values = {
        "containment_rate": m.containment_rate,
        "safety_catch_rate": m.safety_catch_rate,
        "booking_accuracy": m.booking_accuracy,
        "halluc_inv": 1.0 - (m.hallucination_rate or 0.0),
    }
    score = sum(weights[k] * values[k] for k in weights)
    return max(0.0, min(1.0, score))


def _health_from_headline(score: float) -> HealthStatus:
    if score >= 0.90:
        return "healthy"
    if score >= 0.70:
        return "warning"
    return "critical"


def _count_emergencies(
    snapshots: list[TenantSnapshot],
    *,
    data_dir: Path | None,
) -> int:
    """Cross-tenant total emergency count from the trace entries."""
    total = 0
    for snap in snapshots:
        if not snap.has_data:
            continue
        try:
            trace = data_loader.load_trace_report(snap.customer_id, data_dir)
        except (FileNotFoundError, data_loader.SchemaVersionMismatchError):
            continue
        total += sum(1 for e in trace.entries if _is_emergency(e))
    return total


def _is_emergency(entry: TraceEntry) -> bool:
    """A trace entry is an emergency if it has the emergency outcome
    OR the guardrail-fired escalation reason."""
    if entry.actual_outcome == "escalated_emergency":
        return True
    return any(reason.startswith("emergency") for reason in entry.escalation_reasons)


def _escalation_severity(entry: TraceEntry) -> HealthStatus:
    """Map an entry's escalation score band to a UI severity colour."""
    score = entry.escalation_score or 0.0
    if score >= 0.80 or _is_emergency(entry):
        return "critical"
    if score >= 0.50:
        return "warning"
    return "info" if entry.escalation_reasons else "unknown"  # type: ignore[return-value]


def _format_escalation_summary(entry: TraceEntry) -> str:
    """Compact summary line for one escalation row.

    Stitches together the reasons + a tiny scenario id suffix so the
    viewer can correlate to the Trace Explorer tab without leaving
    Mission Control."""
    reasons = ", ".join(entry.escalation_reasons[:3]) or "escalated"
    tail = entry.scenario_id[-8:]
    return f"{reasons}  ·  scenario_{tail}"


def _format_emergency_summary(entry: TraceEntry) -> str:
    if entry.agent_replies:
        return f"{entry.agent_replies[0][:80].rstrip()}…"
    return f"emergency intent on scenario_{entry.scenario_id[-8:]}"


def _pct(values: list[float], pct: float) -> float:
    """Cheap percentile — sorts in place, no numpy."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    # Linear interpolation between the two bracketing samples.
    return s[f] + (s[c] - s[f]) * (k - f)


def _relative_time(ts: datetime | None) -> str:
    """Render a UTC timestamp as "12m ago" / "3h ago" / "yesterday".

    Phase B uses a fixed reference of "now". Phase F can swap to a
    rolling-window aggregator if we want sub-minute precision."""
    if ts is None:
        return "—"
    # Normalize to UTC so subtraction works regardless of tz info.
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    now = datetime.now(UTC)
    delta = now - ts
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    if secs < 86400 * 2:
        return "yesterday"
    return f"{secs // 86400}d ago"


# ============================================================
# Patient 360 — per-patient longitudinal record
# ============================================================
#
# A read-only roll-up of one synthetic patient's interactions
# with Clarion: their profile, the timeline of touchpoints
# (voice calls, appointments, eligibility checks, escalations),
# their care team, and their insurance state. Purely synthetic
# data; no PHI; no SQLite persistence yet (a future task can
# wire this to the same store that backs HealthcareOps).


@dataclass(frozen=True)
class PatientTimelineEvent:
    """One event in a patient's care timeline.

    `kind` drives the icon + color on the rendered row.
    `severity` is one of healthy / info / warning / critical so the
    row can be tinted. `ts` is timezone-aware UTC.
    """

    ts: datetime
    kind: Literal[
        "voice_call", "appointment", "eligibility", "escalation", "pms_task", "note"
    ]
    title: str
    detail: str
    severity: Literal["healthy", "info", "warning", "critical"] = "info"


@dataclass(frozen=True)
class PatientCareTeamMember:
    name: str
    role: str
    is_primary: bool = False


@dataclass(frozen=True)
class PatientInsurance:
    payer: str
    member_id: str
    plan: str
    eligibility_status: Literal["active", "pending", "lapsed", "unknown"]
    last_verified_at: datetime | None


@dataclass(frozen=True)
class PatientProfile:
    """One patient's longitudinal record."""

    patient_id: str
    display_name: str
    dob_display: str
    age_years: int
    phone_display: str
    email: str
    address: str
    preferred_language: str
    customer_id: str

    # Engagement + sentiment scoring (Bayesian-smoothed beta-binomial
    # over their interaction history; values in [0, 1]).
    engagement_score: float
    sentiment_score: float
    trust_score: float

    care_team: tuple[PatientCareTeamMember, ...]
    insurance: PatientInsurance | None
    timeline: tuple[PatientTimelineEvent, ...]


@dataclass(frozen=True)
class Patient360Snapshot:
    """The full Patient 360 view payload — patients in this tenant
    plus the currently-selected patient's profile.

    Empty `patients` triggers the view's empty state.
    """

    schema_version: str
    customer_id: str
    patients: tuple[PatientProfile, ...]
    selected: PatientProfile | None


_PATIENT_360_SCHEMA_VERSION = "1.0.0"


def _synthetic_patient_360(
    customer_id: str,
) -> tuple[PatientProfile, ...]:
    """Generate a small deterministic synthetic patient roster
    per tenant. Two patients each, with different journey shapes.

    No PHI; no live data. Useful for empty-data demos so the view
    renders something meaningful without needing a runner pass.
    """
    now = datetime.now(UTC)

    def event(
        hours_ago: int,
        kind: Literal[
            "voice_call",
            "appointment",
            "eligibility",
            "escalation",
            "pms_task",
            "note",
        ],
        title: str,
        detail: str,
        severity: Literal["healthy", "info", "warning", "critical"] = "info",
    ) -> PatientTimelineEvent:
        return PatientTimelineEvent(
            ts=now - timedelta(hours=hours_ago),
            kind=kind,
            title=title,
            detail=detail,
            severity=severity,
        )

    if customer_id == "orthopedics":
        roster = (
            PatientProfile(
                patient_id="pt_ortho_1024",
                display_name="Dana Whitfield",
                dob_display="1972-04-18",
                age_years=53,
                phone_display="(415) 555-2104",
                email="dana.w@example.invalid",
                address="900 Folsom St, San Francisco, CA 94107",
                preferred_language="en",
                customer_id=customer_id,
                engagement_score=0.71,
                sentiment_score=0.58,
                trust_score=0.93,
                care_team=(
                    PatientCareTeamMember(
                        "Dr. Priya Anand", "Orthopedic Surgeon", is_primary=True
                    ),
                    PatientCareTeamMember("Marcus Reed, NP", "Care Navigator"),
                    PatientCareTeamMember("Helen Vargas", "Patient Liaison"),
                ),
                insurance=PatientInsurance(
                    payer="Anthem Blue",
                    member_id="ABX-441-208",
                    plan="PPO Gold",
                    eligibility_status="active",
                    last_verified_at=now - timedelta(days=6),
                ),
                timeline=(
                    event(
                        4,
                        "voice_call",
                        "Voice intake — post-op pain follow-up",
                        "Caller reported pain 4/10, slight swelling. Routed to NP.",
                        "info",
                    ),
                    event(
                        29,
                        "appointment",
                        "Appointment scheduled",
                        "Knee arthroscopy follow-up booked with Dr. Anand.",
                        "healthy",
                    ),
                    event(
                        48,
                        "eligibility",
                        "Eligibility re-verified",
                        "Anthem PPO Gold active; co-pay $35 collected at intake.",
                        "healthy",
                    ),
                    event(
                        76,
                        "pms_task",
                        "PMS task: schedule physical therapy",
                        "Routed to ortho-ops queue; assignee Marcus Reed, NP.",
                        "info",
                    ),
                    event(
                        144,
                        "escalation",
                        "Sentinel escalation",
                        "Low confidence on insurance plan match; human resolved.",
                        "warning",
                    ),
                ),
            ),
            PatientProfile(
                patient_id="pt_ortho_1031",
                display_name="Theo Brennan",
                dob_display="1988-11-02",
                age_years=37,
                phone_display="(415) 555-0890",
                email="theo.b@example.invalid",
                address="221 Townsend St, San Francisco, CA 94107",
                preferred_language="es",
                customer_id=customer_id,
                engagement_score=0.42,
                sentiment_score=0.31,
                trust_score=0.81,
                care_team=(
                    PatientCareTeamMember(
                        "Dr. Lin Hwang", "Sports Medicine", is_primary=True
                    ),
                    PatientCareTeamMember("Jordan Ellis", "Care Navigator"),
                ),
                insurance=PatientInsurance(
                    payer="United Healthcare",
                    member_id="UHC-877-016",
                    plan="HMO Choice",
                    eligibility_status="pending",
                    last_verified_at=now - timedelta(days=21),
                ),
                timeline=(
                    event(
                        2,
                        "escalation",
                        "Sentinel escalation — frustration spike",
                        "Caller reported third reschedule; routed to live agent.",
                        "critical",
                    ),
                    event(
                        3,
                        "voice_call",
                        "Voice retry — booking confusion",
                        "Frustration detected mid-call; agent reassured patient.",
                        "warning",
                    ),
                    event(
                        50,
                        "appointment",
                        "Appointment rescheduled",
                        "Original slot conflicted with provider time-off.",
                        "info",
                    ),
                    event(
                        120,
                        "note",
                        "Care navigator note",
                        "Patient prefers Spanish call-backs after 5pm PT.",
                        "info",
                    ),
                ),
            ),
        )
        return roster

    # Default tenant — ophthalmology.
    roster = (
        PatientProfile(
            patient_id="pt_oph_2017",
            display_name="Avery Sinclair",
            dob_display="1956-09-30",
            age_years=68,
            phone_display="(206) 555-3104",
            email="avery.s@example.invalid",
            address="118 Pine St, Seattle, WA 98101",
            preferred_language="en",
            customer_id=customer_id,
            engagement_score=0.78,
            sentiment_score=0.66,
            trust_score=0.95,
            care_team=(
                PatientCareTeamMember(
                    "Dr. Ines Park", "Retina Specialist", is_primary=True
                ),
                PatientCareTeamMember("Camille Ortega, OD", "Optometrist"),
                PatientCareTeamMember("Robin Lutz", "Patient Liaison"),
            ),
            insurance=PatientInsurance(
                payer="Premera Blue Cross",
                member_id="PBC-3318-104",
                plan="HMO Standard",
                eligibility_status="active",
                last_verified_at=now - timedelta(days=2),
            ),
            timeline=(
                event(
                    6,
                    "voice_call",
                    "Voice intake — cataract pre-op",
                    "Reviewed pre-op fasting + dilation drops protocol.",
                    "info",
                ),
                event(
                    30,
                    "appointment",
                    "Appointment confirmed",
                    "Cataract consult with Dr. Park on Friday at 10:30.",
                    "healthy",
                ),
                event(
                    54,
                    "eligibility",
                    "Eligibility verified",
                    "Premera HMO active; pre-op covered with $50 co-pay.",
                    "healthy",
                ),
                event(
                    102,
                    "note",
                    "Care team note",
                    "Patient reports mild floaters; flagged for retina review.",
                    "info",
                ),
            ),
        ),
        PatientProfile(
            patient_id="pt_oph_2042",
            display_name="Mira Khoury",
            dob_display="1991-02-14",
            age_years=34,
            phone_display="(206) 555-7782",
            email="mira.k@example.invalid",
            address="2400 Westlake Ave N, Seattle, WA 98109",
            preferred_language="ar",
            customer_id=customer_id,
            engagement_score=0.53,
            sentiment_score=0.74,
            trust_score=0.88,
            care_team=(
                PatientCareTeamMember(
                    "Dr. Salim Rahimi", "General Ophthalmology", is_primary=True
                ),
                PatientCareTeamMember("Ana Belmonte, OD", "Optometrist"),
            ),
            insurance=PatientInsurance(
                payer="Kaiser Permanente",
                member_id="KP-8821-704",
                plan="PPO Plus",
                eligibility_status="active",
                last_verified_at=now - timedelta(days=11),
            ),
            timeline=(
                event(
                    1,
                    "appointment",
                    "Appointment booked",
                    "Glaucoma screening with Dr. Rahimi next Tuesday.",
                    "healthy",
                ),
                event(
                    12,
                    "voice_call",
                    "Voice intake — screening inquiry",
                    "Caller asked about IOP follow-up scheduling.",
                    "info",
                ),
                event(
                    72,
                    "pms_task",
                    "PMS task: send glaucoma education packet",
                    "Routed to patient-support queue; delivered via email.",
                    "info",
                ),
            ),
        ),
    )
    return roster


def build_patient_360(
    customer_id: str,
    *,
    selected_patient_id: str | None = None,
) -> Patient360Snapshot:
    """Build a Patient360Snapshot for one tenant.

    Currently uses synthetic patients. A future task can switch
    this to read from `data/<tenant>/patients.sqlite` (or wherever
    the M1 PMS writeback lands long-form patient records).
    """
    patients = _synthetic_patient_360(customer_id)
    selected: PatientProfile | None = None
    if selected_patient_id is not None:
        for p in patients:
            if p.patient_id == selected_patient_id:
                selected = p
                break
    if selected is None and patients:
        selected = patients[0]
    return Patient360Snapshot(
        schema_version=_PATIENT_360_SCHEMA_VERSION,
        customer_id=customer_id,
        patients=patients,
        selected=selected,
    )


# ============================================================
# System Health — the Platform section's status board
# ============================================================
#
# A read-only roll-up of subsystem health for the Deployment /
# System Health view (Phase H2). Mirrors the audit-friendly
# "everything operational?" view typical of mission-control
# dashboards: each subsystem ticks a green/amber/red light with
# a small status note. Today the values are computed from local
# config + env presence; a future task can swap to live probes.


@dataclass(frozen=True)
class SubsystemStatus:
    """One subsystem row on the system-health board."""

    name: str
    status: Literal["healthy", "warning", "critical", "unknown"]
    note: str
    # E.g. "p95 142ms", "v1.0.0", "0 errors / hr"
    metric_display: str = ""


@dataclass(frozen=True)
class ResourceMetric:
    """One resource utilisation bar (CPU / memory / storage)."""

    name: str
    used_pct: float  # 0.0 to 1.0
    detail: str  # e.g. "42% of 16 GB"


@dataclass(frozen=True)
class SystemHealthSnapshot:
    schema_version: str
    overall: Literal["healthy", "warning", "critical", "unknown"]
    subsystems: tuple[SubsystemStatus, ...]
    resources: tuple[ResourceMetric, ...]
    version_display: str


_SYSTEM_HEALTH_SCHEMA_VERSION = "1.0.0"


def build_system_health() -> SystemHealthSnapshot:
    """Build the System Health snapshot.

    Subsystem statuses are derived from local indicators:
      - API service              : always healthy if this builder runs
      - LLM provider             : healthy if OPENAI_API_KEY is set,
                                   else "demo mode"
      - FAISS retriever          : healthy if any tenant has an
                                   index dir on disk
      - Voice (Whisper + TTS)    : healthy if voice schema imports
      - Sentinel guardrails      : always healthy (always-on)
      - Observability tracer     : healthy if span dir is writable
      - Customer config registry : healthy if available_customers()
                                   returns at least one entry

    Resource bars are synthetic placeholders — a future task can
    wire them to /proc/meminfo and psutil. The shape stays.
    """
    import os

    customers = data_loader.available_customers()
    api_healthy = bool(customers)
    llm_healthy = bool(os.environ.get("OPENAI_API_KEY"))
    rag_healthy = any(
        (Path(__file__).parent.parent / "data" / c / "rules.faiss").exists()
        for c in customers
    )

    subsystems: tuple[SubsystemStatus, ...] = (
        SubsystemStatus(
            name="API service",
            status="healthy" if api_healthy else "critical",
            note=(
                f"{len(customers)} customer(s) registered"
                if api_healthy
                else "no customers configured"
            ),
            metric_display="FastAPI · 7860",
        ),
        SubsystemStatus(
            name="LLM provider",
            status="healthy" if llm_healthy else "warning",
            note=(
                "OPENAI_API_KEY present" if llm_healthy else "demo mode (FakeLLM)"
            ),
            metric_display="gpt-4o-mini",
        ),
        SubsystemStatus(
            name="FAISS retriever",
            status="healthy" if rag_healthy else "warning",
            note=(
                "rules indices on disk" if rag_healthy else "no FAISS index found"
            ),
            metric_display="text-embedding-3-small / TF-IDF",
        ),
        SubsystemStatus(
            name="Voice services",
            status="healthy",
            note="Whisper + TTS orchestrator ready",
            metric_display="whisper-1 · tts-1",
        ),
        SubsystemStatus(
            name="Sentinel guardrails",
            status="healthy",
            note="judge + escalation + PHI redactor enabled",
            metric_display="always-on",
        ),
        SubsystemStatus(
            name="Customer registry",
            status="healthy" if customers else "critical",
            note=", ".join(customers) if customers else "empty",
            metric_display=f"{len(customers)} tenant(s)",
        ),
    )

    overall: Literal["healthy", "warning", "critical", "unknown"] = "healthy"
    if any(s.status == "critical" for s in subsystems):
        overall = "critical"
    elif any(s.status == "warning" for s in subsystems):
        overall = "warning"

    # Synthetic resource bars. Stable values so the view doesn't
    # jitter between page refreshes; a real probe slots in later.
    resources: tuple[ResourceMetric, ...] = (
        ResourceMetric("CPU", 0.42, "42% of 2 vCPU"),
        ResourceMetric("Memory", 0.58, "9.3 GB / 16 GB"),
        ResourceMetric("Storage", 0.36, "4.6 GB / 12.5 GB"),
        ResourceMetric("Concurrency", 0.21, "21% of capacity"),
    )

    try:
        import importlib.metadata as _meta

        version_display = _meta.version("clarion")
    except Exception:
        version_display = "dev"

    return SystemHealthSnapshot(
        schema_version=_SYSTEM_HEALTH_SCHEMA_VERSION,
        overall=overall,
        subsystems=subsystems,
        resources=resources,
        version_display=version_display,
    )


__all__ = [
    "AgentFlowSnapshot",
    "AuditTailItem",
    "CostBreakdown",
    "EligibilitySummary",
    "EmergencyItem",
    "EmotionTotal",
    "EscalationItem",
    "FlowNode",
    "FlowPosition",
    "FrustrationPoint",
    "GlobalCostSLO",
    "GlobalKPIs",
    "HealthStatus",
    "HealthcareOpsSnapshot",
    "JudgeAgreement",
    "LatencyBreakdown",
    "NoShowRiskBucket",
    "Patient360Snapshot",
    "PatientCareTeamMember",
    "PatientInsurance",
    "PatientProfile",
    "PatientTimelineEvent",
    "PmsTaskRow",
    "ProviderUtilization",
    "ResourceMetric",
    "SentinelOpsSnapshot",
    "SignalContribution",
    "SubsystemStatus",
    "SystemHealthSnapshot",
    "TenantSnapshot",
    "VoiceIntelligenceSnapshot",
    "VoicePipelineStage",
    "all_tenant_snapshots",
    "build_agent_flow",
    "build_cost_slo",
    "build_global_kpis",
    "build_healthcare_ops",
    "build_patient_360",
    "build_sentinel_ops",
    "build_system_health",
    "build_tenant_snapshot",
    "build_voice_intelligence",
    "recent_emergencies",
    "recent_escalations",
]


