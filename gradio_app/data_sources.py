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

import logging
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from clarion.schemas import (
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


__all__ = [
    "CostBreakdown",
    "EmergencyItem",
    "EscalationItem",
    "GlobalCostSLO",
    "GlobalKPIs",
    "HealthStatus",
    "LatencyBreakdown",
    "TenantSnapshot",
    "all_tenant_snapshots",
    "build_cost_slo",
    "build_global_kpis",
    "build_tenant_snapshot",
    "recent_emergencies",
    "recent_escalations",
]


