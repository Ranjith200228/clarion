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
from datetime import UTC, datetime
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


__all__ = [
    "AgentFlowSnapshot",
    "AuditTailItem",
    "CostBreakdown",
    "EmergencyItem",
    "EscalationItem",
    "FlowNode",
    "FlowPosition",
    "GlobalCostSLO",
    "GlobalKPIs",
    "HealthStatus",
    "JudgeAgreement",
    "LatencyBreakdown",
    "SentinelOpsSnapshot",
    "SignalContribution",
    "TenantSnapshot",
    "all_tenant_snapshots",
    "build_agent_flow",
    "build_cost_slo",
    "build_global_kpis",
    "build_sentinel_ops",
    "build_tenant_snapshot",
    "recent_emergencies",
    "recent_escalations",
]


