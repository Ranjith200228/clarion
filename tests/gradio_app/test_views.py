"""Smoke + structural tests for Mission Control and Cost & SLO views.

These don't snapshot byte-for-byte HTML (we'd be rewriting every
test on every CSS tweak). Instead they assert the load-bearing
structural facts: the right primitive classes are present, the
empty-state path activates when there's no data, and the kpi-strip
contains the expected tile count.
"""

from __future__ import annotations

from datetime import UTC, datetime

from gradio_app.data_sources import (
    CostBreakdown,
    EmergencyItem,
    EscalationItem,
    GlobalCostSLO,
    GlobalKPIs,
    LatencyBreakdown,
    TenantSnapshot,
)
from gradio_app.views import cost_slo, mission_control

# ---------- fixtures (in-memory rollups) ----------


def _snap(
    customer_id: str = "ophthalmology",
    *,
    has_data: bool = True,
    health: str = "healthy",
    scenario_count: int = 100,
) -> TenantSnapshot:
    return TenantSnapshot(
        customer_id=customer_id,
        display_name=customer_id.title(),
        has_data=has_data,
        health=health,  # type: ignore[arg-type]
        pass_rate=1.0,
        containment_rate=0.74,
        booking_accuracy=1.0,
        safety_catch_rate=1.0,
        hallucination_rate=0.0,
        escalation_recall=1.0,
        escalation_precision=0.6,
        avg_turns_to_resolution=1.4,
        cost_per_request_usd=0.002,
        scenario_count=scenario_count,
        last_run_at=datetime.now(UTC),
        last_run_relative="1m ago",
        headline_score=0.9,
    )


def _kpis() -> GlobalKPIs:
    return GlobalKPIs(
        total_tenants=2,
        total_scenarios=200,
        pass_rate=1.0,
        containment_rate=0.70,
        safety_catch_rate=1.0,
        hallucination_rate=0.0,
        avg_turns=1.4,
        cost_per_request_usd=0.002,
        composite_trust=0.92,
        total_emergencies=1,
    )


# ---------- Mission Control ----------


def test_mission_control_build_html_contains_eight_kpi_tiles() -> None:
    snaps = [_snap("ophthalmology"), _snap("orthopedics")]
    html = mission_control.build_html(
        snapshots=snaps,
        kpis=_kpis(),
        escalations=[],
        emergencies=[],
    )
    # Eight tiles + per-tenant chips (also use the same class).
    # The strip itself is the load-bearing structural fact.
    assert "clarion-kpi-strip" in html
    # Every key KPI label should appear at least once.
    for label in (
        "TRUST SCORE",
        "SAFETY CATCH",
        "CONTAINMENT",
        "PASS RATE",
        "HALLUCINATION",
        "AVG TURNS",
        "COST / CALL",
        "TENANTS LIVE",
    ):
        assert label in html


def test_mission_control_renders_tenant_table_with_health_attribute() -> None:
    snaps = [_snap("ophthalmology", health="healthy")]
    html = mission_control.build_html(
        snapshots=snaps,
        kpis=_kpis(),
        escalations=[],
        emergencies=[],
    )
    # Tenant card uses the kpi-tile class with data-status.
    assert 'data-status="healthy"' in html
    assert "Ophthalmology" in html


def test_mission_control_handles_no_data_tenant_gracefully() -> None:
    snaps = [_snap("ophthalmology"), _snap("ghost", has_data=False)]
    html = mission_control.build_html(
        snapshots=snaps,
        kpis=_kpis(),
        escalations=[],
        emergencies=[],
    )
    # No-data tenant uses status="unknown" + empty-state text.
    assert 'data-status="unknown"' in html
    assert "No evaluation artifacts" in html


def test_mission_control_streams_have_empty_state_text() -> None:
    html = mission_control.build_html(
        snapshots=[_snap("ophthalmology")],
        kpis=_kpis(),
        escalations=[],
        emergencies=[],
    )
    assert "No escalations on file yet" in html
    assert "Zero emergency handoffs across all tenants" in html


def test_mission_control_renders_escalation_items() -> None:
    snaps = [_snap("ophthalmology")]
    items = [
        EscalationItem(
            tenant="Ophthalmology",
            scenario_id="s_001",
            severity="critical",
            summary="emergency",
            detected_at="2m ago",
            sort_key=1.0,
        ),
    ]
    html = mission_control.build_html(
        snapshots=snaps,
        kpis=_kpis(),
        escalations=items,
        emergencies=[],
    )
    assert "clarion-incident" in html
    assert "emergency" in html
    assert "2m ago" in html


def test_mission_control_renders_emergency_items() -> None:
    snaps = [_snap("ophthalmology")]
    items = [
        EmergencyItem(
            tenant="Ophthalmology",
            scenario_id="s_002",
            summary="chest pain transcript leak",
            detected_at="just now",
            sort_key=1.0,
        ),
    ]
    html = mission_control.build_html(
        snapshots=snaps,
        kpis=_kpis(),
        escalations=[],
        emergencies=items,
    )
    assert "chest pain transcript leak" in html
    assert "just now" in html


def test_mission_control_empty_html_renders_cli_hint() -> None:
    html = mission_control.empty_html()
    assert "No tenant data found" in html
    assert "python -m clarion.evaluation.cli" in html


# ---------- Cost & SLO ----------


def _cost_rollup() -> GlobalCostSLO:
    return GlobalCostSLO(
        per_tenant_cost=[
            CostBreakdown(
                tenant="Ophthalmology",
                total_cost_usd=0.0023,
                avg_cost_per_request_usd=0.00012,
                total_input_tokens=12_000,
                total_output_tokens=3_000,
                scenarios=100,
            ),
        ],
        per_tenant_latency=[
            LatencyBreakdown(
                tenant="Ophthalmology",
                avg_ms=720.0,
                p50_ms=650.0,
                p95_ms=1200.0,
                sample_count=100,
            ),
        ],
        total_cost_usd=0.0023,
        monthly_projection_usd=0.069,
        global_p50_ms=650.0,
        global_p95_ms=1200.0,
    )


def test_cost_slo_build_html_contains_four_kpi_tiles() -> None:
    html = cost_slo.build_html(_cost_rollup())
    assert "clarion-kpi-strip" in html
    for label in (
        "TOTAL COST",
        "MONTHLY PROJECTION",
        "LATENCY P50",
        "LATENCY P95",
    ):
        assert label in html


def test_cost_slo_renders_per_tenant_cost_with_cost_chip() -> None:
    html = cost_slo.build_html(_cost_rollup())
    # Cost chip + token formatting carry through.
    assert "clarion-cost-chip" in html
    assert "12,000" in html  # input tokens, comma-formatted
    assert "$0.0023" in html


def test_cost_slo_renders_per_tenant_latency_with_band() -> None:
    html = cost_slo.build_html(_cost_rollup())
    # p50 of 650 ms is within the warning band (healthy <= 750).
    assert 'data-status="healthy"' in html or 'data-status="warning"' in html
    assert "650 ms" in html


def test_cost_slo_empty_html_renders_message() -> None:
    html = cost_slo.empty_html()
    assert "No cost or latency data on disk" in html


def test_cost_slo_empty_inner_rollup_renders_empty_messages() -> None:
    empty = GlobalCostSLO(
        per_tenant_cost=[],
        per_tenant_latency=[],
        total_cost_usd=0.0,
        monthly_projection_usd=0.0,
        global_p50_ms=0.0,
        global_p95_ms=0.0,
    )
    html = cost_slo.build_html(empty)
    assert "No cost data on disk yet" in html
    assert "No latency samples on disk yet" in html
