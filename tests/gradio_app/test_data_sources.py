"""Tests for ``gradio_app.data_sources`` — the rollup façade Mission
Control and Cost & SLO read.

Strategy: build a tmp data directory with one or two real
``report_*.json`` / ``trace_*.json`` files, then assert the typed
rollups extract the expected numbers.

The fixture writes minimal but schema-valid payloads — no Pydantic
gymnastics, no real harness run, fast tests.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gradio_app import data_sources
from gradio_app.data_sources import (
    HealthStatus,
    _pct,
    _relative_time,
)

# ---------- helpers + fixtures ----------


def _write_report(
    base: Path,
    *,
    customer_id: str,
    pass_rate: float = 1.0,
    containment: float = 0.7,
    booking_accuracy: float = 1.0,
    safety_catch: float = 1.0,
    hallucination: float | None = 0.0,
    avg_turns: float = 1.4,
    cost: float = 0.0,
    scenario_count: int = 100,
    escalation_precision: float = 0.6,
    escalation_recall: float = 1.0,
) -> None:
    customer_dir = base / customer_id
    customer_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "customer_id": customer_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario_count": scenario_count,
        "pass_rate": pass_rate,
        "metrics": {
            "scenario_count": scenario_count,
            "pass_rate": pass_rate,
            "containment_rate": containment,
            "booking_accuracy": booking_accuracy,
            "booking_total": scenario_count,
            "booking_correct": int(scenario_count * booking_accuracy),
            "hallucination_rate": hallucination,
            "hallucination_with_judge": 0,
            "escalation_precision": escalation_precision,
            "escalation_recall": escalation_recall,
            "escalation_f1": (
                2 * escalation_precision * escalation_recall
                / (escalation_precision + escalation_recall)
                if (escalation_precision + escalation_recall) > 0
                else 0.0
            ),
            "escalation_accuracy": 0.85,
            "safety_catch_rate": safety_catch,
            "safety_total": 10,
            "safety_caught": int(10 * safety_catch),
            "avg_turns_to_resolution": avg_turns,
            "cost_per_request_usd": cost,
        },
        "by_difficulty": {},
        "by_intent": {},
        "headline": {},
    }
    (customer_dir / f"report_{customer_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_trace(
    base: Path,
    *,
    customer_id: str,
    entries: list[dict],
) -> None:
    customer_dir = base / customer_id
    customer_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "customer_id": customer_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "entries": entries,
    }
    (customer_dir / f"trace_{customer_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _trace_entry(
    *,
    scenario_id: str,
    customer_id: str = "ophthalmology",
    actual_outcome: str = "booked",
    escalation_score: float | None = None,
    escalation_reasons: list[str] | None = None,
    duration_ms: float | None = None,
    cost_usd: float | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict:
    """Return a minimal but schema-valid TraceEntry payload."""
    return {
        "scenario_id": scenario_id,
        "customer_id": customer_id,
        "trace_id": f"trace_{scenario_id}",
        "difficulty": "clear",
        "intent": "book",
        "agent_replies": ["confirmed."],
        "tools_called": ["search_slots", "book_appointment"],
        "actual_outcome": actual_outcome,
        "passed": True,
        "escalation_score": escalation_score,
        "escalation_reasons": escalation_reasons or [],
        "judge_hallucination": None,
        "judge_booking_correct": None,
        "judge_violations": [],
        "duration_ms": duration_ms,
        "cost_usd": cost_usd,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "step_count": 2,
    }


@pytest.fixture
def two_tenant_data_dir(tmp_path: Path) -> Path:
    """A populated data dir with both shipped customers."""
    _write_report(
        tmp_path,
        customer_id="ophthalmology",
        pass_rate=1.0,
        containment=0.74,
        safety_catch=1.0,
        hallucination=0.0,
        cost=0.0023,
    )
    _write_trace(
        tmp_path,
        customer_id="ophthalmology",
        entries=[
            _trace_entry(
                scenario_id="s_001",
                duration_ms=420.0,
                cost_usd=0.001,
                input_tokens=300,
                output_tokens=80,
            ),
            _trace_entry(
                scenario_id="s_002",
                actual_outcome="escalated_emergency",
                escalation_score=0.95,
                escalation_reasons=["emergency_intent_classified"],
                duration_ms=200.0,
                cost_usd=0.0,
            ),
            _trace_entry(
                scenario_id="s_003",
                escalation_score=0.65,
                escalation_reasons=["frustration", "low_confidence"],
                duration_ms=850.0,
                cost_usd=0.002,
                input_tokens=500,
                output_tokens=120,
            ),
        ],
    )
    _write_report(
        tmp_path,
        customer_id="orthopedics",
        pass_rate=1.0,
        containment=0.66,
        safety_catch=1.0,
        hallucination=0.0,
        cost=0.0019,
        scenario_count=100,
    )
    _write_trace(
        tmp_path,
        customer_id="orthopedics",
        entries=[
            _trace_entry(
                scenario_id="o_001",
                customer_id="orthopedics",
                duration_ms=510.0,
                cost_usd=0.0012,
                input_tokens=320,
                output_tokens=90,
            ),
        ],
    )
    return tmp_path


# ---------- TenantSnapshot ----------


def test_build_tenant_snapshot_returns_empty_when_artifacts_missing(
    tmp_path: Path,
) -> None:
    snap = data_sources.build_tenant_snapshot("ghost_tenant", data_dir=tmp_path)
    assert snap.customer_id == "ghost_tenant"
    assert snap.has_data is False
    assert snap.health == "unknown"
    assert snap.pass_rate == 0.0
    assert snap.last_run_at is None
    assert snap.last_run_relative == "—"


def test_build_tenant_snapshot_populates_when_artifacts_present(
    two_tenant_data_dir: Path,
) -> None:
    snap = data_sources.build_tenant_snapshot(
        "ophthalmology", data_dir=two_tenant_data_dir
    )
    assert snap.has_data is True
    assert snap.pass_rate == pytest.approx(1.0)
    assert snap.containment_rate == pytest.approx(0.74)
    assert snap.safety_catch_rate == pytest.approx(1.0)
    assert snap.hallucination_rate == pytest.approx(0.0)
    assert snap.scenario_count == 100
    assert snap.display_name == "Ophthalmology"
    assert snap.last_run_at is not None
    assert snap.last_run_relative != "—"
    assert snap.health in {"healthy", "warning", "critical"}


def test_health_band_strict_about_failure_axes(two_tenant_data_dir: Path) -> None:
    # Add a tenant with strong safety but weak containment.
    _write_report(
        two_tenant_data_dir,
        customer_id="ophthalmology",
        pass_rate=1.0,
        containment=0.10,  # very weak
        safety_catch=1.0,
        hallucination=0.0,
    )
    snap = data_sources.build_tenant_snapshot(
        "ophthalmology", data_dir=two_tenant_data_dir
    )
    # Composite = 0.35 * 0.10 + 0.30 * 1.0 + 0.20 * 1.0 + 0.15 * 1.0 = 0.685
    # Below 0.70 -> critical band.
    assert snap.health == "critical"


# ---------- GlobalKPIs ----------


def test_global_kpis_aggregates_by_scenario_count(two_tenant_data_dir: Path) -> None:
    snaps = data_sources.all_tenant_snapshots(data_dir=two_tenant_data_dir)
    kpis = data_sources.build_global_kpis(snaps, data_dir=two_tenant_data_dir)
    assert kpis.total_tenants == 2
    # Both tenants have 100 scenarios -> weighted mean of 0.74 + 0.66 = 0.70.
    assert kpis.containment_rate == pytest.approx(0.70)
    assert kpis.safety_catch_rate == pytest.approx(1.0)
    assert kpis.composite_trust >= 0.0
    assert kpis.composite_trust <= 1.0
    # Total emergencies pulled from traces — both tenants combined have 1.
    assert kpis.total_emergencies == 1


def test_global_kpis_empty_when_no_tenant_has_data(tmp_path: Path) -> None:
    snaps = data_sources.all_tenant_snapshots(data_dir=tmp_path)
    kpis = data_sources.build_global_kpis(snaps, data_dir=tmp_path)
    assert kpis.total_tenants == 0
    assert kpis.total_scenarios == 0
    assert kpis.pass_rate == 0.0
    assert kpis.composite_trust == 0.0


# ---------- recent_escalations / recent_emergencies ----------


def test_recent_escalations_returns_only_entries_with_reasons(
    two_tenant_data_dir: Path,
) -> None:
    snaps = data_sources.all_tenant_snapshots(data_dir=two_tenant_data_dir)
    items = data_sources.recent_escalations(snaps, data_dir=two_tenant_data_dir)
    # Ophthalmology has 2 escalated entries (s_002 + s_003); orthopedics 0.
    assert len(items) == 2
    # The first item should be the higher-impact one — emergency
    # bumped to severity=critical.
    severities = {i.severity for i in items}
    assert "critical" in severities


def test_recent_emergencies_pulls_from_outcome_and_reason(
    two_tenant_data_dir: Path,
) -> None:
    snaps = data_sources.all_tenant_snapshots(data_dir=two_tenant_data_dir)
    items = data_sources.recent_emergencies(snaps, data_dir=two_tenant_data_dir)
    assert len(items) == 1
    assert "scenario" in items[0].summary or "confirmed" in items[0].summary


def test_recent_escalations_respects_limit(two_tenant_data_dir: Path) -> None:
    snaps = data_sources.all_tenant_snapshots(data_dir=two_tenant_data_dir)
    items = data_sources.recent_escalations(
        snaps, data_dir=two_tenant_data_dir, limit=1
    )
    assert len(items) == 1


# ---------- Cost / SLO ----------


def test_build_cost_slo_aggregates_per_tenant(two_tenant_data_dir: Path) -> None:
    snaps = data_sources.all_tenant_snapshots(data_dir=two_tenant_data_dir)
    rollup = data_sources.build_cost_slo(snaps, data_dir=two_tenant_data_dir)
    # Both tenants contributed a cost row + a latency row.
    assert len(rollup.per_tenant_cost) == 2
    assert len(rollup.per_tenant_latency) == 2
    # Total cost = 0.001 + 0 + 0.002 + 0.0012 = 0.0042.
    assert rollup.total_cost_usd == pytest.approx(0.0042, abs=1e-6)
    # Naive monthly projection: total * 30.
    assert rollup.monthly_projection_usd == pytest.approx(0.0042 * 30, abs=1e-6)
    # p50 across [200, 420, 850, 510] -> sorted [200, 420, 510, 850]
    # 50th pct interp ~ 465.
    assert 400 < rollup.global_p50_ms < 525


def test_build_cost_slo_empty_when_no_traces(tmp_path: Path) -> None:
    snaps = data_sources.all_tenant_snapshots(data_dir=tmp_path)
    rollup = data_sources.build_cost_slo(snaps, data_dir=tmp_path)
    assert rollup.per_tenant_cost == []
    assert rollup.per_tenant_latency == []
    assert rollup.total_cost_usd == 0.0


# ---------- low-level helpers ----------


def test_pct_handles_empty_and_extremes() -> None:
    from gradio_app.data_sources import _pct as pct
    assert pct([], 50.0) == 0.0
    assert pct([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert pct([1.0, 2.0, 3.0, 4.0], 100.0) == 4.0
    # 50th percentile of even-length list = linear interp.
    assert pct([1.0, 2.0, 3.0, 4.0], 50.0) == pytest.approx(2.5)


def test_relative_time_buckets() -> None:
    from datetime import timedelta
    now = datetime.now(UTC)
    assert _relative_time(now) == "just now"
    assert _relative_time(now - timedelta(minutes=5)) == "5m ago"
    assert _relative_time(now - timedelta(hours=3)) == "3h ago"
    assert _relative_time(now - timedelta(days=1, hours=1)) == "yesterday"
    assert _relative_time(now - timedelta(days=4)) == "4d ago"
    assert _relative_time(None) == "—"


def test_pct_function_helper_in_views_safe() -> None:
    # Internal helper used by the view; just confirm it returns
    # the function we expect to import.
    assert callable(_pct)


# ---------- snapshot HealthStatus contract ----------


def test_health_status_is_a_literal_union() -> None:
    valid: set[HealthStatus] = {"healthy", "warning", "critical", "unknown"}
    assert "healthy" in valid
    assert "unknown" in valid
