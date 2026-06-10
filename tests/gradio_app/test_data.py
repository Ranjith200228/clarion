"""Tests for the Phase 14 data loader."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from clarion.schemas import (
    REPORT_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    EvaluationCategoryBreakdown,
    EvaluationMetrics,
    EvaluationReport,
    LatencyStats,
    TraceEntry,
    TraceReport,
)

from gradio_app import data
from gradio_app.data import SchemaVersionMismatchError


def _empty_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        scenario_count=0,
        pass_rate=0.0,
        containment_rate=0.0,
        booking_accuracy=0.0,
        booking_total=0,
        booking_correct=0,
        hallucination_rate=None,
        hallucination_with_judge=0,
        escalation_precision=0.0,
        escalation_recall=0.0,
        escalation_f1=0.0,
        escalation_accuracy=0.0,
        safety_catch_rate=0.0,
        safety_total=0,
        safety_caught=0,
        avg_turns_to_resolution=0.0,
        cost_per_request_usd=0.0,
        tokens_per_call=0.0,
        latency_ms=LatencyStats(avg=0.0, p50=0.0, p95=0.0, count=0),
    )


def _write_report(path: Path, *, customer_id: str = "ophthalmology") -> EvaluationReport:
    report = EvaluationReport(
        customer_id=customer_id,
        generated_at=datetime.now(UTC),
        scenario_count=0,
        pass_rate=0.0,
        metrics=_empty_metrics(),
        by_difficulty={"clear": EvaluationCategoryBreakdown(total=0, metrics=_empty_metrics())},
        by_intent={"book": EvaluationCategoryBreakdown(total=0, metrics=_empty_metrics())},
        headline={"containment_rate": 0.0},
        outcome_distribution={"booked": 0},
        escalation_reason_frequency={},
        escalated_scenario_ids=[],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), default=str), encoding="utf-8")
    return report


def _write_trace(path: Path, *, customer_id: str = "ophthalmology") -> TraceReport:
    trace = TraceReport(
        customer_id=customer_id,
        generated_at=datetime.now(UTC),
        entries=[
            TraceEntry(
                scenario_id="oph_clear_book_001",
                customer_id=customer_id,
                trace_id="trace_abc",
                difficulty="clear",
                intent="book",
                agent_replies=["ok"],
                tools_called=["search_slots"],
                actual_outcome="booked",
                passed=True,
            )
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace.model_dump(mode="json"), default=str), encoding="utf-8")
    return trace


# ---------- paths ----------


def test_canonical_paths(tmp_path: Path) -> None:
    assert data.report_path("ophthalmology", tmp_path).name == "report_ophthalmology.json"
    assert data.trace_path("orthopedics", tmp_path).name == "trace_orthopedics.json"
    # Per-customer directory layout.
    assert data.report_path("ophthalmology", tmp_path).parent.name == "ophthalmology"


def test_available_customers_falls_back_to_known(tmp_path: Path) -> None:
    """Empty data dir -> the UI still gets the canonical list so the
    dropdown is populated and the user sees an empty-state hint."""
    found = data.available_customers(tmp_path)
    assert "ophthalmology" in found
    assert "orthopedics" in found


def test_available_customers_filters_to_existing(tmp_path: Path) -> None:
    _write_report(tmp_path / "ophthalmology" / "report_ophthalmology.json")
    found = data.available_customers(tmp_path)
    assert found == ["ophthalmology"]


# ---------- typed loaders ----------


def test_load_report_round_trip(tmp_path: Path) -> None:
    written = _write_report(tmp_path / "ophthalmology" / "report_ophthalmology.json")
    loaded = data.load_report("ophthalmology", tmp_path)
    assert loaded.customer_id == written.customer_id
    assert loaded.schema_version == REPORT_SCHEMA_VERSION


def test_load_trace_report_round_trip(tmp_path: Path) -> None:
    written = _write_trace(tmp_path / "ophthalmology" / "trace_ophthalmology.json")
    loaded = data.load_trace_report("ophthalmology", tmp_path)
    assert loaded.customer_id == written.customer_id
    assert loaded.schema_version == TRACE_SCHEMA_VERSION
    assert len(loaded.entries) == 1


def test_load_artifacts_bundles_both(tmp_path: Path) -> None:
    _write_report(tmp_path / "ophthalmology" / "report_ophthalmology.json")
    _write_trace(tmp_path / "ophthalmology" / "trace_ophthalmology.json")
    artifacts = data.load_artifacts("ophthalmology", tmp_path)
    assert artifacts.customer_id == "ophthalmology"
    assert artifacts.report.customer_id == "ophthalmology"
    assert artifacts.trace_report.customer_id == "ophthalmology"


# ---------- error paths ----------


def test_missing_file_raises_with_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        data.load_report("ophthalmology", tmp_path)
    assert "python -m clarion.eval --customer ophthalmology" in str(exc.value)


def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "ophthalmology" / "report_ophthalmology.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Hand-construct a payload with a wrong schema_version so the lock
    # check fires before Pydantic validation.
    raw = {
        "schema_version": "9.9.9",
        "customer_id": "ophthalmology",
        "generated_at": "2026-06-06T00:00:00+00:00",
        "scenario_count": 0,
        "pass_rate": 0.0,
        "metrics": _empty_metrics().model_dump(mode="json"),
        "by_difficulty": {},
        "by_intent": {},
        "headline": {},
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SchemaVersionMismatchError):
        data.load_report("ophthalmology", tmp_path)


def test_non_object_top_level_raises(tmp_path: Path) -> None:
    path = tmp_path / "ophthalmology" / "report_ophthalmology.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ValueError):
        data.load_report("ophthalmology", tmp_path)
