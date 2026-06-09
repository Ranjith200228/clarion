"""Tests for the Phase 13 TraceReport sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from clarion.evaluation.trace_report import build_trace_report, write_trace_report
from clarion.schemas import (
    TRACE_SCHEMA_VERSION,
    HarnessReport,
    HarnessResult,
    TraceReport,
)


def _result(**overrides: Any) -> HarnessResult:
    base: dict[str, Any] = {
        "scenario_id": "oph_clear_book_001",
        "customer_id": "ophthalmology",
        "difficulty": "clear",
        "intent": "book",
        "actual_outcome": "booked",
        "actual_tools": ["search_slots", "book_appointment"],
        "escalated": False,
        "agent_replies": ["You're booked for June 15."],
        "trace_ids": ["trace_aaa"],
        "passed": True,
        "failure_reasons": [],
        "judge_verdict": None,
        "escalation": {
            "score": 0.1,
            "signals": {
                "low_confidence": 0.0,
                "repeated_clarification": 0.0,
                "rule_conflict": 0.0,
                "frustration": 0.0,
                "unsupported_request": 0.0,
            },
            "threshold": 0.5,
            "should_escalate": False,
            "reasons": [],
        },
    }
    base.update(overrides)
    return HarnessResult(**base)


def _harness_report(results: list[HarnessResult]) -> HarnessReport:
    return HarnessReport(
        customer_id="ophthalmology",
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        pass_rate=(sum(1 for r in results if r.passed) / len(results) if results else 0.0),
        by_difficulty={},
        by_intent={},
        results=results,
    )


def _traces_jsonl(tmp_path: Path) -> Path:
    """Write a synthetic traces.jsonl with one trace per scenario."""
    path = tmp_path / "traces.jsonl"
    lines = []
    for trace_id, duration, cost, in_tokens, out_tokens, steps in [
        ("trace_aaa", 1500.0, 0.0042, 300, 50, 2),
        ("trace_bbb", 800.0, 0.0021, 150, 25, 1),
    ]:
        lines.append(
            json.dumps(
                {
                    "trace_id": trace_id,
                    "customer_id": "ophthalmology",
                    "spans": [
                        {
                            "name": "agent.chat",
                            "duration_ms": duration,
                            "attributes": {},
                        },
                        {
                            "name": "react.step",
                            "attributes": {},
                            "duration_ms": 0,
                        }
                        if steps >= 1
                        else None,
                        {
                            "name": "llm.complete",
                            "attributes": {
                                "cost_usd": cost,
                                "input_tokens": in_tokens,
                                "output_tokens": out_tokens,
                            },
                            "duration_ms": 0,
                        },
                    ],
                }
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------- shape tests ----------


def test_empty_report_yields_empty_trace_report() -> None:
    report = _harness_report([])
    tr = build_trace_report("ophthalmology", report)
    assert isinstance(tr, TraceReport)
    assert tr.entries == []
    assert tr.customer_id == "ophthalmology"
    assert tr.schema_version == TRACE_SCHEMA_VERSION


def test_one_result_yields_one_entry_without_traces() -> None:
    """When no traces.jsonl is supplied, the entry still has every
    field except duration_ms / cost_usd (which become None)."""
    report = _harness_report([_result()])
    tr = build_trace_report("ophthalmology", report)
    assert len(tr.entries) == 1
    e = tr.entries[0]
    assert e.scenario_id == "oph_clear_book_001"
    assert e.passed is True
    assert e.duration_ms is None
    assert e.cost_usd is None
    assert e.input_tokens == 0
    assert e.escalation_score == 0.1
    assert e.escalation_reasons == []


def test_trace_summary_populates_duration_cost_tokens(tmp_path: Path) -> None:
    traces_path = _traces_jsonl(tmp_path)
    report = _harness_report([_result()])
    tr = build_trace_report("ophthalmology", report, traces_path=traces_path)
    e = tr.entries[0]
    assert e.duration_ms == pytest.approx(1500.0)
    assert e.cost_usd == pytest.approx(0.0042)
    assert e.input_tokens == 300
    assert e.output_tokens == 50
    assert e.step_count == 1


def test_judge_fields_extracted_from_verdict_dict() -> None:
    verdict = {
        "booking_correct": 0.95,
        "hallucination": 0.1,
        "policy_violations": [
            {"kind": "unsupported_claim", "description": "x"},
            {"kind": "invented_provider", "description": "y"},
        ],
        "violation_severity": 0.4,
        "confidence": 0.9,
        "rationale": "looks ok",
    }
    report = _harness_report([_result(judge_verdict=verdict)])
    tr = build_trace_report("ophthalmology", report)
    e = tr.entries[0]
    assert e.judge_booking_correct == pytest.approx(0.95)
    assert e.judge_hallucination == pytest.approx(0.1)
    assert "unsupported_claim" in e.judge_violations
    assert "invented_provider" in e.judge_violations


def test_escalation_score_pulled_from_dict() -> None:
    escalation = {
        "score": 0.85,
        "signals": {
            "low_confidence": 0.8,
            "repeated_clarification": 0.0,
            "rule_conflict": 0.0,
            "frustration": 0.7,
            "unsupported_request": 0.0,
        },
        "threshold": 0.5,
        "should_escalate": True,
        "reasons": ["low_confidence=0.80", "frustration=0.70"],
    }
    report = _harness_report([_result(escalation=escalation)])
    tr = build_trace_report("ophthalmology", report)
    e = tr.entries[0]
    assert e.escalation_score == pytest.approx(0.85)
    assert len(e.escalation_reasons) == 2


# ---------- IO + schema lock ----------


def test_write_and_round_trip_through_pydantic(tmp_path: Path) -> None:
    report = _harness_report([_result()])
    tr = build_trace_report("ophthalmology", report)
    out = tmp_path / "trace_ophthalmology.json"
    written = write_trace_report(tr, out)
    assert written == out
    parsed = json.loads(out.read_text(encoding="utf-8"))
    # Schema lock — the version field is present and matches the constant.
    assert parsed["schema_version"] == TRACE_SCHEMA_VERSION
    assert parsed["customer_id"] == "ophthalmology"
    # Round-trip through the model.
    rehydrated = TraceReport.model_validate(parsed)
    assert rehydrated.schema_version == TRACE_SCHEMA_VERSION
    assert len(rehydrated.entries) == 1
