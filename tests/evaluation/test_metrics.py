"""Unit tests for the Phase 12 metric computation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from clarion.evaluation.metrics import (
    compute_evaluation_metrics,
    load_trace_summaries,
)
from clarion.schemas import (
    GroundTruth,
    HarnessReport,
    HarnessResult,
    Scenario,
)

# ---------- fixture builders ----------


def _scenario(
    *,
    scenario_id: str,
    intent: str = "book",
    difficulty: str = "clear",
    expected_outcome: str = "booked",
    should_escalate: bool = False,
    expected_tools: list[str] | None = None,
    expected_appointment_type: str | None = None,
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        customer_id="demo",
        difficulty=difficulty,  # type: ignore[arg-type]
        intent=intent,  # type: ignore[arg-type]
        messages=["hi"],
        ground_truth=GroundTruth(
            expected_outcome=expected_outcome,  # type: ignore[arg-type]
            should_escalate=should_escalate,
            expected_tools=expected_tools or [],
            expected_appointment_type=expected_appointment_type,
        ),
    )


def _result(
    *,
    scenario_id: str,
    intent: str = "book",
    difficulty: str = "clear",
    actual_outcome: str = "booked",
    actual_tools: list[str] | None = None,
    escalated: bool = False,
    passed: bool = True,
    trace_ids: list[str] | None = None,
    judge_verdict: dict[str, Any] | None = None,
    escalation: dict[str, Any] | None = None,
) -> HarnessResult:
    return HarnessResult(
        scenario_id=scenario_id,
        customer_id="demo",
        difficulty=difficulty,  # type: ignore[arg-type]
        intent=intent,  # type: ignore[arg-type]
        actual_outcome=actual_outcome,  # type: ignore[arg-type]
        actual_tools=actual_tools or [],
        escalated=escalated,
        agent_replies=["ok"],
        trace_ids=trace_ids or [],
        passed=passed,
        failure_reasons=[],
        judge_verdict=judge_verdict,
        escalation=escalation,
    )


def _report(results: list[HarnessResult]) -> HarnessReport:
    passed = sum(1 for r in results if r.passed)
    return HarnessReport(
        customer_id="demo",
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        pass_rate=passed / len(results) if results else 0.0,
        by_difficulty={},
        by_intent={},
        results=results,
    )


# ---------- empty / null cases ----------


def test_empty_results_returns_empty_metrics() -> None:
    metrics = compute_evaluation_metrics([], _report([]))
    assert metrics.scenario_count == 0
    assert metrics.pass_rate == 0.0
    assert metrics.containment_rate == 0.0
    assert metrics.booking_accuracy == 0.0
    assert metrics.hallucination_rate is None
    assert metrics.latency_ms is None


def test_no_judge_anywhere_yields_none_hallucination() -> None:
    scenario = _scenario(scenario_id="s1")
    result = _result(scenario_id="s1", judge_verdict=None)
    metrics = compute_evaluation_metrics([scenario], _report([result]))
    assert metrics.hallucination_rate is None
    assert metrics.hallucination_with_judge == 0


# ---------- containment ----------


def test_containment_counts_contained_outcomes_only() -> None:
    scenarios = [
        _scenario(scenario_id="s1", intent="book"),
        _scenario(scenario_id="s2", intent="cancel", expected_outcome="cancelled"),
        _scenario(scenario_id="s3", intent="faq", expected_outcome="info_provided"),
        _scenario(scenario_id="s4", intent="emergency", expected_outcome="escalated_emergency"),
    ]
    results = [
        _result(scenario_id="s1", actual_outcome="booked"),
        _result(scenario_id="s2", actual_outcome="cancelled"),
        _result(scenario_id="s3", actual_outcome="info_provided"),
        _result(scenario_id="s4", actual_outcome="escalated_emergency", escalated=True),
    ]
    metrics = compute_evaluation_metrics(scenarios, _report(results))
    # 3 of 4 contained.
    assert metrics.containment_rate == 0.75


# ---------- booking accuracy ----------


def test_booking_accuracy_only_over_booking_scenarios() -> None:
    """A FAQ scenario answering info_provided shouldn't count toward
    booking accuracy at all."""
    scenarios = [
        _scenario(scenario_id="s1", intent="book", expected_outcome="booked"),
        _scenario(scenario_id="s2", intent="book", expected_outcome="booked"),
        _scenario(scenario_id="s3", intent="faq", expected_outcome="info_provided"),
    ]
    results = [
        _result(scenario_id="s1", actual_outcome="booked", passed=True),
        _result(scenario_id="s2", actual_outcome="booked", passed=False),  # failure
        _result(scenario_id="s3", actual_outcome="info_provided", passed=True),
    ]
    metrics = compute_evaluation_metrics(scenarios, _report(results))
    assert metrics.booking_total == 2
    assert metrics.booking_correct == 1
    assert metrics.booking_accuracy == 0.5


# ---------- hallucination ----------


def test_hallucination_averages_judge_values() -> None:
    scenarios = [_scenario(scenario_id=f"s{i}") for i in range(3)]
    results = [
        _result(scenario_id="s0", judge_verdict={"hallucination": 0.1}),
        _result(scenario_id="s1", judge_verdict={"hallucination": 0.3}),
        _result(scenario_id="s2", judge_verdict={"hallucination": 0.5}),
    ]
    metrics = compute_evaluation_metrics(scenarios, _report(results))
    assert metrics.hallucination_with_judge == 3
    assert metrics.hallucination_rate == pytest.approx(0.3, rel=1e-9)


def test_hallucination_ignores_results_without_judge() -> None:
    scenarios = [_scenario(scenario_id=f"s{i}") for i in range(3)]
    results = [
        _result(scenario_id="s0", judge_verdict={"hallucination": 0.2}),
        _result(scenario_id="s1", judge_verdict=None),
        _result(scenario_id="s2", judge_verdict={"hallucination": 0.8}),
    ]
    metrics = compute_evaluation_metrics(scenarios, _report(results))
    assert metrics.hallucination_with_judge == 2
    assert metrics.hallucination_rate == pytest.approx(0.5, rel=1e-9)


# ---------- escalation ----------


def test_escalation_pr_passes_through_phase11_compute_stats() -> None:
    scenarios = [
        _scenario(scenario_id="s1", should_escalate=True),
        _scenario(scenario_id="s2", should_escalate=False),
        _scenario(scenario_id="s3", should_escalate=True),
    ]
    results = [
        _result(scenario_id="s1", escalation={"should_escalate": True}),  # TP
        _result(scenario_id="s2", escalation={"should_escalate": False}),  # TN
        _result(scenario_id="s3", escalation={"should_escalate": False}),  # FN
    ]
    metrics = compute_evaluation_metrics(scenarios, _report(results))
    # TP=1 FP=0 TN=1 FN=1 -> precision=1.0 recall=0.5 f1=0.6667
    assert metrics.escalation_precision == pytest.approx(1.0, abs=1e-3)
    assert metrics.escalation_recall == pytest.approx(0.5, abs=1e-3)
    assert metrics.escalation_f1 == pytest.approx(0.6667, abs=1e-3)


def test_escalation_metrics_skip_results_without_score() -> None:
    scenarios = [
        _scenario(scenario_id="s1", should_escalate=True),
        _scenario(scenario_id="s2", should_escalate=True),
    ]
    results = [
        _result(scenario_id="s1", escalation={"should_escalate": True}),  # TP
        _result(scenario_id="s2", escalation=None),  # skipped — no score
    ]
    metrics = compute_evaluation_metrics(scenarios, _report(results))
    # Only 1 prediction (TP), so precision=recall=1.0
    assert metrics.escalation_precision == 1.0
    assert metrics.escalation_recall == 1.0


# ---------- safety catch ----------


def test_safety_catch_rate_only_over_safety_intents() -> None:
    scenarios = [
        _scenario(scenario_id="s1", intent="emergency", expected_outcome="escalated_emergency"),
        _scenario(scenario_id="s2", intent="clinical_advice", expected_outcome="refused_clinical"),
        _scenario(scenario_id="s3", intent="book"),
    ]
    results = [
        _result(scenario_id="s1", intent="emergency", passed=True),
        _result(scenario_id="s2", intent="clinical_advice", passed=False),  # missed
        _result(scenario_id="s3", intent="book", passed=True),
    ]
    metrics = compute_evaluation_metrics(scenarios, _report(results))
    assert metrics.safety_total == 2
    assert metrics.safety_caught == 1
    assert metrics.safety_catch_rate == 0.5


# ---------- latency + cost ----------


def test_latency_and_cost_pulled_from_traces(tmp_path: Path) -> None:
    """When a traces.jsonl is present, durations and cost roll up; the
    avg/p50/p95 percentiles use the canonical values."""
    traces_path = tmp_path / "traces.jsonl"
    # Three traces, each with one agent.chat root, one react.step, one
    # llm.complete. Durations 100, 200, 300 -> avg=200, p50=200, p95=290.
    payloads = []
    for i, (dur, cost) in enumerate([(100, 0.001), (200, 0.002), (300, 0.003)]):
        payloads.append(
            {
                "trace_id": f"trace_{i}",
                "spans": [
                    {"name": "agent.chat", "duration_ms": dur, "attributes": {}},
                    {"name": "react.step", "attributes": {}},
                    {
                        "name": "llm.complete",
                        "attributes": {"cost_usd": cost, "input_tokens": 10, "output_tokens": 2},
                    },
                ],
            }
        )
    traces_path.write_text("\n".join(json.dumps(p) for p in payloads) + "\n", encoding="utf-8")

    scenarios = [_scenario(scenario_id=f"s{i}") for i in range(3)]
    results = [_result(scenario_id=f"s{i}", trace_ids=[f"trace_{i}"]) for i in range(3)]

    metrics = compute_evaluation_metrics(scenarios, _report(results), traces_path=traces_path)
    assert metrics.latency_ms is not None
    assert metrics.latency_ms.count == 3
    assert metrics.latency_ms.avg == pytest.approx(200.0, abs=1e-3)
    assert metrics.latency_ms.p50 == pytest.approx(200.0, abs=1e-3)
    assert metrics.latency_ms.p95 == pytest.approx(290.0, abs=1e-3)

    # Cost per request = 0.006 / 3 = 0.002
    assert metrics.cost_per_request_usd == pytest.approx(0.002, abs=1e-6)

    # Each trace has one react.step -> avg turns = 1.0
    assert metrics.avg_turns_to_resolution == pytest.approx(1.0, abs=1e-3)


def test_no_traces_file_keeps_latency_none(tmp_path: Path) -> None:
    scenarios = [_scenario(scenario_id="s1")]
    results = [_result(scenario_id="s1", trace_ids=["trace_x"])]
    metrics = compute_evaluation_metrics(
        scenarios, _report(results), traces_path=tmp_path / "missing.jsonl"
    )
    assert metrics.latency_ms is None
    assert metrics.cost_per_request_usd == 0.0
    assert metrics.avg_turns_to_resolution == 0.0


def test_malformed_trace_lines_are_skipped(tmp_path: Path) -> None:
    """A corrupt JSONL line shouldn't nuke the run."""
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_text(
        "not json at all\n"
        + json.dumps(
            {
                "trace_id": "trace_ok",
                "spans": [
                    {"name": "agent.chat", "duration_ms": 100, "attributes": {}},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summaries = load_trace_summaries(traces_path)
    assert "trace_ok" in summaries
    assert summaries["trace_ok"]["duration_ms"] == 100.0
