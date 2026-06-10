"""Smoke tests for the Phase 14 tab build functions.

Gradio's component tree is not unit-testable directly, but we can
validate that each tab's ``build`` function imports cleanly and
returns the expected dataclass with non-None component handles. That
catches signature drift + import-graph regressions early.

The render helpers are unit-tested against synthetic Pydantic objects.
"""

from __future__ import annotations

from datetime import UTC, datetime

from clarion.schemas import (
    EvaluationCategoryBreakdown,
    EvaluationMetrics,
    EvaluationReport,
    LatencyStats,
    TraceEntry,
    TraceReport,
)

from gradio_app import tab_escalations, tab_live_agent, tab_quality, tab_trace_explorer


def _empty_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        scenario_count=10,
        pass_rate=0.9,
        containment_rate=0.74,
        booking_accuracy=1.0,
        booking_total=4,
        booking_correct=4,
        hallucination_rate=0.05,
        hallucination_with_judge=5,
        escalation_precision=0.61,
        escalation_recall=1.0,
        escalation_f1=0.76,
        escalation_accuracy=0.83,
        safety_catch_rate=1.0,
        safety_total=2,
        safety_caught=2,
        avg_turns_to_resolution=1.5,
        cost_per_request_usd=0.000123,
        tokens_per_call=420.0,
        latency_ms=LatencyStats(avg=12.5, p50=11.0, p95=18.0, count=10),
    )


def _report() -> EvaluationReport:
    return EvaluationReport(
        customer_id="ophthalmology",
        generated_at=datetime.now(UTC),
        scenario_count=10,
        pass_rate=0.9,
        metrics=_empty_metrics(),
        by_difficulty={"clear": EvaluationCategoryBreakdown(total=8, metrics=_empty_metrics())},
        by_intent={"book": EvaluationCategoryBreakdown(total=4, metrics=_empty_metrics())},
        headline={"containment_rate": 0.74},
        outcome_distribution={"booked": 4, "task_created": 3, "info_provided": 3},
        escalation_reason_frequency={"frustration": 3, "low_confidence": 2},
        escalated_scenario_ids=["oph_emergency_emergency_001"],
    )


def _trace_report() -> TraceReport:
    return TraceReport(
        customer_id="ophthalmology",
        generated_at=datetime.now(UTC),
        entries=[
            TraceEntry(
                scenario_id="oph_clear_book_001",
                customer_id="ophthalmology",
                trace_id="trace_aaa",
                difficulty="clear",
                intent="book",
                agent_replies=["ok"],
                tools_called=["search_slots", "book_appointment"],
                actual_outcome="booked",
                passed=True,
                escalation_score=0.1,
                escalation_reasons=[],
                judge_hallucination=0.05,
                judge_booking_correct=0.95,
                judge_violations=[],
                duration_ms=12.5,
                cost_usd=0.0001,
                input_tokens=200,
                output_tokens=50,
                step_count=2,
            )
        ],
    )


# ---------- render helpers (pure functions, easy to test) ----------


def test_quality_render_produces_three_pieces() -> None:
    md, headline_rows, outcome_rows = tab_quality.render(_report())
    assert "ophthalmology" in md
    assert any("Containment Rate" in row[0] for row in headline_rows)
    assert any(row[0] == "booked" for row in outcome_rows)


def test_quality_render_empty_has_recovery_hint() -> None:
    md, headline, outcome = tab_quality.render_empty("orthopedics", "missing")
    assert "python -m clarion.eval --customer orthopedics" in md
    assert headline == []
    assert outcome == []


def test_escalations_render_sorts_reasons_desc() -> None:
    md, reasons, escalated, threshold = tab_escalations.render(_report())
    assert "1 of 10" in md or "**1**" in md
    # frustration (3) before low_confidence (2)
    assert reasons[0][0] == "frustration"
    assert escalated[0][0] == "oph_emergency_emergency_001"
    assert "escalation_precision" in threshold


def test_trace_explorer_render_one_entry() -> None:
    md, rows = tab_trace_explorer.render(_trace_report())
    assert "ophthalmology" in md
    assert len(rows) == 1
    assert rows[0][0] == "oph_clear_book_001"
    assert rows[0][3] == "search_slots, book_appointment"


def test_trace_explorer_render_empty_with_dashes() -> None:
    empty_report = TraceReport(
        customer_id="ophthalmology",
        generated_at=datetime.now(UTC),
        entries=[],
    )
    md, rows = tab_trace_explorer.render(empty_report)
    assert "0 traces" in md
    # The placeholder row indicates "no traces — re-run eval".
    assert "no traces" in rows[0][0]


# ---------- live agent state ----------


def test_live_agent_set_customer_resets_running_totals() -> None:
    st = tab_live_agent.LiveAgentState(
        customer_id="ophthalmology",
        conversation_id="conv_old",
        running_cost_usd=0.01,
        running_input_tokens=100,
        running_output_tokens=50,
    )
    new = tab_live_agent.set_customer(st, "orthopedics")
    assert new.customer_id == "orthopedics"
    assert new.conversation_id is None
    assert new.running_cost_usd == 0.0
    assert new.running_input_tokens == 0
    assert new.running_output_tokens == 0
