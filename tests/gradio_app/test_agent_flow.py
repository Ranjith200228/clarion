"""Tests for the Agent Flow view: data_sources rollup + view HTML."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gradio_app import data_sources
from gradio_app.data_sources import AgentFlowSnapshot, FlowNode
from gradio_app.views import agent_flow as view

# ---------- on-disk helpers (mirror test_sentinel_ops) ----------


def _write_report(base: Path, customer_id: str) -> None:
    customer_dir = base / customer_id
    customer_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "customer_id": customer_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario_count": 4,
        "pass_rate": 1.0,
        "metrics": {
            "scenario_count": 4,
            "pass_rate": 1.0,
            "containment_rate": 0.75,
            "booking_accuracy": 1.0,
            "booking_total": 4,
            "booking_correct": 4,
            "hallucination_rate": 0.0,
            "hallucination_with_judge": 0,
            "escalation_precision": 0.8,
            "escalation_recall": 1.0,
            "escalation_f1": 0.89,
            "escalation_accuracy": 0.9,
            "safety_catch_rate": 1.0,
            "safety_total": 2,
            "safety_caught": 2,
            "avg_turns_to_resolution": 1.5,
            "cost_per_request_usd": 0.001,
        },
        "by_difficulty": {},
        "by_intent": {},
        "headline": {},
    }
    (customer_dir / f"report_{customer_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_trace(base: Path, customer_id: str, entries: list[dict]) -> None:
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


def _entry(
    *,
    scenario_id: str,
    customer_id: str = "ophthalmology",
    intent: str = "book",
    actual_outcome: str = "booked",
    tools_called: list[str] | None = None,
    escalation_score: float | None = 0.10,
    escalation_reasons: list[str] | None = None,
    judge_violations: list[str] | None = None,
    judge_hallucination: float | None = None,
    duration_ms: float | None = 600.0,
    cost_usd: float | None = 0.002,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "customer_id": customer_id,
        "trace_id": f"trace_{scenario_id}",
        "difficulty": "clear",
        "intent": intent,
        "agent_replies": ["ok"],
        "tools_called": tools_called or [],
        "actual_outcome": actual_outcome,
        "passed": True,
        "escalation_score": escalation_score,
        "escalation_reasons": escalation_reasons or [],
        "judge_hallucination": judge_hallucination,
        "judge_booking_correct": None,
        "judge_violations": judge_violations or [],
        "duration_ms": duration_ms,
        "cost_usd": cost_usd,
        "input_tokens": 300,
        "output_tokens": 80,
        "step_count": 2,
    }


@pytest.fixture
def flow_data_dir(tmp_path: Path) -> Path:
    """A tenant with four turns exercising the four specialist
    inference paths (booking, eligibility, cancel, emergency)."""
    _write_report(tmp_path, "ophthalmology")
    _write_trace(
        tmp_path,
        "ophthalmology",
        [
            _entry(
                scenario_id="book_001",
                intent="book",
                tools_called=["search_slots", "book_appointment"],
                escalation_score=0.12,
            ),
            _entry(
                scenario_id="elig_001",
                intent="eligibility",
                tools_called=["check_eligibility"],
                escalation_score=0.05,
            ),
            _entry(
                scenario_id="cancel_001",
                intent="cancel",
                tools_called=["cancel_appointment"],
                escalation_score=0.20,
            ),
            _entry(
                scenario_id="emerg_001",
                intent="emergency",
                actual_outcome="escalated_emergency",
                tools_called=[],
                escalation_score=0.95,
                escalation_reasons=["emergency_intent_classified"],
                judge_violations=["unsupported_claim"],
            ),
        ],
    )
    return tmp_path


# ---------- build_agent_flow ----------


def test_build_agent_flow_empty_when_no_trace(tmp_path: Path) -> None:
    flow = data_sources.build_agent_flow("ghost", data_dir=tmp_path)
    assert flow.has_data is False
    assert flow.scenario_id == "—"
    assert flow.nodes == {}
    assert flow.available_turns == []


def test_build_agent_flow_defaults_to_first_entry(flow_data_dir: Path) -> None:
    flow = data_sources.build_agent_flow("ophthalmology", data_dir=flow_data_dir)
    assert flow.has_data is True
    assert flow.scenario_id == "book_001"
    assert flow.chosen_specialist == "Booking"
    assert "Eligibility" in flow.other_specialists
    assert len(flow.other_specialists) == 4
    assert len(flow.available_turns) == 4


def test_build_agent_flow_can_pick_a_specific_scenario(
    flow_data_dir: Path,
) -> None:
    flow = data_sources.build_agent_flow(
        "ophthalmology",
        scenario_id="elig_001",
        data_dir=flow_data_dir,
    )
    assert flow.scenario_id == "elig_001"
    assert flow.chosen_specialist == "Eligibility"


def test_build_agent_flow_picks_cancel_specialist_from_tool(
    flow_data_dir: Path,
) -> None:
    flow = data_sources.build_agent_flow(
        "ophthalmology",
        scenario_id="cancel_001",
        data_dir=flow_data_dir,
    )
    assert flow.chosen_specialist == "Cancel"


def test_build_agent_flow_picks_emergency_over_tools(
    flow_data_dir: Path,
) -> None:
    flow = data_sources.build_agent_flow(
        "ophthalmology",
        scenario_id="emerg_001",
        data_dir=flow_data_dir,
    )
    assert flow.chosen_specialist == "Emergency"
    assert flow.tools_called == []
    # Sentinel node should reflect the high escalation score.
    sentinel = flow.nodes["sentinel"]
    assert sentinel.state == "escalated"
    # Response node should reflect the emergency outcome.
    response = flow.nodes["response"]
    assert response.state == "escalated"


def test_build_agent_flow_distributes_duration_across_nodes(
    flow_data_dir: Path,
) -> None:
    flow = data_sources.build_agent_flow(
        "ophthalmology",
        scenario_id="book_001",
        data_dir=flow_data_dir,
    )
    nodes = flow.nodes
    # Patient has no time attribution (it's the user, not a graph node).
    assert nodes["patient"].ms is None
    # Specialist gets the largest slice (55%).
    assert nodes["specialist"].ms is not None
    assert nodes["router"].ms is not None
    assert nodes["specialist"].ms > nodes["router"].ms
    # Sum across the lit nodes is bounded by duration_ms = 600.
    total = sum(
        n.ms or 0
        for k, n in nodes.items()
        if k in {"router", "specialist", "tools", "sentinel", "response"}
    )
    assert total <= 600


def test_build_agent_flow_falls_back_when_scenario_unknown(
    flow_data_dir: Path,
) -> None:
    flow = data_sources.build_agent_flow(
        "ophthalmology",
        scenario_id="does_not_exist",
        data_dir=flow_data_dir,
    )
    # Falls back to the first entry rather than raising.
    assert flow.has_data is True
    assert flow.scenario_id == "book_001"


# ---------- view: build_html ----------


def _snapshot(*, has_data: bool = True) -> AgentFlowSnapshot:
    nodes = {
        "patient": FlowNode(
            name="PATIENT", state="done", ms=None, cost_usd=None,
            detail="book · clear",
        ),
        "router": FlowNode(
            name="ROUTER", state="done", ms=60, cost_usd=0.0002,
            detail="chose Booking",
        ),
        "specialist": FlowNode(
            name="BOOKING", state="done", ms=330, cost_usd=0.0011,
            detail="2 tools fired",
        ),
        "tools": FlowNode(
            name="TOOLS", state="done", ms=90, cost_usd=None,
            detail="search_slots, book_appointment",
        ),
        "sentinel": FlowNode(
            name="SENTINEL", state="done", ms=60, cost_usd=None,
            detail="score 0.12 · halluc 0.00",
        ),
        "response": FlowNode(
            name="RESPONSE", state="done", ms=60, cost_usd=0.0002,
            detail="booked",
        ),
    } if has_data else {}
    return AgentFlowSnapshot(
        tenant="Ophthalmology",
        scenario_id="book_001" if has_data else "—",
        intent="book",
        user_message="book · clear" if has_data else "",
        nodes=nodes,
        chosen_specialist="Booking" if has_data else "—",
        other_specialists=["Eligibility", "Info", "Cancel", "Emergency"],
        tools_called=["search_slots", "book_appointment"] if has_data else [],
        escalation_score=0.12 if has_data else 0.0,
        escalation_reasons=["frustration=0.20"] if has_data else [],
        judge_violations=[],
        final_outcome="booked" if has_data else "—",
        has_data=has_data,
        available_turns=["book_001"] if has_data else [],
    )


def test_view_routes_to_empty_when_no_data() -> None:
    html = view.build_html(_snapshot(has_data=False))
    assert "Awaiting Data" in html
    assert "python -m clarion.evaluation.cli" in html


def test_view_renders_turn_header_with_outcome_badge() -> None:
    html = view.build_html(_snapshot())
    assert "book_001" in html
    assert "BOOKED" in html
    assert 'data-state="healthy"' in html


def test_view_renders_six_agent_nodes() -> None:
    html = view.build_html(_snapshot())
    # agent_node primitive uses .clarion-agent-node — 6 in the
    # diagram + 5 in the specialist panel = 11 expected.
    assert html.count("clarion-agent-node") >= 6
    for label in ("PATIENT", "ROUTER", "BOOKING", "TOOLS", "SENTINEL", "RESPONSE"):
        assert label in html


def test_view_specialist_panel_marks_one_active_four_idle() -> None:
    html = view.build_html(_snapshot())
    # Active specialist card uses data-state=active; the other four
    # use idle.
    # Render lists 5 cards; the diagram's specialist node uses
    # state=done so it doesn't collide with the panel's active card.
    assert html.count('data-state="active"') >= 1
    assert html.count('data-state="idle"') >= 4


def test_view_tools_panel_lists_each_tool_name() -> None:
    html = view.build_html(_snapshot())
    assert "search_slots" in html
    assert "book_appointment" in html


def test_view_tools_panel_empty_state() -> None:
    snap = _snapshot()
    snap_no_tools = AgentFlowSnapshot(**{**snap.__dict__, "tools_called": []})
    html = view.build_html(snap_no_tools)
    assert "No tool calls fired" in html


def test_view_trust_posture_renders_score_chip() -> None:
    html = view.build_html(_snapshot())
    assert "sentinel score 0.12" in html
    assert "outcome: booked" in html


def test_view_trust_posture_lists_escalation_reasons() -> None:
    html = view.build_html(_snapshot())
    # The reasons chip carries the raw reason string.
    assert "frustration=0.20" in html


def test_view_trust_posture_renders_violations_when_present() -> None:
    snap = _snapshot()
    snap_with_violations = AgentFlowSnapshot(
        **{**snap.__dict__, "judge_violations": ["unsupported_claim"]}
    )
    html = view.build_html(snap_with_violations)
    assert "unsupported_claim" in html


def test_view_empty_html_explicit_tenant_name() -> None:
    html = view.empty_html(tenant="Orthopedics")
    assert "Orthopedics" in html
    assert "orthopedics" in html  # CLI hint lowercases
