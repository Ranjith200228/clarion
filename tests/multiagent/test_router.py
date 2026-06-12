"""Tests for clarion.multiagent.router."""

from __future__ import annotations

from clarion.agents.llm import FakeLLM, LLMResponse, Message, ToolCall
from clarion.multiagent import HeuristicIntentRouter, LLMIntentRouter
from clarion.multiagent.state import initial_state

# ---------- HeuristicIntentRouter ----------


def test_heuristic_router_classifies_emergency_first() -> None:
    r = HeuristicIntentRouter()
    state = initial_state(
        user_message="I had a stroke yesterday — please reschedule my appointment",
        transcript=[],
    )
    out = r(state)
    # Emergency wins over booking even when both signals are present.
    assert out["intent"] == "emergency"


def test_heuristic_router_recognizes_booking() -> None:
    r = HeuristicIntentRouter()
    out = r(initial_state(user_message="I'd like to book an eye exam", transcript=[]))
    assert out["intent"] == "booking"


def test_heuristic_router_recognizes_eligibility() -> None:
    r = HeuristicIntentRouter()
    out = r(initial_state(user_message="Do you accept Aetna?", transcript=[]))
    assert out["intent"] == "eligibility"


def test_heuristic_router_recognizes_cancel() -> None:
    r = HeuristicIntentRouter()
    out = r(initial_state(user_message="Please cancel my appointment", transcript=[]))
    assert out["intent"] == "cancel"


def test_heuristic_router_defaults_to_info() -> None:
    r = HeuristicIntentRouter()
    out = r(initial_state(user_message="What are your hours?", transcript=[]))
    assert out["intent"] == "info"


# ---------- LLMIntentRouter ----------


def test_llm_router_reads_tool_call() -> None:
    fake = FakeLLM(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call_1",
                        name="route",
                        arguments={"intent": "booking", "rationale": "wants a slot"},
                    ),
                ),
            )
        ]
    )
    r = LLMIntentRouter(llm=fake)
    out = r(initial_state(user_message="book me", transcript=[]))
    assert out["intent"] == "booking"
    # Confirm the router actually advertised the `route` tool.
    sent_messages, sent_tools = fake.calls[0]
    assert isinstance(sent_messages[0], Message) and sent_messages[0].role == "system"
    assert any(t.name == "route" for t in sent_tools)


def test_llm_router_falls_back_to_info_on_parse_failure() -> None:
    fake = FakeLLM(
        responses=[
            LLMResponse(content="I don't know how to classify this", tool_calls=())
        ]
    )
    r = LLMIntentRouter(llm=fake)
    out = r(initial_state(user_message="???", transcript=[]))
    # Never fall back to emergency on parse error — that would be safety theater.
    assert out["intent"] == "info"


def test_llm_router_ignores_unknown_intent_string() -> None:
    fake = FakeLLM(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(id="call_1", name="route", arguments={"intent": "nonsense"}),
                ),
            )
        ]
    )
    r = LLMIntentRouter(llm=fake)
    out = r(initial_state(user_message="x", transcript=[]))
    assert out["intent"] == "info"
