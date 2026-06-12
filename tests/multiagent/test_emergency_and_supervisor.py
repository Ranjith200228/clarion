"""Tests for the Supervisor decision tree + EmergencySpecialist."""

from __future__ import annotations

from clarion.agents.llm import Message
from clarion.multiagent import EmergencySpecialist, Supervisor
from clarion.multiagent.state import initial_state
from clarion.multiagent.supervisor import (
    ESCALATION_HANDOFF_TEXT,
    route_after_supervisor,
)

# ---------- EmergencySpecialist ----------


def test_emergency_specialist_short_circuits_with_canned_reply() -> None:
    spec = EmergencySpecialist(
        llm=None,  # type: ignore[arg-type]
        customer=None,  # type: ignore[arg-type]
        ctx=None,  # type: ignore[arg-type]
    )
    state = initial_state(user_message="chest pain", transcript=[])
    out = spec(state)
    assert "911" in out["assistant_text"]
    assert out["escalated"] is True
    assert "emergency_intent_classified" in out["escalation_reasons"]
    # Outbound message must be assistant role with the canned text.
    msgs = out["messages"]
    assert len(msgs) == 1 and msgs[0].role == "assistant"


# ---------- Supervisor ----------


def test_supervisor_finishes_when_specialist_text_is_fine() -> None:
    # No scorer attached -> escalation path #3 is skipped; assistant_text
    # exists and no prior escalated flag -> default FINISH.
    sup = Supervisor(scorer=None)
    state = initial_state(user_message="hi", transcript=[])
    state["assistant_text"] = "Your appointment is confirmed."
    out = sup(state)
    assert out["decision"] == "finish"
    assert out["visits"] == 1  # bumped from 0


def test_supervisor_honors_specialist_escalation_without_text_override() -> None:
    sup = Supervisor(scorer=None)
    state = initial_state(user_message="help", transcript=[])
    state["assistant_text"] = "Please call 911 immediately."
    state["escalated"] = True
    state["escalation_reasons"] = ["emergency_intent_classified"]
    out = sup(state)
    assert out["decision"] == "escalate"
    # We did NOT clobber the specialist's text — the 911 reply stands.
    assert "assistant_text" not in out
    assert "messages" not in out


def test_supervisor_escalates_when_visit_count_exceeds_max() -> None:
    sup = Supervisor(scorer=None, max_visits=2)
    state = initial_state(user_message="please help", transcript=[])
    state["assistant_text"] = "Let me try again."
    state["visits"] = 2  # already at the cap; supervisor will see 3 > 2
    out = sup(state)
    assert out["decision"] == "escalate"
    assert out["escalated"] is True
    assert any("router_loop_exceeded" in r for r in out["escalation_reasons"])
    # Supervisor wrote the handoff text on this path.
    assert out["assistant_text"] == ESCALATION_HANDOFF_TEXT


# ---------- route_after_supervisor ----------


def test_route_after_supervisor_finish_maps_to_END() -> None:
    state = initial_state(user_message="x", transcript=[])
    state["decision"] = "finish"
    assert route_after_supervisor(state) == "END"


def test_route_after_supervisor_route_maps_to_router() -> None:
    state = initial_state(user_message="x", transcript=[])
    state["decision"] = "route"
    assert route_after_supervisor(state) == "router"


def test_route_after_supervisor_escalate_terminates() -> None:
    state = initial_state(user_message="x", transcript=[])
    state["decision"] = "escalate"
    assert route_after_supervisor(state) == "END"


# ---------- transcript-aware scorer signal harvesting ----------


def test_supervisor_facts_collect_user_and_assistant_turns() -> None:
    sup = Supervisor(scorer=None)
    transcript = [
        Message.user("first turn"),
        Message.assistant(text="we can do that"),
        Message.user("second turn"),
    ]
    state = initial_state(user_message="third turn", transcript=transcript)
    state["assistant_text"] = "third reply"
    facts = sup._facts_for(state)
    # All three user turns + both assistant turns end up in the facts.
    assert facts.user_messages == ["first turn", "second turn", "third turn"]
    assert facts.agent_replies == ["we can do that", "third reply"]
    assert facts.judge is None
    assert facts.already_escalated is False
