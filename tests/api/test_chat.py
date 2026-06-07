"""Tests for POST /chat."""

from __future__ import annotations

from datetime import date

from clarion.agents.llm import FakeLLM, LLMResponse, LLMUsage, ToolCall
from fastapi.testclient import TestClient


def _set_responses(client: TestClient, responses: list[LLMResponse]) -> FakeLLM:
    fake: FakeLLM = client.app.state.fake_llm  # type: ignore[attr-defined]
    fake.responses.extend(responses)
    return fake


# ---------- happy path ----------


def test_chat_returns_reply_and_allocates_conversation_id(client: TestClient) -> None:
    _set_responses(
        client,
        [
            LLMResponse(
                content="Hello!",
                usage=LLMUsage(model="gpt-4o-mini", input_tokens=10, output_tokens=2),
            )
        ],
    )
    r = client.post(
        "/chat",
        json={"customer_id": "ophthalmology", "message": "Hi there"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "Hello!"
    assert body["customer_id"] == "ophthalmology"
    assert body["conversation_id"].startswith("conv_")
    assert body["trace_id"].startswith("trace_")


def test_chat_with_tool_call_returns_human_reply(client: TestClient) -> None:
    _set_responses(
        client,
        [
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="search_slots",
                        arguments={
                            "appointment_type": "Cataract Pre-Op Consult",
                            "on_or_after": date(2026, 6, 1).isoformat(),
                        },
                    ),
                ),
                usage=LLMUsage(model="gpt-4o-mini", input_tokens=200, output_tokens=20),
            ),
            LLMResponse(
                content="I have one slot June 15.",
                usage=LLMUsage(model="gpt-4o-mini", input_tokens=300, output_tokens=10),
            ),
        ],
    )
    r = client.post(
        "/chat",
        json={
            "customer_id": "ophthalmology",
            "message": "I'd like a cataract pre-op consult after June 1.",
        },
    )
    assert r.status_code == 200
    assert "June 15" in r.json()["reply"]


def test_chat_preserves_transcript_across_requests(client: TestClient) -> None:
    _set_responses(
        client,
        [
            LLMResponse(content="First", usage=LLMUsage(model="gpt-4o-mini")),
            LLMResponse(content="Second", usage=LLMUsage(model="gpt-4o-mini")),
        ],
    )
    first = client.post(
        "/chat",
        json={"customer_id": "ophthalmology", "message": "hi"},
    ).json()
    second = client.post(
        "/chat",
        json={
            "customer_id": "ophthalmology",
            "message": "again",
            "conversation_id": first["conversation_id"],
        },
    ).json()
    assert second["reply"] == "Second"
    assert second["conversation_id"] == first["conversation_id"]
    # Trace IDs differ — one per turn.
    assert second["trace_id"] != first["trace_id"]


# ---------- multi-tenant ----------


def test_chat_orthopedics_uses_orthopedics_persona(client: TestClient) -> None:
    fake = _set_responses(client, [LLMResponse(content="ok", usage=LLMUsage(model="gpt-4o-mini"))])
    client.post(
        "/chat",
        json={"customer_id": "orthopedics", "message": "hi"},
    )
    # Find the call we just made.
    messages, _ = fake.calls[-1]
    system = next(m for m in messages if m.role == "system")
    assert "Summit Orthopedic" in (system.content or "")


def test_chat_two_customers_have_separate_transcripts(client: TestClient) -> None:
    _set_responses(
        client,
        [LLMResponse(content="oph reply", usage=LLMUsage(model="gpt-4o-mini"))] * 2
        + [LLMResponse(content="ortho reply", usage=LLMUsage(model="gpt-4o-mini"))],
    )
    oph = client.post(
        "/chat",
        json={"customer_id": "ophthalmology", "message": "hi"},
    ).json()
    ortho = client.post(
        "/chat",
        json={"customer_id": "orthopedics", "message": "hi"},
    ).json()
    assert oph["conversation_id"] != ortho["conversation_id"]


# ---------- guardrails ----------


def test_chat_emergency_short_circuits_without_llm_call(client: TestClient) -> None:
    """Don't script any responses — if the LLM is consulted, FakeLLM
    will raise."""
    r = client.post(
        "/chat",
        json={
            "customer_id": "ophthalmology",
            "message": "I suddenly lost my sight!",
        },
    )
    assert r.status_code == 200
    assert "911" in r.json()["reply"]
    fake: FakeLLM = client.app.state.fake_llm  # type: ignore[attr-defined]
    assert fake.turns_consumed == 0


# ---------- validation + errors ----------


def test_chat_rejects_unknown_customer(client: TestClient) -> None:
    r = client.post(
        "/chat",
        json={"customer_id": "ghost", "message": "hi"},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["detail"]["code"] == "customer_not_found"


def test_chat_rejects_empty_message(client: TestClient) -> None:
    r = client.post(
        "/chat",
        json={"customer_id": "ophthalmology", "message": ""},
    )
    assert r.status_code == 422  # FastAPI Pydantic default


def test_chat_rejects_uppercase_customer_id(client: TestClient) -> None:
    r = client.post(
        "/chat",
        json={"customer_id": "OPH", "message": "hi"},
    )
    assert r.status_code == 422
