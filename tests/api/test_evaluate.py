"""Tests for POST /evaluate."""

from __future__ import annotations

from datetime import date

from clarion.agents.llm import LLMResponse, LLMUsage, ToolCall
from fastapi.testclient import TestClient


def test_evaluate_runs_scenario_and_returns_metrics(client: TestClient) -> None:
    fake = client.app.state.fake_llm  # type: ignore[attr-defined]
    fake.responses.extend(
        [
            LLMResponse(
                content="hi back",
                usage=LLMUsage(model="gpt-4o-mini", input_tokens=10, output_tokens=2),
            ),
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
                content="One slot on June 15.",
                usage=LLMUsage(model="gpt-4o-mini", input_tokens=320, output_tokens=10),
            ),
        ]
    )

    r = client.post(
        "/evaluate",
        json={
            "customer_id": "ophthalmology",
            "messages": [
                "hi",
                "I'd like a cataract pre-op consult after June 1.",
            ],
            "scenario_id": "smoke_book_cataract",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["scenario_id"] == "smoke_book_cataract"
    assert body["customer_id"] == "ophthalmology"
    assert body["conversation_id"].startswith("conv_")
    assert len(body["transcript"]) == 2
    assert "June 15" in body["transcript"][1]["agent_reply"]
    assert len(body["trace_ids"]) == 2

    m = body["metrics"]
    assert m["turns"] == 2
    assert m["total_steps"] >= 1
    assert m["total_input_tokens"] > 0
    assert m["total_output_tokens"] > 0
    assert m["total_cost_usd"] > 0
    assert m["total_latency_ms"] > 0
    assert m["tools_used"].get("search_slots", 0) == 1


def test_evaluate_rejects_empty_messages(client: TestClient) -> None:
    r = client.post(
        "/evaluate",
        json={"customer_id": "ophthalmology", "messages": []},
    )
    assert r.status_code == 422


def test_evaluate_handles_guardrail_short_circuit_mid_scenario(
    client: TestClient,
) -> None:
    """If a guardrail fires partway through, /evaluate still returns the
    full transcript with the canned reply for the offending turn."""
    fake = client.app.state.fake_llm  # type: ignore[attr-defined]
    fake.responses.extend(
        [
            LLMResponse(
                content="hello",
                usage=LLMUsage(model="gpt-4o-mini", input_tokens=10, output_tokens=2),
            ),
        ]
    )
    r = client.post(
        "/evaluate",
        json={
            "customer_id": "ophthalmology",
            "messages": ["hi", "I think I'm having a stroke"],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["transcript"][0]["agent_reply"] == "hello"
    assert "911" in body["transcript"][1]["agent_reply"]


def test_evaluate_returns_404_for_unknown_customer(client: TestClient) -> None:
    r = client.post(
        "/evaluate",
        json={"customer_id": "ghost", "messages": ["hi"]},
    )
    assert r.status_code == 404
