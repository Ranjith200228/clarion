"""Tests for GET /health."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["customers_loaded"] == []  # nothing touched yet


def test_health_customers_loaded_grows_after_chat(client: TestClient) -> None:
    # Trigger a customer load via /chat first.
    from clarion.agents.llm import LLMResponse

    fake = client.app.state.fake_llm  # type: ignore[attr-defined]
    fake.responses.append(LLMResponse(content="hi"))
    chat_resp = client.post(
        "/chat",
        json={"customer_id": "ophthalmology", "message": "hi"},
    )
    assert chat_resp.status_code == 200

    health = client.get("/health").json()
    assert "ophthalmology" in health["customers_loaded"]
