"""Tests for the token-bucket rate limiter + the gating middleware."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from clarion.agents.llm import FakeLLM, LLMClient, LLMResponse
from clarion.config import Settings
from fastapi.testclient import TestClient

from api.app import create_app
from api.middleware import TokenBucketLimiter
from api.sessions import make_session_manager

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


# ---------- TokenBucketLimiter unit tests ----------


def test_first_request_is_always_allowed() -> None:
    lim = TokenBucketLimiter(rps=1.0, burst=3, clock=lambda: 0.0)
    assert lim.acquire(customer_id="ophthalmology", ip="1.1.1.1") is True


def test_drains_after_burst_then_refills_with_elapsed_time() -> None:
    fake_now = [0.0]
    lim = TokenBucketLimiter(rps=2.0, burst=3, clock=lambda: fake_now[0])
    for _ in range(3):
        assert lim.acquire(customer_id="c", ip="1.1.1.1") is True
    # Bucket empty.
    assert lim.acquire(customer_id="c", ip="1.1.1.1") is False
    # 0.5s later -> 1 new token (rps=2).
    fake_now[0] = 0.5
    assert lim.acquire(customer_id="c", ip="1.1.1.1") is True
    assert lim.acquire(customer_id="c", ip="1.1.1.1") is False


def test_buckets_are_isolated_by_customer_and_ip() -> None:
    lim = TokenBucketLimiter(rps=1.0, burst=2, clock=lambda: 0.0)
    # Drain customer A.
    assert lim.acquire(customer_id="A", ip="1.1.1.1") is True
    assert lim.acquire(customer_id="A", ip="1.1.1.1") is True
    assert lim.acquire(customer_id="A", ip="1.1.1.1") is False
    # B has its own bucket.
    assert lim.acquire(customer_id="B", ip="1.1.1.1") is True
    # Different IP for A is a separate bucket too.
    assert lim.acquire(customer_id="A", ip="2.2.2.2") is True


def test_retry_after_returns_at_least_one_second() -> None:
    lim = TokenBucketLimiter(rps=10.0, burst=1, clock=lambda: 0.0)
    lim.acquire(customer_id="c", ip="1.1.1.1")
    # Bucket empty; rps=10 means a token in 100 ms, but we ceil to >= 1s.
    assert lim.retry_after_seconds(customer_id="c", ip="1.1.1.1") == 1


def test_retry_after_scales_with_deficit_when_rps_is_low() -> None:
    lim = TokenBucketLimiter(rps=0.5, burst=1, clock=lambda: 0.0)
    lim.acquire(customer_id="c", ip="1.1.1.1")
    # Need 1 token at rps=0.5 -> 2 seconds.
    assert lim.retry_after_seconds(customer_id="c", ip="1.1.1.1") == 2


def test_rps_and_burst_validated_at_construction() -> None:
    with pytest.raises(ValueError, match="rps"):
        TokenBucketLimiter(rps=0)
    with pytest.raises(ValueError, match="burst"):
        TokenBucketLimiter(rps=1.0, burst=0)


# ---------- middleware integration ----------


@pytest.fixture
def rate_limited_client(tmp_path: Path) -> Iterator[TestClient]:
    """A TestClient with a deliberately tiny bucket (burst=2, rps=0.1)
    so we can drain it in a couple of POSTs without time travel."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    from clarion.pipelines.structured import StructuredStore
    from clarion.schemas import AvailabilitySlot, EligibilityRecord, Provider

    seeds_dir = REPO_ROOT / "data" / "seeds"
    payload = json.loads((seeds_dir / "ophthalmology.json").read_text(encoding="utf-8"))
    store = StructuredStore.for_customer("ophthalmology", data_dir)
    for p in payload["providers"]:
        store.upsert_provider(Provider(**p))
    for s in payload["availability"]:
        store.upsert_slot(AvailabilitySlot(**s))
    for e in payload["eligibility"]:
        store.upsert_eligibility(EligibilityRecord(**e))

    settings = Settings(customer="ophthalmology", config_dir=CONFIGS_DIR, data_dir=data_dir)

    def factory() -> LLMClient:
        return FakeLLM(
            responses=[
                LLMResponse(content="ok", tool_calls=[]),
                LLMResponse(content="ok", tool_calls=[]),
                LLMResponse(content="ok", tool_calls=[]),
            ]
        )

    sessions = make_session_manager(settings, llm_factory=factory)
    limiter = TokenBucketLimiter(rps=0.1, burst=2, clock=lambda: 0.0)
    app = create_app(settings=settings, sessions=sessions, rate_limiter=limiter)
    with TestClient(app) as c:
        yield c


def test_health_is_never_rate_limited(rate_limited_client: TestClient) -> None:
    # Burn through many gets — health is not in RATE_LIMITED_PATHS.
    for _ in range(50):
        r = rate_limited_client.get("/health")
        assert r.status_code == 200


def test_chat_returns_429_after_burst_drained(rate_limited_client: TestClient) -> None:
    body = {"customer_id": "ophthalmology", "message": "book"}
    ok1 = rate_limited_client.post("/chat", json=body)
    ok2 = rate_limited_client.post("/chat", json=body)
    rate_limited = rate_limited_client.post("/chat", json=body)
    assert ok1.status_code == 200
    assert ok2.status_code == 200
    assert rate_limited.status_code == 429
    assert "Retry-After" in rate_limited.headers
    detail = rate_limited.json()["detail"]
    assert detail["code"] == "rate_limited"


def test_429_response_carries_correlation_id(rate_limited_client: TestClient) -> None:
    """Correlation id is set BEFORE the rate-limit middleware fires
    (correlation is registered later, so it runs first outbound on
    the response). A 429 must still have an X-Request-Id so the
    client can correlate the rejection."""
    body = {"customer_id": "ophthalmology", "message": "book"}
    # Drain the bucket.
    rate_limited_client.post("/chat", json=body)
    rate_limited_client.post("/chat", json=body)
    r = rate_limited_client.post(
        "/chat", json=body, headers={"X-Request-Id": "drain-rejection"}
    )
    assert r.status_code == 429
    assert r.headers["x-request-id"] == "drain-rejection"


def test_malformed_body_falls_through_to_route(rate_limited_client: TestClient) -> None:
    """A missing customer_id means we can't key a bucket; let the route
    handler produce its native 422 instead of synthesizing a 429."""
    r = rate_limited_client.post("/chat", json={"message": "no customer"})
    assert r.status_code in {400, 422}  # FastAPI / pydantic body validation


def test_xff_header_used_for_ip_isolation(rate_limited_client: TestClient) -> None:
    """Two requests with different X-Forwarded-For values land in
    different buckets, so neither drains the other."""
    body = {"customer_id": "ophthalmology", "message": "book"}
    # Client A drains:
    rate_limited_client.post("/chat", json=body, headers={"X-Forwarded-For": "10.0.0.1"})
    rate_limited_client.post("/chat", json=body, headers={"X-Forwarded-For": "10.0.0.1"})
    rejected_a = rate_limited_client.post(
        "/chat", json=body, headers={"X-Forwarded-For": "10.0.0.1"}
    )
    assert rejected_a.status_code == 429
    # Client B still has tokens:
    ok_b = rate_limited_client.post(
        "/chat", json=body, headers={"X-Forwarded-For": "10.0.0.2"}
    )
    assert ok_b.status_code == 200
