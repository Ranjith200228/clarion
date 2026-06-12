"""SLA enforcement for POST /chat under FakeLLM (P18).

Marked ``loadtest`` so CI runs it explicitly (off by default for fast
local runs, on for release branches). FakeLLM is wired in so the
numbers reflect framework + middleware + agent overhead — the right
signal for catching hot-path regressions without burning OpenAI
credits.

SLO (from loadtest/README.md):

  | metric     | target  |
  | ---------- | ------- |
  | p50        | < 200ms |
  | p95        | < 500ms |
  | error_rate | < 0.5%  |
"""

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
from loadtest.in_process_load import measure_chat_burst

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"

P50_BUDGET_MS = 200.0
P95_BUDGET_MS = 500.0
MAX_ERROR_RATE = 0.005


pytestmark = pytest.mark.loadtest


@pytest.fixture
def sla_client(tmp_path: Path) -> Iterator[TestClient]:
    """A TestClient with a permissive rate limiter so the burst isn't
    throttled, and an infinite FakeLLM response queue."""
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

    class _RecyclingFakeLLM(FakeLLM):
        """FakeLLM that never runs out of canned responses."""

        def complete(self, *args: object, **kwargs: object) -> LLMResponse:  # type: ignore[override]
            return LLMResponse(content="ok", tool_calls=[])

    def factory() -> LLMClient:
        return _RecyclingFakeLLM(responses=[])

    sessions = make_session_manager(settings, llm_factory=factory)
    # Very high burst so the burst test isn't gated.
    limiter = TokenBucketLimiter(rps=10000.0, burst=10000)
    app = create_app(
        settings=settings,
        sessions=sessions,
        rate_limiter=limiter,
        install_logging=False,
    )
    with TestClient(app) as c:
        yield c


def test_chat_p95_under_sla(sla_client: TestClient) -> None:
    report = measure_chat_burst(sla_client, n_requests=200)
    # Quick sanity: at least one of every 4 hits a new conversation
    # (covers the session-pool hot path).
    assert report.count == 200
    assert report.error_rate < MAX_ERROR_RATE, report
    assert report.p50_ms < P50_BUDGET_MS, report
    assert report.p95_ms < P95_BUDGET_MS, report
