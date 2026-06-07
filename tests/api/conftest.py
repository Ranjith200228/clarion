"""FastAPI TestClient fixtures for the API tests.

Each test gets its own app instance pointing at a tmp data dir. The LLM
factory wires a FakeLLM with a queue of canned responses, so the same
fixture supports happy-path tests (script the LLM) and guardrail tests
(no LLM responses required).
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
from api.sessions import make_session_manager

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Tmp data dir pre-built so per-customer subdirs are write-ready."""
    (tmp_path / "data").mkdir()
    return tmp_path / "data"


@pytest.fixture
def fake_llm_responses() -> list[LLMResponse]:
    """Override per-test via indirect parametrize or direct assignment.

    Empty by default; tests that script the LLM extend the list.
    """
    return []


@pytest.fixture
def seed_customers(tmp_data_dir: Path) -> Path:
    """Seed structured stores for both shipped customers inside the tmp
    data dir so /chat doesn't crash on first use.

    Skip building the FAISS index here — the API tests focus on the HTTP
    surface and session plumbing, not on RAG accuracy (Phase 3's tests
    already cover that). The SessionManager's load_customer_retriever()
    will see no rules.faiss and return None, which the agent handles
    by emitting a retrieval span with hit_count=0 and omitting the
    rules block in the prompt.
    """
    from clarion.pipelines.structured import StructuredStore
    from clarion.schemas import (
        AvailabilitySlot,
        EligibilityRecord,
        Provider,
    )

    seeds_dir = REPO_ROOT / "data" / "seeds"
    for customer_id in ("ophthalmology", "orthopedics"):
        store = StructuredStore.for_customer(customer_id, tmp_data_dir)
        payload = json.loads((seeds_dir / f"{customer_id}.json").read_text(encoding="utf-8"))
        for p in payload["providers"]:
            store.upsert_provider(Provider(**p))
        for s in payload["availability"]:
            store.upsert_slot(AvailabilitySlot(**s))
        for e in payload["eligibility"]:
            store.upsert_eligibility(EligibilityRecord(**e))
    return tmp_data_dir


@pytest.fixture
def client(
    tmp_data_dir: Path,
    fake_llm_responses: list[LLMResponse],
    seed_customers: Path,
) -> Iterator[TestClient]:
    settings = Settings(
        customer="ophthalmology",
        config_dir=CONFIGS_DIR,
        data_dir=tmp_data_dir,
    )
    # One FakeLLM per process is fine — tests don't run in parallel.
    fake = FakeLLM(responses=list(fake_llm_responses))

    def factory() -> LLMClient:
        return fake

    sessions = make_session_manager(settings, llm_factory=factory)
    app = create_app(settings=settings, sessions=sessions)
    with TestClient(app) as c:
        # Hand the FakeLLM back so individual tests can append responses
        # and assert call counts.
        c.app.state.fake_llm = fake  # type: ignore[attr-defined]
        yield c
