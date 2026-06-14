"""Tests for the /chat soft-error guard.

When the LLM call fails (bad key, rate limit, network), the route
must NOT 500. It returns 200 with a Markdown reply explaining what
went wrong. This is what makes a public demo robust to revoked /
quota-capped keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from clarion.agents.llm import FakeLLM, LLMClient
from clarion.config import Settings
from fastapi.testclient import TestClient

from api.app import create_app
from api.sessions import make_session_manager

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


class _BoomLLM:
    """LLMClient stub that raises a chosen exception on every complete."""

    def __init__(self, exc_class: type[Exception], message: str) -> None:
        self._exc_class = exc_class
        self._message = message

    def complete(self, messages, *, tools=None):  # type: ignore[no-untyped-def]
        raise self._exc_class(self._message)


class AuthenticationError(Exception):
    """Stand-in for openai.AuthenticationError so we don't take a hard dep."""


class RateLimitError(Exception):
    """Stand-in for openai.RateLimitError."""


@pytest.fixture
def app_with_broken_llm(
    tmp_path: Path, request: pytest.FixtureRequest
) -> TestClient:
    exc_class, message = request.param
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings = Settings(
        customer="ophthalmology",
        config_dir=CONFIGS_DIR,
        data_dir=data_dir,
    )

    def factory() -> LLMClient:
        return _BoomLLM(exc_class, message)  # type: ignore[return-value]

    sessions = make_session_manager(settings, llm_factory=factory)
    app = create_app(
        settings=settings,
        sessions=sessions,
        install_logging=False,
    )
    return TestClient(app)


@pytest.mark.parametrize(
    "app_with_broken_llm",
    [(AuthenticationError, "Incorrect API key provided")],
    indirect=True,
)
def test_auth_error_returns_friendly_reply(app_with_broken_llm: TestClient) -> None:
    r = app_with_broken_llm.post(
        "/chat",
        json={"customer_id": "ophthalmology", "message": "hi"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "invalid or revoked" in body["reply"].lower()
    assert "OPENAI_API_KEY" in body["reply"]


@pytest.mark.parametrize(
    "app_with_broken_llm",
    [(RateLimitError, "Rate limit reached for requests")],
    indirect=True,
)
def test_rate_limit_returns_friendly_reply(app_with_broken_llm: TestClient) -> None:
    r = app_with_broken_llm.post(
        "/chat",
        json={"customer_id": "ophthalmology", "message": "hi"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    reply = body["reply"].lower()
    assert "rate limit" in reply or "quota" in reply


@pytest.mark.parametrize(
    "app_with_broken_llm",
    [(RuntimeError, "wires crossed")],
    indirect=True,
)
def test_generic_error_returns_friendly_catchall(
    app_with_broken_llm: TestClient,
) -> None:
    r = app_with_broken_llm.post(
        "/chat",
        json={"customer_id": "ophthalmology", "message": "hi"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "model call failed" in body["reply"].lower()
    assert "RuntimeError" in body["reply"]


def test_happy_path_unchanged(tmp_path: Path) -> None:
    """Soft-error guard must not affect the success path."""
    from clarion.agents.llm import LLMResponse

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings = Settings(
        customer="ophthalmology",
        config_dir=CONFIGS_DIR,
        data_dir=data_dir,
    )
    fake = FakeLLM(
        responses=[LLMResponse(content="Sure, let me help.", tool_calls=())]
    )

    def factory() -> LLMClient:
        return fake

    sessions = make_session_manager(settings, llm_factory=factory)
    app = create_app(settings=settings, sessions=sessions, install_logging=False)
    client = TestClient(app)
    r = client.post(
        "/chat",
        json={"customer_id": "ophthalmology", "message": "hi"},
    )
    assert r.status_code == 200
    assert r.json()["reply"] == "Sure, let me help."
