"""Tests for clarion.observability.logging — JSON formatter +
correlation id contextvar + the API middleware that binds it."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from clarion.agents.llm import FakeLLM, LLMClient
from clarion.config import Settings
from clarion.observability import (
    JsonFormatter,
    configure_logging,
    correlation_id_scope,
    get_correlation_id,
    new_correlation_id,
)
from fastapi.testclient import TestClient

from api.app import create_app
from api.sessions import make_session_manager

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


# ---------- formatter ----------


def _format_one(record: logging.LogRecord) -> dict[str, object]:
    formatter = JsonFormatter()
    return json.loads(formatter.format(record))  # type: ignore[no-any-return]


def test_formatter_emits_required_fields() -> None:
    record = logging.LogRecord(
        name="clarion.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    payload = _format_one(record)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "clarion.test"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload
    # No correlation id outside a request scope.
    assert "correlation_id" not in payload


def test_formatter_includes_correlation_id_when_bound() -> None:
    record = logging.LogRecord(
        name="clarion.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="bound",
        args=(),
        exc_info=None,
    )
    with correlation_id_scope("abc123"):
        payload = _format_one(record)
    assert payload["correlation_id"] == "abc123"


def test_formatter_forwards_extras() -> None:
    record = logging.LogRecord(
        name="clarion.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="extra",
        args=(),
        exc_info=None,
    )
    record.customer_id = "ophthalmology"  # type: ignore[attr-defined]
    record.tool = "search_slots"  # type: ignore[attr-defined]
    payload = _format_one(record)
    assert payload["extra"] == {
        "customer_id": "ophthalmology",
        "tool": "search_slots",
    }


def test_formatter_flattens_exception_to_single_line() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        record = logging.LogRecord(
            name="clarion.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="caught",
            args=(),
            exc_info=sys.exc_info(),
        )
    payload = _format_one(record)
    assert "exception" in payload
    assert "RuntimeError" in str(payload["exception"])
    # Single-line — no embedded \n.
    assert "\n" not in str(payload["exception"])


# ---------- configure_logging ----------


def test_configure_logging_installs_one_handler_idempotently() -> None:
    buf = io.StringIO()
    configure_logging(level=logging.DEBUG, stream=buf)
    configure_logging(level=logging.DEBUG, stream=buf)
    root = logging.getLogger()
    assert len(root.handlers) == 1

    log = logging.getLogger("clarion.test.logging")
    log.info("hi there", extra={"k": "v"})
    line = buf.getvalue().strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["message"] == "hi there"
    assert payload["extra"] == {"k": "v"}


# ---------- correlation id helpers ----------


def test_correlation_id_scope_is_lexical() -> None:
    assert get_correlation_id() is None
    with correlation_id_scope("outer"):
        assert get_correlation_id() == "outer"
        with correlation_id_scope("inner"):
            assert get_correlation_id() == "inner"
        assert get_correlation_id() == "outer"
    assert get_correlation_id() is None


def test_new_correlation_id_is_unique_hex() -> None:
    a = new_correlation_id()
    b = new_correlation_id()
    assert a != b
    assert len(a) == 32
    int(a, 16)  # parseable as hex


# ---------- CorrelationIdMiddleware ----------


@pytest.fixture
def correlation_client(tmp_path: Path) -> Iterator[TestClient]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings = Settings(customer="ophthalmology", config_dir=CONFIGS_DIR, data_dir=data_dir)

    def factory() -> LLMClient:
        return FakeLLM(responses=[])

    sessions = make_session_manager(settings, llm_factory=factory)
    app = create_app(settings=settings, sessions=sessions, install_logging=False)
    with TestClient(app) as c:
        yield c


def test_middleware_echoes_client_request_id(correlation_client: TestClient) -> None:
    r = correlation_client.get("/health", headers={"X-Request-Id": "client-supplied"})
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "client-supplied"


def test_middleware_mints_id_when_absent(correlation_client: TestClient) -> None:
    r = correlation_client.get("/health")
    assert r.status_code == 200
    cid = r.headers["x-request-id"]
    assert len(cid) == 32
    int(cid, 16)
