"""Structured JSON logging + correlation IDs.

Production deployments want each log line as a single JSON object so
log aggregators (Loki, Datadog, CloudWatch) can index without grok.
Every line carries:

* ``timestamp`` — ISO 8601 with ms precision
* ``level``     — "DEBUG" / "INFO" / "WARNING" / "ERROR" / "CRITICAL"
* ``logger``    — Python logger name (module path)
* ``message``   — the formatted log message
* ``correlation_id`` — set by the API middleware; null in CLI contexts
* anything passed via ``extra={...}`` on the LogRecord

The correlation id is held in a contextvar so it survives across
async hops and threadpool calls — every span emitted during one
request shares one id, which is the magic that makes debugging a
production incident tractable.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

# Context-bound correlation id. The API middleware sets it; CLI / test
# code leaves it as None. Threadpool workers inherit the value via
# contextvars.copy_context (Python's default for run_in_executor).
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "clarion_correlation_id", default=None
)

# Reserved LogRecord attribute names — anything else passed via
# ``extra={...}`` is forwarded into the JSON payload's ``extra`` block.
_STANDARD_RECORD_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "taskName",
    }
)


def get_correlation_id() -> str | None:
    """Return the current request's correlation id, if any.

    Use sparingly outside logging — most callers should pass the id
    explicitly. This is the right hook for span emitters that want
    to stamp the id on their own payloads.
    """
    return _correlation_id.get()


def set_correlation_id(value: str | None) -> contextvars.Token[str | None]:
    """Bind a correlation id for the duration of the current context.

    Returns the token so callers can reset cleanly. Prefer
    :func:`correlation_id_scope` over a raw set + reset pair.
    """
    return _correlation_id.set(value)


def reset_correlation_id(token: contextvars.Token[str | None]) -> None:
    _correlation_id.reset(token)


@contextmanager
def correlation_id_scope(value: str | None) -> Iterator[str | None]:
    """Bind ``value`` as the current correlation id for the with-block.

    Usage::

        with correlation_id_scope(request_id) as cid:
            log.info("processing", extra={"step": "ingest"})

    Yields whatever was bound (the API middleware accepts the
    client's X-Request-Id when supplied; otherwise generates one).
    """
    token = _correlation_id.set(value)
    try:
        yield value
    finally:
        _correlation_id.reset(token)


def new_correlation_id() -> str:
    """Generate a fresh correlation id — UUID4, hex form."""
    return uuid.uuid4().hex


class JsonFormatter(logging.Formatter):
    """LogRecord -> single-line JSON.

    The output is one JSON object per log line, no embedded newlines
    in field values (exceptions are pre-flattened so they round-trip
    through line-oriented log shippers).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        cid = _correlation_id.get()
        if cid is not None:
            payload["correlation_id"] = cid

        # Anything attached via ``extra={...}`` lands in payload.extra.
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_RECORD_FIELDS and not k.startswith("_")
        }
        if extras:
            payload["extra"] = extras

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info).replace("\n", " | ")

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(
    *,
    level: int | str = logging.INFO,
    stream: Any = None,
    force: bool = True,
) -> None:
    """Install the JSON formatter on the root logger.

    Idempotent by default (``force=True`` removes any existing
    handlers first) so calling it twice in tests or from both the
    CLI entry point and the API factory is safe.

    Args:
        level: minimum log level to emit
        stream: file-like target (defaults to stderr to keep stdout
            clean for CLI output)
        force: when True, wipe existing root handlers before
            installing ours
    """
    root = logging.getLogger()
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
