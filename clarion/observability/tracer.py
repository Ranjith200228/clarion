"""Stack-based span tracer — one ``Tracer`` per ``Agent.chat`` call.

A ``Trace`` is a flat list of timed ``Span``s with parent references so the
shape can be reconstructed::

    agent.chat                     (root)
      retrieval                    (child of root)
      react.step                   (child of root)
        llm.complete               (child of step)
        tool.search_slots          (child of step)
      react.step
        llm.complete
        tool.book_appointment
      ...

Tracer is intentionally simple: one stack, ``span()`` is a context manager
that opens a span on enter, closes it on exit. Spans collected on the
Tracer can be drained into a ``Trace`` and written to disk by
``observability.writer``.

Not thread-safe — one Tracer per turn, callers create a new one per
``chat()``. The FastAPI service in Phase 8 will allocate one per request.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def new_trace_id() -> str:
    return _new_id("trace")


@dataclass
class Span:
    """One timed event with arbitrary attributes."""

    span_id: str
    parent_id: str | None
    name: str
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        """Attach an attribute to this span. Last write wins."""
        self.attributes[key] = value

    def update(self, mapping: dict[str, Any]) -> None:
        self.attributes.update(mapping)

    @property
    def is_open(self) -> bool:
        return self.ended_at is None


@dataclass
class Trace:
    """One conversation turn worth of spans."""

    trace_id: str
    customer_id: str | None = None
    conversation_id: str | None = None
    spans: list[Span] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "customer_id": self.customer_id,
            "conversation_id": self.conversation_id,
            "spans": [
                {
                    "span_id": s.span_id,
                    "parent_id": s.parent_id,
                    "name": s.name,
                    "started_at": s.started_at.isoformat(),
                    "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                    "duration_ms": s.duration_ms,
                    "attributes": s.attributes,
                }
                for s in self.spans
            ],
        }


class Tracer:
    """Stack-based span collector. One per chat() turn."""

    def __init__(
        self,
        *,
        trace_id: str | None = None,
        customer_id: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self._trace_id = trace_id or new_trace_id()
        self._customer_id = customer_id
        self._conversation_id = conversation_id
        self._spans: list[Span] = []
        self._stack: list[Span] = []

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def current(self) -> Span | None:
        """The innermost open span, or ``None`` if no span is active."""
        return self._stack[-1] if self._stack else None

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """Open a new span as a child of the current top of stack."""
        parent = self.current
        span = Span(
            span_id=_new_id("span"),
            parent_id=parent.span_id if parent else None,
            name=name,
            started_at=datetime.now(UTC),
            attributes=dict(attrs),
        )
        self._spans.append(span)
        self._stack.append(span)
        t0 = time.perf_counter()
        try:
            yield span
        finally:
            span.ended_at = datetime.now(UTC)
            span.duration_ms = (time.perf_counter() - t0) * 1000.0
            self._stack.pop()

    def emit(self) -> Trace:
        """Build the immutable ``Trace`` payload for writing."""
        return Trace(
            trace_id=self._trace_id,
            customer_id=self._customer_id,
            conversation_id=self._conversation_id,
            spans=list(self._spans),
        )
