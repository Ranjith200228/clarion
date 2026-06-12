"""Circuit breaker for a single upstream dependency.

Retries cap individual-call latency; a circuit breaker caps
*aggregate* latency when the upstream is genuinely down. Without
one, every request keeps paying the full retry envelope (~5 s of
backoff) before failing — under load that turns a 30-second outage
into thread-pool exhaustion.

States — the canonical three-state machine:

* **closed**     — calls pass through; failures increment a counter.
* **open**       — every call fast-fails with ``CircuitOpenError``
  for ``cooldown_s`` seconds. We don't even attempt the upstream;
  the whole point is to relieve pressure on it.
* **half_open**  — one probe call is allowed through to test
  recovery. Success closes the circuit; failure reopens it (with
  a fresh cooldown).

Used to wrap the OpenAI client's ``_chat_completions_create``
method — the same boundary the retry decorator wraps. The two
compose naturally: retry tries up to ``max_attempts`` times,
then if it still fails the failure counts toward the breaker.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_FAILURE_THRESHOLD = 5
DEFAULT_COOLDOWN_S = 30.0


class CircuitState(str, Enum):
    """Three canonical states. str-mixin makes them log-friendly."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised by a wrapped callable when the breaker is open.

    Subclass of RuntimeError so existing ``except Exception`` handlers
    still catch it; the dedicated class lets callers fast-path the
    "circuit open, don't retry" decision without sniffing strings.
    """


@dataclass
class _BreakerState:
    """Internal state — protected by the breaker's lock."""

    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0


class CircuitBreaker:
    """One breaker per upstream dependency.

    Construct one at module scope (or on the client instance) and
    wrap any number of methods via :meth:`wrap` / :meth:`__call__`.
    All wrapped callables share the breaker's state.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        expected_exception: type[BaseException] = Exception,
        clock: Callable[[], float] = time.monotonic,
        name: str = "circuit",
    ) -> None:
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")
        if cooldown_s < 0:
            raise ValueError(f"cooldown_s must be >= 0, got {cooldown_s}")
        self._failure_threshold = failure_threshold
        self._cooldown_s = cooldown_s
        self._expected = expected_exception
        self._clock = clock
        self._name = name
        self._state = _BreakerState()
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Current observed state. Reads are lock-free since reads of a
        single attribute are atomic in CPython; tests and the dashboard
        accept the small staleness window."""
        return self._state.state

    @property
    def consecutive_failures(self) -> int:
        return self._state.consecutive_failures

    def wrap(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Decorate ``fn`` so its calls flow through this breaker."""

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            self._before_call()
            try:
                result = fn(*args, **kwargs)
            except self._expected as exc:
                self._on_failure(exc)
                raise
            else:
                self._on_success()
                return result

        return wrapper

    __call__ = wrap

    # ---------- state machine ----------

    def _before_call(self) -> None:
        """Decide whether the call is allowed to proceed."""
        with self._lock:
            if self._state.state == CircuitState.OPEN:
                if self._clock() - self._state.opened_at >= self._cooldown_s:
                    # Cooldown elapsed — try one probe call.
                    self._state.state = CircuitState.HALF_OPEN
                    log.warning(
                        "circuit %s transitioning open -> half_open",
                        self._name,
                        extra={"breaker": self._name},
                    )
                else:
                    raise CircuitOpenError(
                        f"circuit {self._name!r} is open; cooldown {self._cooldown_s:.1f}s"
                    )

    def _on_success(self) -> None:
        with self._lock:
            if self._state.state == CircuitState.HALF_OPEN:
                log.info(
                    "circuit %s probe succeeded; closing",
                    self._name,
                    extra={"breaker": self._name},
                )
            self._state.state = CircuitState.CLOSED
            self._state.consecutive_failures = 0

    def _on_failure(self, exc: BaseException) -> None:
        with self._lock:
            if self._state.state == CircuitState.HALF_OPEN:
                # Probe failed — reopen with fresh cooldown.
                self._state.state = CircuitState.OPEN
                self._state.opened_at = self._clock()
                log.warning(
                    "circuit %s probe failed; reopening",
                    self._name,
                    extra={"breaker": self._name, "error": str(exc)},
                )
                return
            self._state.consecutive_failures += 1
            if self._state.consecutive_failures >= self._failure_threshold:
                self._state.state = CircuitState.OPEN
                self._state.opened_at = self._clock()
                log.warning(
                    "circuit %s tripped after %d failures; cooling down %.1fs",
                    self._name,
                    self._state.consecutive_failures,
                    self._cooldown_s,
                    extra={"breaker": self._name},
                )

    def reset(self) -> None:
        """Force the breaker back to CLOSED. Useful for ops + tests."""
        with self._lock:
            self._state = _BreakerState()
