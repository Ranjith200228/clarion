"""Tests for clarion.resilience.circuit_breaker."""

from __future__ import annotations

import pytest
from clarion.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


def _flaky(state: dict[str, int], n_failures: int) -> object:
    """Returns a callable that raises ConnectionError n_failures times,
    then returns 'ok'."""

    def fn() -> str:
        state["calls"] += 1
        if state["calls"] <= n_failures:
            raise ConnectionError(f"fail #{state['calls']}")
        return "ok"

    return fn


# ---------- construction validation ----------


def test_rejects_zero_failure_threshold() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=0)


def test_rejects_negative_cooldown() -> None:
    with pytest.raises(ValueError, match="cooldown_s"):
        CircuitBreaker(cooldown_s=-1.0)


# ---------- closed -> open ----------


def test_starts_closed() -> None:
    cb = CircuitBreaker()
    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 0


def test_consecutive_failures_count() -> None:
    cb = CircuitBreaker(failure_threshold=10, clock=lambda: 0.0)
    state = {"calls": 0}
    fn = cb.wrap(_flaky(state, n_failures=3))
    for _ in range(3):
        with pytest.raises(ConnectionError):
            fn()
    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 3
    # A success resets the counter.
    fn()
    assert cb.consecutive_failures == 0


def test_trips_after_threshold_consecutive_failures() -> None:
    cb = CircuitBreaker(failure_threshold=3, cooldown_s=10.0, clock=lambda: 0.0)
    state = {"calls": 0}
    fn = cb.wrap(_flaky(state, n_failures=100))

    for _ in range(3):
        with pytest.raises(ConnectionError):
            fn()
    assert cb.state == CircuitState.OPEN

    # Further calls fast-fail with CircuitOpenError; upstream NOT called.
    with pytest.raises(CircuitOpenError):
        fn()
    assert state["calls"] == 3  # upstream untouched


# ---------- open -> half_open -> closed/open ----------


def test_half_open_after_cooldown_then_closes_on_success() -> None:
    fake_now = [0.0]
    cb = CircuitBreaker(failure_threshold=2, cooldown_s=5.0, clock=lambda: fake_now[0])
    state = {"calls": 0}
    fn = cb.wrap(_flaky(state, n_failures=2))

    # Trip the circuit.
    with pytest.raises(ConnectionError):
        fn()
    with pytest.raises(ConnectionError):
        fn()
    assert cb.state == CircuitState.OPEN

    # Within cooldown -> fast-fail.
    fake_now[0] = 4.99
    with pytest.raises(CircuitOpenError):
        fn()

    # Past cooldown -> half_open probe runs, upstream now healthy -> closed.
    fake_now[0] = 5.01
    result = fn()
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 0


def test_half_open_probe_failure_reopens_with_fresh_cooldown() -> None:
    fake_now = [0.0]
    cb = CircuitBreaker(failure_threshold=2, cooldown_s=5.0, clock=lambda: fake_now[0])
    state = {"calls": 0}
    # First 2 calls fail; probe (call #3) also fails; then a success.
    fn = cb.wrap(_flaky(state, n_failures=3))

    with pytest.raises(ConnectionError):
        fn()
    with pytest.raises(ConnectionError):
        fn()
    assert cb.state == CircuitState.OPEN

    # Move past cooldown -> probe runs and fails -> reopens.
    fake_now[0] = 6.0
    with pytest.raises(ConnectionError):
        fn()
    assert cb.state == CircuitState.OPEN
    # Cooldown reset to the new opened_at — calls at 6.0 + tiny epsilon
    # still fast-fail.
    fake_now[0] = 6.01
    with pytest.raises(CircuitOpenError):
        fn()


# ---------- expected_exception filter ----------


def test_non_expected_exception_doesnt_count_toward_threshold() -> None:
    cb = CircuitBreaker(
        failure_threshold=2,
        expected_exception=ConnectionError,
        clock=lambda: 0.0,
    )

    @cb.wrap
    def fn() -> str:
        raise ValueError("not transient")

    for _ in range(5):
        with pytest.raises(ValueError):
            fn()
    # ValueError isn't in ``expected_exception``, so it propagates
    # without touching the breaker's counter.
    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 0


# ---------- reset ----------


def test_reset_returns_breaker_to_closed_with_zero_counter() -> None:
    cb = CircuitBreaker(failure_threshold=2, clock=lambda: 0.0)
    state = {"calls": 0}
    fn = cb.wrap(_flaky(state, n_failures=2))

    with pytest.raises(ConnectionError):
        fn()
    with pytest.raises(ConnectionError):
        fn()
    assert cb.state == CircuitState.OPEN

    cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 0


# ---------- __call__ alias ----------


def test_call_alias_is_equivalent_to_wrap() -> None:
    cb = CircuitBreaker()

    @cb  # uses __call__ as a decorator
    def ok() -> str:
        return "ok"

    assert ok() == "ok"
    assert cb.state == CircuitState.CLOSED
