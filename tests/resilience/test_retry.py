"""Tests for clarion.resilience.retry."""

from __future__ import annotations

import pytest
from clarion.resilience import compute_delay, is_transient_openai_error, retry_with_backoff

# ---------- compute_delay ----------


def test_compute_delay_grows_exponentially_until_cap() -> None:
    # rng pinned to 1.0 so we read the upper bound of each window.
    deltas = [compute_delay(a, base_delay_s=0.5, cap_s=4.0, rng=lambda: 1.0) for a in range(1, 6)]
    # attempt=1 -> 0.5, =2 -> 1.0, =3 -> 2.0, =4 -> 4.0 (capped), =5 -> 4.0 (capped).
    assert deltas == [0.5, 1.0, 2.0, 4.0, 4.0]


def test_compute_delay_jitter_window_is_full() -> None:
    # rng=0.0 -> bottom of window, rng=0.5 -> midpoint, rng=1.0 -> top.
    assert compute_delay(2, 0.5, 8.0, rng=lambda: 0.0) == 0.0
    assert compute_delay(2, 0.5, 8.0, rng=lambda: 0.5) == 0.5
    assert compute_delay(2, 0.5, 8.0, rng=lambda: 1.0) == 1.0


# ---------- retry_with_backoff ----------


def test_returns_immediately_on_success() -> None:
    calls = {"n": 0}

    @retry_with_backoff(max_attempts=4, sleep=lambda _: None)
    def ok() -> str:
        calls["n"] += 1
        return "ok"

    assert ok() == "ok"
    assert calls["n"] == 1


def test_retries_then_succeeds() -> None:
    state = {"n": 0}
    slept: list[float] = []

    @retry_with_backoff(
        max_attempts=4,
        base_delay_s=0.1,
        sleep=slept.append,
        rng=lambda: 1.0,  # always take the top of the window
    )
    def flaky() -> str:
        state["n"] += 1
        if state["n"] < 3:
            raise ConnectionError("nope")
        return "yay"

    assert flaky() == "yay"
    assert state["n"] == 3
    # Two failures means two sleeps; the third call succeeded.
    assert slept == [0.1, 0.2]


def test_gives_up_after_max_attempts_and_reraises_last() -> None:
    @retry_with_backoff(max_attempts=3, sleep=lambda _: None)
    def always_fails() -> str:
        raise TimeoutError("still broken")

    with pytest.raises(TimeoutError, match="still broken"):
        always_fails()


def test_non_retryable_class_propagates_immediately() -> None:
    state = {"n": 0}

    @retry_with_backoff(
        max_attempts=5,
        retry_on=(ConnectionError,),
        sleep=lambda _: None,
    )
    def auth_error() -> str:
        state["n"] += 1
        raise ValueError("bad arg")  # not in retry_on

    with pytest.raises(ValueError):
        auth_error()
    assert state["n"] == 1


def test_should_retry_predicate_overrides_class_match() -> None:
    state = {"n": 0}

    class HttpError(Exception):
        def __init__(self, status: int) -> None:
            self.status = status

    @retry_with_backoff(
        max_attempts=5,
        retry_on=(HttpError,),
        should_retry=lambda e: getattr(e, "status", 0) >= 500,
        sleep=lambda _: None,
    )
    def fn() -> str:
        state["n"] += 1
        raise HttpError(404)  # client error -> don't retry

    with pytest.raises(HttpError):
        fn()
    assert state["n"] == 1


def test_rejects_zero_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        retry_with_backoff(max_attempts=0)


# ---------- is_transient_openai_error ----------


def test_is_transient_recognizes_named_classes() -> None:
    class APITimeoutError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    class AuthenticationError(Exception):
        pass

    assert is_transient_openai_error(APITimeoutError())
    assert is_transient_openai_error(RateLimitError())
    assert not is_transient_openai_error(AuthenticationError())


def test_is_transient_falls_back_to_builtin_class() -> None:
    assert is_transient_openai_error(ConnectionError())
    assert is_transient_openai_error(TimeoutError())
    assert not is_transient_openai_error(ValueError())
