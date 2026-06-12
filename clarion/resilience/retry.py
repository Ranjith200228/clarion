"""Retry decorator with exponential backoff + full jitter.

Why this and not tenacity? We need exactly three things — a max
attempt count, exponential delay with jitter, and a per-call
predicate that picks which exceptions are worth retrying. Tenacity
solves a much larger problem; importing it would land another
transitive dep on the wire for a 30-line algorithm.

Backoff schedule (full jitter, AWS Architecture Blog flavor):

    delay = uniform(0, min(cap_s, base_delay_s * 2 ** attempt))

Full jitter dominates equal jitter / decorrelated jitter for
small retry counts (the regime we're in — 3-5 attempts) because
it spreads concurrent retries evenly across the window. With our
typical max_attempts=4 and base_delay_s=0.25, the worst-case
total wait is ~5s, well under any reasonable request timeout.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_BASE_DELAY_S = 0.25
DEFAULT_CAP_S = 8.0
DEFAULT_MAX_ATTEMPTS = 4


def retry_with_backoff(
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
    cap_s: float = DEFAULT_CAP_S,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    should_retry: Callable[[BaseException], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorate a callable with retry + exponential backoff + full jitter.

    Args:
        max_attempts: total number of tries (including the first call;
            so ``max_attempts=4`` -> up to three retries).
        base_delay_s: delay scale; the first retry waits between 0 and
            ``base_delay_s`` seconds.
        cap_s: maximum single-attempt sleep, regardless of attempt
            count. Prevents runaway backoff under many retries.
        retry_on: exception classes that even qualify for retry. Other
            exceptions propagate immediately — never swallow programmer
            errors or auth failures.
        should_retry: optional callable for finer filtering (e.g. inspect
            an HTTP status code on the exception). Called only when the
            exception type matches ``retry_on``. Return False to give up
            and re-raise.
        sleep: injectable sleep (tests pass a fake; production keeps
            time.sleep).
        rng: injectable [0, 1) uniform source for the jitter; tests pass
            a deterministic stub.

    Raises whatever the wrapped callable raised on its final attempt.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    if base_delay_s < 0:
        raise ValueError(f"base_delay_s must be >= 0, got {base_delay_s}")

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            last_exc: BaseException | None = None
            while attempt < max_attempts:
                try:
                    return fn(*args, **kwargs)
                except retry_on as exc:
                    last_exc = exc
                    if should_retry is not None and not should_retry(exc):
                        raise
                    attempt += 1
                    if attempt >= max_attempts:
                        log.warning(
                            "retry %s giving up after %d attempts",
                            getattr(fn, "__qualname__", "fn"),
                            attempt,
                            extra={"attempts": attempt, "error": str(exc)},
                        )
                        raise
                    delay = compute_delay(attempt, base_delay_s, cap_s, rng=rng)
                    log.info(
                        "retry %s attempt %d/%d after %.3fs (cause=%s)",
                        getattr(fn, "__qualname__", "fn"),
                        attempt + 1,
                        max_attempts,
                        delay,
                        type(exc).__name__,
                    )
                    sleep(delay)
            # Defensive — the loop only exits via return or raise above.
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


def compute_delay(
    attempt: int,
    base_delay_s: float,
    cap_s: float,
    *,
    rng: Callable[[], float] = random.random,
) -> float:
    """Full-jitter delay: uniform(0, min(cap, base * 2**(attempt-1))).

    ``attempt`` is 1-indexed — attempt=1 means "about to do the first
    retry" (i.e. the original call already failed).
    """
    exp: float = base_delay_s * (2 ** max(0, attempt - 1))
    window: float = min(cap_s, exp)
    sample: float = rng()
    result: float = sample * window
    return result


def is_transient_openai_error(exc: BaseException) -> bool:
    """Predicate suitable for ``should_retry`` on OpenAI calls.

    Retries network / timeout / 5xx classes; refuses to retry
    AuthenticationError, BadRequestError, NotFoundError, and the
    catch-all OpenAIError (which is the parent of programmer errors
    we'd rather see crash loudly).

    Implemented by class-name sniffing so the openai package isn't
    a hard dep of this module — we want retry to be importable
    without dragging openai into the test path.
    """
    name = type(exc).__name__
    transient_names = {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "APIError",  # base for many transient 5xx variants
    }
    if name in transient_names:
        return True
    # Some SDK versions raise ConnectionError / TimeoutError directly.
    return isinstance(exc, ConnectionError | TimeoutError)
