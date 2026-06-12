"""Production resilience primitives: retry, circuit breaker."""

from clarion.resilience.retry import (
    DEFAULT_BASE_DELAY_S,
    DEFAULT_CAP_S,
    DEFAULT_MAX_ATTEMPTS,
    compute_delay,
    is_transient_openai_error,
    retry_with_backoff,
)

__all__ = [
    "DEFAULT_BASE_DELAY_S",
    "DEFAULT_CAP_S",
    "DEFAULT_MAX_ATTEMPTS",
    "compute_delay",
    "is_transient_openai_error",
    "retry_with_backoff",
]
