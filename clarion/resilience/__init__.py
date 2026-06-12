"""Production resilience primitives: retry, circuit breaker."""

from clarion.resilience.circuit_breaker import (
    DEFAULT_COOLDOWN_S,
    DEFAULT_FAILURE_THRESHOLD,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from clarion.resilience.retry import (
    DEFAULT_BASE_DELAY_S,
    DEFAULT_CAP_S,
    DEFAULT_MAX_ATTEMPTS,
    compute_delay,
    is_transient_openai_error,
    retry_with_backoff,
)

__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "DEFAULT_BASE_DELAY_S",
    "DEFAULT_CAP_S",
    "DEFAULT_COOLDOWN_S",
    "DEFAULT_FAILURE_THRESHOLD",
    "DEFAULT_MAX_ATTEMPTS",
    "compute_delay",
    "is_transient_openai_error",
    "retry_with_backoff",
]
