"""Token-bucket rate limiter, scoped per (customer_id, client IP).

Why token bucket (and not a fixed-window or sliding-window counter):

* **Burst-friendly.** Real clients send a few requests in quick
  succession then idle; a sliding window punishes that pattern,
  even when the sustained rate is well under the limit. Tokens
  accumulate during quiet periods up to ``burst`` and deplete on
  use — the natural pattern is what we want.
* **No state-store dependency.** A token bucket needs one float
  ("tokens remaining") and one timestamp per key; we can hold
  that in-process. Fixed-window counters need atomic ints in
  Redis to be honest under concurrency; this skips that complexity
  for the single-replica HF Spaces deployment.

Limits are intentionally generous for the demo deployment (10 rps,
burst 30 per customer/IP); the constants are easy to tune via
``RateLimiter(...)`` constructor args when this lands behind a
real production gateway.

The middleware identifies a request by ``(customer_id, ip)``:

* ``customer_id`` is read from the JSON body when the route is
  POST /chat or POST /voice/turn (the two cost-bearing endpoints).
  GET /health is free. POST /evaluate has its own internal
  cost-bearing structure but is gated by deployment ops, not
  per-request limits — we exempt it here.
* ``ip`` comes from the X-Forwarded-For chain when present
  (HF Spaces sits behind a proxy), otherwise the direct peer.
  We take the first value in the chain — the closest one to the
  original client.

When a request is rejected, we respond 429 with a ``Retry-After``
header indicating roughly how long until a token will be available.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

log = logging.getLogger(__name__)

DEFAULT_RPS = 10.0
DEFAULT_BURST = 30
RATE_LIMITED_PATHS: frozenset[str] = frozenset({"/chat", "/voice/turn"})


@dataclass
class _Bucket:
    """Single token bucket. ``tokens`` is a fractional float."""

    tokens: float
    last_refill: float


class TokenBucketLimiter:
    """In-process token bucket keyed by ``(customer_id, ip)``.

    Thread-safe via a single coarse lock; the contention window is
    a few microseconds (read clock, do arithmetic, write back), so
    splitting locks per key isn't worth the bookkeeping.
    """

    def __init__(
        self,
        *,
        rps: float = DEFAULT_RPS,
        burst: int = DEFAULT_BURST,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rps <= 0:
            raise ValueError(f"rps must be > 0, got {rps}")
        if burst < 1:
            raise ValueError(f"burst must be >= 1, got {burst}")
        self._rps = rps
        self._burst = burst
        self._clock = clock
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._lock = threading.Lock()

    def acquire(self, *, customer_id: str, ip: str) -> bool:
        """Try to consume one token. Returns True if allowed."""
        key = (customer_id, ip)
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                # First request from this key: full bucket, take one.
                self._buckets[key] = _Bucket(
                    tokens=float(self._burst - 1), last_refill=now
                )
                return True
            # Refill since the last visit.
            elapsed = max(0.0, now - bucket.last_refill)
            bucket.tokens = min(float(self._burst), bucket.tokens + elapsed * self._rps)
            bucket.last_refill = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False

    def retry_after_seconds(self, *, customer_id: str, ip: str) -> int:
        """How long until a token is available, ceiled to whole seconds.

        Always returns at least 1 — clients that get a 429 should not
        immediately retry against the same bucket.
        """
        key = (customer_id, ip)
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None or bucket.tokens >= 1.0:
                return 1
            needed = 1.0 - bucket.tokens
            seconds = needed / self._rps
        return max(1, math.ceil(seconds))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that gates the cost-bearing routes.

    Routes outside :data:`RATE_LIMITED_PATHS` pass through. For gated
    routes, we sniff ``customer_id`` from the JSON body (the request
    is buffered once, reset after, so downstream handlers still parse
    normally), look up the (customer_id, ip) bucket, and either pass
    the call through or short-circuit with a 429.
    """

    def __init__(self, app: ASGIApp, *, limiter: TokenBucketLimiter) -> None:
        super().__init__(app)
        self._limiter = limiter

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path not in RATE_LIMITED_PATHS:
            return await call_next(request)

        customer_id = await _sniff_customer_id(request)
        if customer_id is None:
            # Body was malformed or missing customer_id; let the route
            # produce its normal 4xx response.
            return await call_next(request)

        ip = _client_ip(request)
        if self._limiter.acquire(customer_id=customer_id, ip=ip):
            return await call_next(request)

        retry = self._limiter.retry_after_seconds(customer_id=customer_id, ip=ip)
        log.warning(
            "rate-limit reject",
            extra={"customer_id": customer_id, "ip": ip, "path": request.url.path},
        )
        return JSONResponse(
            status_code=429,
            content={
                "detail": {
                    "detail": (
                        f"rate limit exceeded for customer={customer_id!r}; "
                        f"retry in {retry}s"
                    ),
                    "code": "rate_limited",
                },
            },
            headers={"Retry-After": str(retry)},
        )


# ---------- helpers ----------


async def _sniff_customer_id(request: Request) -> str | None:
    """Buffer the request body, parse JSON, return customer_id or None.

    Starlette's BaseHTTPMiddleware re-serves the same body to the route
    handler, so the buffered read here doesn't break downstream parsing.
    """
    try:
        body = await request.body()
        if not body:
            return None
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    customer_id = payload.get("customer_id")
    return customer_id if isinstance(customer_id, str) else None


def _client_ip(request: Request) -> str:
    """Closest-original IP from the X-Forwarded-For chain, or direct peer."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        first = xff.split(",", 1)[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"
