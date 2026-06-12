"""In-process burst test against a FastAPI TestClient.

Used by the CI ``loadtest`` marker to enforce the p50 / p95 SLA
without needing a real OpenAI key. The FakeLLM responds instantly,
so the numbers we measure are pure framework + middleware + agent
overhead — the right signal for catching regressions in the hot
path.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

from fastapi.testclient import TestClient


@dataclass(frozen=True)
class LatencyReport:
    """Latency summary in milliseconds."""

    count: int
    errors: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    error_rate: float


def measure_chat_burst(
    client: TestClient,
    *,
    n_requests: int = 200,
    customer_id: str = "ophthalmology",
    message_factory: Callable[[int], str] = lambda i: f"loadtest message {i}",
) -> LatencyReport:
    """Fire ``n_requests`` sequential POST /chat calls; collect latencies.

    Sequential rather than concurrent — TestClient runs the ASGI app
    in the same thread, so true concurrency requires httpx + a real
    server. For SLA regression detection, the sequential profile
    catches per-request slowdowns; we leave concurrency stress to
    the locust profile against a live deployment.
    """
    latencies_ms: list[float] = []
    errors = 0
    for i in range(n_requests):
        body = {
            "customer_id": customer_id,
            "message": message_factory(i),
            "conversation_id": f"loadtest-{i // 4}",  # 4 turns per conv
        }
        start = time.perf_counter()
        resp = client.post("/chat", json=body)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)
        if resp.status_code != 200:
            errors += 1

    return _summarize(latencies_ms, errors=errors)


def _summarize(latencies_ms: list[float], *, errors: int) -> LatencyReport:
    if not latencies_ms:
        return LatencyReport(0, errors, 0.0, 0.0, 0.0, 0.0, 1.0)
    sorted_ms = sorted(latencies_ms)
    return LatencyReport(
        count=len(latencies_ms),
        errors=errors,
        p50_ms=round(_percentile(sorted_ms, 50), 2),
        p95_ms=round(_percentile(sorted_ms, 95), 2),
        p99_ms=round(_percentile(sorted_ms, 99), 2),
        max_ms=round(max(latencies_ms), 2),
        error_rate=round(errors / max(1, len(latencies_ms)), 4),
    )


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile, matching statistics.quantiles."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    # Inclusive method to match what most operators expect.
    return float(statistics.quantiles(sorted_values, n=100, method="inclusive")[int(pct) - 1])
