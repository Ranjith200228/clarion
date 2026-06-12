"""Locust load profile for POST /chat against a running Clarion API.

Usage (after ``pip install locust`` in the dev env):

    locust -f loadtest/locustfile.py --host http://localhost:8080

The default scenario opens with a short, well-formed booking request.
We don't randomize over a corpus here — the goal is to stress the
agent loop + middleware + token bucket, not to test scenario
coverage (the eval harness already does that).

Tuning notes:
- The default token bucket allows 10 rps + burst 30 per
  (customer_id, ip). Locust users all hit from one IP, so they
  share one bucket; expect 429s to dominate above ~10 sustained
  rps unless you pass a more permissive ``rate_limiter`` to
  ``create_app``.
- We mix the customer_id between requests so a real multi-tenant
  cluster sees its bucket isolation exercised.
"""

from __future__ import annotations

import os
import random
import uuid

try:
    from locust import HttpUser, between, task  # type: ignore[import-not-found]
except ImportError as e:  # pragma: no cover - locust is an optional dev dep
    raise SystemExit(
        "locust is not installed. Install it with: pip install locust"
    ) from e

_PROMPTS = (
    "I'd like to book an appointment for next Tuesday.",
    "Can you tell me your hours?",
    "Do you accept Aetna?",
    "I'd like to reschedule my appointment to next week please.",
    "What's the difference between a routine eye exam and a comprehensive one?",
)
_CUSTOMERS = ("ophthalmology", "orthopedics")


class ClarionChatUser(HttpUser):
    """One simulated client — repeatedly POSTs /chat with a random prompt."""

    wait_time = between(0.5, 2.5)

    def on_start(self) -> None:
        # Each simulated user gets one stable conversation id so the
        # session pool exercises caching + transcript continuity.
        self._conversation_id = f"loadtest-{uuid.uuid4().hex[:12]}"
        self._customer_id = random.choice(_CUSTOMERS)

    @task
    def post_chat(self) -> None:
        payload = {
            "customer_id": self._customer_id,
            "conversation_id": self._conversation_id,
            "message": random.choice(_PROMPTS),
        }
        with self.client.post(
            "/chat",
            json=payload,
            name="POST /chat",
            catch_response=True,
        ) as resp:
            if resp.status_code == 429:
                # 429 is an expected outcome under load; mark it but
                # don't count it as a failure.
                resp.success()
                return
            if resp.status_code != 200:
                resp.failure(f"unexpected status {resp.status_code}: {resp.text[:200]}")
                return
            try:
                data = resp.json()
            except ValueError:
                resp.failure("response body was not JSON")
                return
            if not data.get("reply"):
                resp.failure("response missing 'reply'")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
