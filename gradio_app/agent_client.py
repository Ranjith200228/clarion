"""Thin httpx client for the Phase 14 Live Agent tab.

The tab calls the existing FastAPI ``/chat`` endpoint via HTTP rather
than reaching into ``clarion.agents`` directly. Two reasons:

1. **Process isolation.** The Phase 15 container runs the Gradio app
   and the FastAPI service as two processes; the UI shouldn't depend
   on importing the entire agent runtime.
2. **No business logic in UI.** The agent + tools + guardrails + judge
   stay in the FastAPI process. The Gradio tab is purely a renderer.

URL is configurable via ``CLARION_API_URL`` (default
``http://localhost:8000``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

DEFAULT_API_URL = os.environ.get("CLARION_API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT_S = float(os.environ.get("CLARION_API_TIMEOUT_S", "30"))


@dataclass
class TurnReply:
    """One ``/chat`` reply unpacked into the four fields the UI renders.

    Kept as a plain dataclass (not the API's Pydantic model) because the
    UI doesn't need extra=forbid + min_length-style strictness here —
    the API has already validated the wire shape. This is just
    structured access.
    """

    reply: str
    conversation_id: str
    trace_id: str
    escalation_score: float | None
    last_tool_call: str | None
    cost_usd: float
    input_tokens: int
    output_tokens: int


class ApiError(RuntimeError):
    """Raised when the API returns a non-2xx response or is unreachable."""


@dataclass
class AgentClient:
    """One client per Gradio session is fine; Gradio runs the UI in a
    single process and httpx clients are thread-safe."""

    base_url: str = DEFAULT_API_URL
    timeout_s: float = DEFAULT_TIMEOUT_S

    def chat(
        self,
        *,
        customer_id: str,
        message: str,
        conversation_id: str | None = None,
    ) -> TurnReply:
        """POST one user message to ``/chat`` and return the unpacked reply.

        Raises ``ApiError`` on transport failures or non-2xx responses
        so the UI can render a clear inline error rather than swallow
        the issue.
        """
        payload: dict[str, object] = {
            "customer_id": customer_id,
            "message": message,
        }
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id

        try:
            response = httpx.post(
                f"{self.base_url.rstrip('/')}/chat",
                json=payload,
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as e:
            raise ApiError(f"agent backend unreachable at {self.base_url}: {e}") from e

        if response.status_code >= 400:
            raise ApiError(f"agent backend returned {response.status_code}: {response.text[:200]}")

        data = response.json()
        metrics = data.get("last_turn_metrics") or {}
        return TurnReply(
            reply=str(data.get("reply", "")),
            conversation_id=str(data.get("conversation_id", "")),
            trace_id=str(data.get("trace_id", "")),
            escalation_score=metrics.get("escalation_score"),
            last_tool_call=metrics.get("last_tool_call"),
            cost_usd=float(metrics.get("cost_usd", 0.0) or 0.0),
            input_tokens=int(metrics.get("input_tokens", 0) or 0),
            output_tokens=int(metrics.get("output_tokens", 0) or 0),
        )

    def health(self) -> bool:
        """Quick liveness probe. Returns False instead of raising so the
        UI can show a "backend not reachable" banner."""
        try:
            response = httpx.get(
                f"{self.base_url.rstrip('/')}/health",
                timeout=min(self.timeout_s, 3.0),
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200
