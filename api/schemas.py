"""Request/response schemas for the FastAPI service.

These are deliberately separate from the internal Pydantic models in
``clarion.schemas`` — the API surface is a contract with HTTP clients and
should be free to evolve independently. We also strip PHI patterns from
echo fields in the response if needed (Phase 14 hardening).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------- /chat ----------


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=4000)
    # Optional — when None, a new conversation_id is allocated and returned.
    conversation_id: str | None = Field(default=None, min_length=1)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str
    conversation_id: str
    trace_id: str
    reply: str


# ---------- /evaluate ----------


class EvaluateRequest(BaseModel):
    """Run a scripted sequence of user messages through the agent.

    Each message advances the same conversation in order. This is the
    Phase 8 placeholder for the richer scenario format Phase 9 will
    produce.
    """

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    messages: list[str] = Field(min_length=1, max_length=20)
    # Optional label so the dashboard (Phase 13) can group reruns.
    scenario_id: str | None = Field(default=None, min_length=1)


class EvaluateTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_message: str
    agent_reply: str


class EvaluateMetrics(BaseModel):
    """Aggregate stats across the scenario's turns."""

    model_config = ConfigDict(extra="forbid")

    turns: int
    total_steps: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    total_latency_ms: float
    tools_used: dict[str, int]  # tool name → count


class EvaluateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str
    scenario_id: str | None
    conversation_id: str
    trace_ids: list[str]
    transcript: list[EvaluateTurn]
    metrics: EvaluateMetrics


# ---------- /health ----------


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str  # "ok" | "degraded"
    version: str
    customers_loaded: list[str]


# ---------- error envelope ----------


class ErrorResponse(BaseModel):
    """Uniform error shape — easier for clients than FastAPI's default."""

    model_config = ConfigDict(extra="forbid")

    detail: str
    code: str
    extra: dict[str, Any] | None = None
