"""FastAPI service for Clarion.

Exposes the configurable multi-agent over HTTP:

- POST /chat       — single user turn for one conversation
- POST /evaluate   — run a scripted sequence of messages, return metrics
- GET  /health     — liveness probe (used by Cloud Run / k8s)

The app is built via ``create_app()`` so tests can spin up isolated
instances pointing at tmp data dirs.
"""

from api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    EvaluateMetrics,
    EvaluateRequest,
    EvaluateResponse,
    EvaluateTurn,
    HealthResponse,
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ErrorResponse",
    "EvaluateMetrics",
    "EvaluateRequest",
    "EvaluateResponse",
    "EvaluateTurn",
    "HealthResponse",
]
