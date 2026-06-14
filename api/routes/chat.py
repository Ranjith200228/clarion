"""``POST /chat`` — one user turn for one conversation.

Reuses the session manager so the rolling transcript persists between
requests. Errors are mapped to a uniform ErrorResponse shape so clients
get a predictable contract.
"""

from __future__ import annotations

import contextlib
import logging

from clarion.config import CustomerConfigError, CustomerNotFoundError
from fastapi import APIRouter, HTTPException, Request

from api.schemas import ChatRequest, ChatResponse, ErrorResponse, LastTurnMetrics

log = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["agent"],
    summary="Send one user turn through the agent",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid customer or bad input"},
        404: {"model": ErrorResponse, "description": "Customer config not found"},
    },
)
def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Advance one conversation by one user turn.

    If ``conversation_id`` is omitted, a new one is allocated and returned
    in the response so the client can include it in the next turn to
    preserve transcript continuity.
    """
    sessions = request.app.state.sessions
    try:
        conversation_id, agent = sessions.get_or_create_session(
            body.customer_id, body.conversation_id
        )
    except CustomerNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"detail": str(e), "code": "customer_not_found"},
        ) from e
    except CustomerConfigError as e:
        raise HTTPException(
            status_code=400,
            detail={"detail": str(e), "code": "customer_config_invalid"},
        ) from e

    try:
        reply = agent.chat(body.message)
    except Exception as e:
        # Production should never 500 from a misconfigured / revoked /
        # rate-limited LLM key. Surface a clean Markdown reply
        # explaining what went wrong; the dashboard renders it like
        # any other turn. Trust engine + traces + audit ran up to
        # this point.
        log.warning(
            "agent.chat failed — returning soft error to client",
            extra={"error_class": type(e).__name__, "error": str(e)[:300]},
        )
        reply = _soft_error_reply(e)
    metrics = _build_last_turn_metrics(agent)
    return ChatResponse(
        customer_id=body.customer_id,
        conversation_id=conversation_id,
        trace_id=agent.last_trace_id,
        reply=reply,
        last_turn_metrics=metrics,
    )


def _soft_error_reply(exc: BaseException) -> str:
    """Map an LLM-side exception to a friendly Markdown chat bubble.

    Distinguishes the three failure modes a public demo actually
    hits: bad/revoked key, quota / rate limit, generic upstream
    fault. Anything else lands in a generic-but-honest catch-all.
    The class name match means we don't take a hard dep on the
    openai SDK's exception types here.
    """
    name = type(exc).__name__
    msg = str(exc)
    auth_hits = ("Authentication", "401", "Unauthorized", "invalid_api_key")
    quota_hits = ("RateLimit", "429", "insufficient_quota", "rate_limit", "QuotaExceeded")
    if any(h in name or h in msg for h in auth_hits):
        return (
            "**The OpenAI API key for this Space looks invalid or revoked.**\n\n"
            "If you're the owner: open the Space's **Settings → Variables "
            "and secrets**, replace `OPENAI_API_KEY` with a current "
            "`sk-...` value, then restart the Space.\n\n"
            "If you're a visitor: nothing you can do — the rest of the "
            "dashboard still works without a key (Quality Metrics, "
            "Escalations, Trace Explorer).\n\n"
            "*(Trust engine + traces + audit ran fine up to the LLM "
            "call. Only the model call itself failed.)*"
        )
    if any(h in name or h in msg for h in quota_hits):
        return (
            "**The OpenAI account behind this Space hit a rate limit or "
            "quota cap.**\n\n"
            "Try again in a minute. If the limit persists, the Space "
            "owner needs to top up their OpenAI billing.\n\n"
            "*(The other dashboard tabs work without the LLM — Quality "
            "Metrics, Escalations, Trace Explorer are all live.)*"
        )
    return (
        "**The model call failed.**\n\n"
        f"Error class: `{name}`. The Space's structured logs have the "
        "full trace under the request's `X-Request-Id`.\n\n"
        "If this keeps happening, the read-only tabs (Quality Metrics, "
        "Escalations, Trace Explorer) still render the locked-schema "
        "evaluation reports without the LLM."
    )


def _build_last_turn_metrics(agent) -> LastTurnMetrics:  # type: ignore[no-untyped-def]
    """Pull the just-completed turn's spans into a LastTurnMetrics shape.

    Phase 14 Live Agent tab renders these fields. Pulled directly off
    the in-memory Tracer the Agent owns — no trace file reread, no
    cross-process state needed.
    """
    # Track the most recent agent.chat span (last item in turn_spans by
    # the convention used by the Tracer).
    turn_spans = getattr(agent, "_last_turn_spans", None)
    escalation_score: float | None = None
    last_tool_call: str | None = None
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    if turn_spans:
        for span in turn_spans:
            attrs = getattr(span, "attributes", None) or {}
            name = getattr(span, "name", "")
            if name == "llm.complete":
                input_tokens += int(attrs.get("input_tokens", 0) or 0)
                output_tokens += int(attrs.get("output_tokens", 0) or 0)
                cost_usd += float(attrs.get("cost_usd", 0.0) or 0.0)
            elif name.startswith("tool."):
                # last_tool_call = the most recent fired tool name.
                last_tool_call = name.removeprefix("tool.")
            elif name == "agent.chat":
                score = attrs.get("escalation_score")
                if score is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        escalation_score = float(score)
    return LastTurnMetrics(
        escalation_score=escalation_score,
        last_tool_call=last_tool_call,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost_usd, 6),
    )
