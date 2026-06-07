"""``POST /chat`` — one user turn for one conversation.

Reuses the session manager so the rolling transcript persists between
requests. Errors are mapped to a uniform ErrorResponse shape so clients
get a predictable contract.
"""

from __future__ import annotations

import logging

from clarion.config import CustomerConfigError, CustomerNotFoundError
from fastapi import APIRouter, HTTPException, Request

from api.schemas import ChatRequest, ChatResponse, ErrorResponse

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

    reply = agent.chat(body.message)
    return ChatResponse(
        customer_id=body.customer_id,
        conversation_id=conversation_id,
        trace_id=agent.last_trace_id,
        reply=reply,
    )
