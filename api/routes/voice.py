"""``POST /voice/turn`` — one voice round-trip for one conversation.

Speech in (base64 bytes + AudioMetadata) → STT → Agent → TTS →
speech out (base64 bytes + AudioMetadata). Reuses the same
SessionManager as ``/chat`` so a session can mix voice + text
turns and the rolling transcript stays coherent.

The endpoint is mounted unconditionally; the orchestrator + adapter
choice is per-app-state injection so customers with
``modules.voice = False`` simply never instantiate one. The route
returns 503 when no orchestrator is configured on app.state.
"""

from __future__ import annotations

import base64
import binascii
import logging

from clarion.config import CustomerConfigError, CustomerNotFoundError
from clarion.schemas import VoiceTurnRequest, VoiceTurnResponse
from fastapi import APIRouter, HTTPException, Request

from api.schemas import ErrorResponse

log = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/voice/turn",
    response_model=VoiceTurnResponse,
    tags=["voice"],
    summary="Send one voice turn through the agent (Module M5)",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid customer / malformed audio"},
        404: {"model": ErrorResponse, "description": "Customer config not found"},
        503: {"model": ErrorResponse, "description": "Voice module not configured"},
    },
)
def voice_turn(request: Request, body: VoiceTurnRequest) -> VoiceTurnResponse:
    """Run STT → agent → TTS for one base64-framed audio turn.

    ``body.session_id`` is treated as the conversation id so multi-turn
    voice sessions keep a coherent transcript. If the session doesn't
    exist yet, the session manager allocates one — but the client must
    pass the SAME id on the next turn.
    """
    orchestrator = getattr(request.app.state, "voice_orchestrator", None)
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail={
                "detail": (
                    "voice module not configured for this deployment — "
                    "set app.state.voice_orchestrator via create_app() "
                    "to enable POST /voice/turn"
                ),
                "code": "voice_not_configured",
            },
        )

    try:
        audio_in = base64.b64decode(body.audio_b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail={"detail": f"audio_b64 is not valid base64: {e}", "code": "bad_audio_b64"},
        ) from e

    if len(audio_in) != body.audio_metadata.n_bytes:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": (
                    f"audio_metadata.n_bytes={body.audio_metadata.n_bytes} disagrees with "
                    f"decoded payload length {len(audio_in)}"
                ),
                "code": "audio_length_mismatch",
            },
        )

    sessions = request.app.state.sessions
    try:
        _, agent = sessions.get_or_create_session(body.customer_id, body.session_id)
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
        result: VoiceTurnResponse = orchestrator.turn(
            agent,
            audio_in,
            customer_id=body.customer_id,
            session_id=body.session_id,
            sample_rate_hz=body.audio_metadata.sample_rate_hz,
        )
    except Exception as exc:
        exc_name = type(exc).__name__
        # OpenAI auth / quota errors mean the key is missing or invalid —
        # treat them the same as "no orchestrator configured" so the Gradio
        # UI shows the polished demo bubble instead of a raw error.
        if exc_name in {"AuthenticationError", "PermissionDeniedError", "RateLimitError"}:
            log.warning("voice: OpenAI key issue (%s): %s", exc_name, exc)
            raise HTTPException(
                status_code=503,
                detail={
                    "detail": (
                        f"voice OpenAI key unavailable ({exc_name}) — "
                        "set OPENAI_API_KEY in the Space secrets to enable live voice"
                    ),
                    "code": "voice_not_configured",
                },
            ) from exc
        log.exception("voice turn failed for customer=%r session=%r", body.customer_id, body.session_id)
        raise HTTPException(
            status_code=500,
            detail={
                "detail": f"voice turn failed: {exc_name}: {exc}",
                "code": "voice_turn_error",
            },
        ) from exc
    return result
