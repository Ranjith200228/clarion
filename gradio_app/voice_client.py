"""Thin httpx client for the Voice Agent tab — POST /voice/turn.

Mirrors :mod:`gradio_app.agent_client` but for the voice endpoint.
Same process-isolation rationale: the Gradio tab never imports the
voice module directly. The browser records audio, Gradio hands it to
this client as raw bytes, the client base64-frames it for the JSON
body and unpacks the response.

URL is shared with the chat client via ``CLARION_API_URL``.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field

import httpx

from gradio_app.agent_client import DEFAULT_API_URL, DEFAULT_TIMEOUT_S, ApiError

# Voice turns are larger + slower than /chat — give ourselves
# headroom past the default 30 s so a Whisper + LLM + TTS chain on
# a cold CPU doesn't trip. Evaluated at module import time, not on
# every dataclass instantiation (RUF009).
_VOICE_TIMEOUT_S = max(DEFAULT_TIMEOUT_S, 60.0)

log = logging.getLogger(__name__)


@dataclass
class VoiceTurnReply:
    """One ``/voice/turn`` reply unpacked into render-ready fields."""

    transcript: str
    assistant_text: str
    audio_bytes: bytes
    audio_format: str
    sample_rate_hz: int
    duration_ms: int
    latency_ms_stt: int
    latency_ms_agent: int
    latency_ms_tts: int


@dataclass
class VoiceClient:
    """One-method client for the voice round-trip."""

    base_url: str = DEFAULT_API_URL
    timeout_s: float = field(default=_VOICE_TIMEOUT_S)

    def turn(
        self,
        *,
        customer_id: str,
        session_id: str,
        audio: bytes,
        audio_format: str,
        sample_rate_hz: int,
    ) -> VoiceTurnReply:
        """POST one voice turn and return the unpacked reply.

        Raises :class:`ApiError` on transport failures or non-2xx
        responses so the UI can render a clear inline error.
        """
        payload: dict[str, object] = {
            "customer_id": customer_id,
            "session_id": session_id,
            "audio_b64": base64.b64encode(audio).decode("ascii"),
            "audio_metadata": {
                "format": audio_format,
                "sample_rate_hz": sample_rate_hz,
                # The wire schema demands duration_ms; we don't know
                # it client-side without decoding, so we estimate
                # from the byte count assuming 16-bit PCM at the
                # declared sample rate (good enough for WAV; the
                # server only uses this to detect truncation).
                "duration_ms": max(1, int(len(audio) / max(1, sample_rate_hz * 2) * 1000)),
                "n_bytes": len(audio),
            },
        }
        try:
            response = httpx.post(
                f"{self.base_url.rstrip('/')}/voice/turn",
                json=payload,
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as e:
            raise ApiError(f"voice backend unreachable at {self.base_url}: {e}") from e

        if response.status_code >= 400:
            raise ApiError(
                f"voice backend returned {response.status_code}: {response.text[:300]}"
            )

        data = response.json()
        meta = data.get("audio_metadata") or {}
        transcription = data.get("transcription") or {}
        return VoiceTurnReply(
            transcript=str(transcription.get("text", "")),
            assistant_text=str(data.get("assistant_text", "")),
            audio_bytes=base64.b64decode(str(data.get("audio_b64", ""))),
            audio_format=str(meta.get("format", "wav")),
            sample_rate_hz=int(meta.get("sample_rate_hz", 24000)),
            duration_ms=int(meta.get("duration_ms", 0)),
            latency_ms_stt=int(data.get("latency_ms_stt", 0)),
            latency_ms_agent=int(data.get("latency_ms_agent", 0)),
            latency_ms_tts=int(data.get("latency_ms_tts", 0)),
        )
