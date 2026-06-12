"""Voice orchestrator for Module M5.

Chains the three voice-layer operations into one ``turn``:

    inbound audio  ->  Transcriber.transcribe   ->  text
    text           ->  Agent.chat                ->  assistant text
    assistant text ->  Speaker.synthesize        ->  outbound audio

The orchestrator stays deliberately stateless and ignorant of
session storage / retries / streaming. Those concerns live in the
caller (the FastAPI endpoint, the Gradio mic widget). This module
is "given a turn, run the three adapters, surface the latencies."

Reuses the existing Agent so voice is genuinely a layer over the
same conversation engine — no parallel control path, no duplicated
guardrails or escalation logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from clarion.agents import Agent
from clarion.modules.voice.stt import TranscriberProtocol
from clarion.modules.voice.tts import SpeakerProtocol
from clarion.schemas import (
    AudioMetadata,
    TranscriptionResult,
    VoiceTurnResponse,
)


@dataclass(frozen=True)
class TurnLatencies:
    """Per-stage wall-clock latencies, milliseconds.

    Exposed as a small struct so the API layer can fold them straight
    into the VoiceTurnResponse without re-measuring.
    """

    stt_ms: int
    agent_ms: int
    tts_ms: int


class VoiceOrchestrator:
    """STT -> Agent -> TTS for one voice turn.

    The transcriber + speaker are injected so customers can mix and
    match (faster-whisper STT + OpenAI TTS in prod; echo + sine in
    tests). The Agent is owned by the caller — that keeps multi-turn
    conversations coherent across multiple ``turn`` calls.
    """

    def __init__(
        self,
        *,
        transcriber: TranscriberProtocol,
        speaker: SpeakerProtocol,
    ) -> None:
        self._transcriber = transcriber
        self._speaker = speaker

    @property
    def transcriber_version(self) -> str:
        return self._transcriber.version

    @property
    def speaker_version(self) -> str:
        return self._speaker.version

    def turn(
        self,
        agent: Agent,
        audio: bytes,
        *,
        customer_id: str,
        session_id: str,
        sample_rate_hz: int,
    ) -> VoiceTurnResponse:
        """Run one full voice round-trip.

        ``agent`` is the caller's existing Agent (so multi-turn
        context survives across turns); ``audio`` is the raw inbound
        bytes (the API layer is responsible for the base64 hop).
        """
        transcription, stt_ms = self._timed_transcribe(audio, sample_rate_hz=sample_rate_hz)

        agent_start = time.perf_counter()
        assistant_text = agent.chat(transcription.text)
        agent_ms = _elapsed_ms(agent_start)

        out_audio, out_meta, tts_ms = self._timed_synthesize(assistant_text)

        return VoiceTurnResponse(
            customer_id=customer_id,
            session_id=session_id,
            transcription=transcription,
            assistant_text=assistant_text,
            audio_b64=_b64encode(out_audio),
            audio_metadata=out_meta,
            latency_ms_stt=stt_ms,
            latency_ms_agent=agent_ms,
            latency_ms_tts=tts_ms,
        )

    # ---------- internals ----------

    def _timed_transcribe(
        self, audio: bytes, *, sample_rate_hz: int
    ) -> tuple[TranscriptionResult, int]:
        start = time.perf_counter()
        result = self._transcriber.transcribe(audio, sample_rate_hz=sample_rate_hz)
        return result, _elapsed_ms(start)

    def _timed_synthesize(self, text: str) -> tuple[bytes, AudioMetadata, int]:
        start = time.perf_counter()
        audio, meta = self._speaker.synthesize(text)
        return audio, meta, _elapsed_ms(start)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _b64encode(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")
