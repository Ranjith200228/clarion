"""Voice Agent tab — speak to Clarion, hear it speak back.

The tab is a thin Gradio shell over ``POST /voice/turn``:

  gr.Microphone (record + upload audio bytes)
       │
       ▼
  VoiceClient.turn  -- HTTP -->  FastAPI /voice/turn
       │                              │
       │                              ▼
       │                       VoiceOrchestrator
       │                              │
       │           OpenAIWhisperTranscriber -> Agent -> OpenAITtsSpeaker
       │                              │
       ▼                              ▼
  gr.Audio (autoplays response)   transcript + per-stage latency
       │
       ▼
  gr.Markdown shows: heard, reply, STT/agent/TTS latencies

When the backend has no orchestrator wired (no OPENAI_API_KEY on the
deployment), the route 503s and we render a clear "voice not
configured" bubble pointing the visitor at the Settings flow that
turns it on.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
import wave
from dataclasses import dataclass

import gradio as gr
import numpy as np

from gradio_app.agent_client import ApiError
from gradio_app.voice_client import VoiceClient

log = logging.getLogger(__name__)


_DEMO_BUBBLE_TEXT = (
    "**Voice tab in demo mode** — this Space's backend has no voice "
    "orchestrator wired in (no `OPENAI_API_KEY` was set when the FastAPI "
    "service booted).\n\n"
    "What this tab *would* do when live:\n\n"
    "1. **Speak** — your browser captures audio from the mic.\n"
    "2. **Transcribe** — OpenAI Whisper (`whisper-1`) converts it to text "
    "server-side.\n"
    "3. **Reason** — the same Clarion agent that powers the Live Agent tab "
    "answers the request, with the Sentinel trust engine in the loop.\n"
    "4. **Speak back** — OpenAI TTS (`tts-1`) reads the reply and the "
    "audio plays here.\n\n"
    "Latency, transcript, and per-stage timing all surface below the "
    "audio player. To enable: set `OPENAI_API_KEY` in **Settings → "
    "Variables and secrets** and restart the Space."
)


_HEADER_TEXT = (
    "## Voice Agent\n"
    "Speak to Clarion. The browser records, OpenAI Whisper transcribes, "
    "the agent answers, and OpenAI TTS reads the reply back. Per-stage "
    "latency surfaces below each turn.\n"
)


@dataclass
class VoiceAgentState:
    """One Gradio session's voice context.

    ``session_id`` is the conversation handle the API uses to keep
    transcripts coherent across multiple voice turns. The customer
    switcher at the top of the page resets it (a new tenant ID
    starts a fresh conversation).
    """

    customer_id: str = "ophthalmology"
    session_id: str = ""


@dataclass
class VoiceAgentTab:
    state: gr.State
    metrics_md: gr.Markdown
    mic_input: gr.Audio
    audio_output: gr.Audio
    transcript_md: gr.Markdown


def build(client: VoiceClient | None = None) -> VoiceAgentTab:
    """Compose the Voice Agent tab.

    Pass ``client`` for tests. Production callers let the function
    instantiate the default VoiceClient pointed at ``CLARION_API_URL``.
    """
    api = client or VoiceClient()

    gr.Markdown(_HEADER_TEXT)
    state = gr.State(value=VoiceAgentState())
    metrics_md = gr.Markdown(
        "_Press record below, speak a request, then stop to submit._"
    )

    with gr.Row():
        # ``sources=["microphone"]`` opens the browser mic-record UI;
        # ``type="numpy"`` hands us (sample_rate, np.ndarray) so we
        # can WAV-frame it ourselves without an extra Gradio dep.
        mic_input = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="Speak your request",
            interactive=True,
        )
        audio_output = gr.Audio(
            label="Clarion's reply",
            autoplay=True,
            interactive=False,
        )

    transcript_md = gr.Markdown(
        "_Transcript + assistant reply will appear here after each turn._"
    )

    # Submit button is explicit so the user can re-record before
    # sending — the auto-submit on stop pattern fires too eagerly
    # on quiet rooms.
    submit_btn = gr.Button("Send voice turn", variant="primary")
    submit_btn.click(
        fn=_handle_turn(api),
        inputs=[mic_input, state],
        outputs=[audio_output, transcript_md, state],
    )

    return VoiceAgentTab(
        state=state,
        metrics_md=metrics_md,
        mic_input=mic_input,
        audio_output=audio_output,
        transcript_md=transcript_md,
    )


# ---------- handler ----------


def _handle_turn(api: VoiceClient):  # type: ignore[no-untyped-def]
    """Closure that captures the VoiceClient for the Gradio callback."""

    def _on_submit(
        audio: tuple[int, np.ndarray] | None,
        state: VoiceAgentState,
    ) -> tuple[str | None, str, VoiceAgentState]:
        if audio is None:
            return (
                None,
                "**No audio recorded yet** — press record, speak, then send.",
                state,
            )
        sample_rate, samples = audio
        wav_bytes = _numpy_to_wav_bytes(samples, sample_rate)

        # Allocate a session id on the first turn; reuse it from then on.
        if not state.session_id:
            state.session_id = f"voice_{uuid.uuid4().hex[:12]}"

        try:
            result = api.turn(
                customer_id=state.customer_id,
                session_id=state.session_id,
                audio=wav_bytes,
                audio_format="wav",
                sample_rate_hz=sample_rate,
            )
        except ApiError as e:
            msg = str(e)
            if "503" in msg and "voice_not_configured" in msg:
                # The polished demo-mode bubble — recruiters arriving
                # without an OPENAI_API_KEY should see what the tab
                # would do, not a stack trace.
                return None, _DEMO_BUBBLE_TEXT, state
            return None, f"**Voice backend error**: {msg}", state

        out_audio_path = _write_temp_audio(result.audio_bytes, result.audio_format)
        markdown = (
            f"**You said:** {result.transcript or '_(empty transcript)_'}\n\n"
            f"**Clarion:** {result.assistant_text or '_(empty reply)_'}\n\n"
            f"---\n"
            f"_Latency_ — STT: {result.latency_ms_stt} ms · "
            f"Agent: {result.latency_ms_agent} ms · "
            f"TTS: {result.latency_ms_tts} ms · "
            f"_session: `{state.session_id}`_"
        )
        return out_audio_path, markdown, state

    return _on_submit


# ---------- audio helpers ----------


def _numpy_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode a numpy mic sample buffer as a WAV bytes object.

    Gradio's ``gr.Audio(type="numpy")`` hands back float32 in [-1, 1]
    OR int16 depending on the source — we coerce to int16 mono so the
    Whisper endpoint gets a stable PCM format. Two-channel inputs get
    mixed down to mono by averaging.
    """
    arr = samples
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if arr.dtype != np.int16:
        # float32 -> int16 with clipping. We don't trust callers to
        # have already clipped to [-1, 1].
        clipped = np.clip(arr, -1.0, 1.0)
        arr = (clipped * 32767.0).astype(np.int16)

    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(arr.tobytes())
    return buf.getvalue()


def _write_temp_audio(data: bytes, audio_format: str) -> str:
    """Write the TTS bytes to a temp file Gradio can serve.

    Gradio's gr.Audio accepts either (sample_rate, np.ndarray) or a
    filepath. mp3 / wav / ogg are all detected by the browser via the
    file extension, so we suffix the temp file accordingly.
    """
    suffix = "." + audio_format.lower().lstrip(".")
    fd, path = tempfile.mkstemp(prefix="clarion_tts_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except Exception:
        os.unlink(path)
        raise
    return path
