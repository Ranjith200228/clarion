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
    "Speak to Clarion. Tap **record**, talk, then tap **stop** - the "
    "turn auto-submits, the reply plays back, and the mic clears so "
    "you can immediately record the next turn. Per-stage latency "
    "surfaces below each turn.\n"
)

# A "tap" recording shorter than this is treated as accidental and
# dropped silently rather than firing a noisy turn at the backend.
_MIN_RECORDING_SECONDS = 0.6


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
        "_Tap record, talk, tap stop - the turn submits and the reply "
        "plays back automatically._"
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

    # Continuous-conversation UX: auto-submit when the user taps
    # stop on the recorder. A short-recording guard inside the
    # handler drops accidental sub-`_MIN_RECORDING_SECONDS`-second
    # taps before they hit the backend. After every turn the
    # handler returns ``None`` for the mic component so the
    # previous recording clears and the user can tap record again
    # without a manual reset.
    handler = _handle_turn(api)
    mic_input.stop_recording(
        fn=handler,
        inputs=[mic_input, state],
        outputs=[audio_output, transcript_md, state, mic_input],
    )
    # Keep an explicit "Resend last" button as a safety net for
    # when stop_recording doesn't fire (e.g. broken mic, browser
    # quirk) - same handler, same outputs.
    submit_btn = gr.Button("Resend last recording", variant="secondary")
    submit_btn.click(
        fn=handler,
        inputs=[mic_input, state],
        outputs=[audio_output, transcript_md, state, mic_input],
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
    """Closure that captures the VoiceClient for the Gradio callback.

    Returns a 4-tuple ``(reply_audio_path, transcript_md, state,
    mic_clear)``. The last element is always ``None`` on a successful
    turn so the mic_input component clears - this is what makes the
    "tap record again, immediately ready" loop feel continuous.
    """

    def _on_submit(
        audio: tuple[int, np.ndarray] | None,
        state: VoiceAgentState,
    ) -> tuple[str | None, str, VoiceAgentState, None]:
        if audio is None:
            return (
                None,
                "_Waiting for your next recording..._",
                state,
                None,
            )
        sample_rate, samples = audio

        # Short-recording guard: filter out accidental taps before
        # they hit the backend. Whisper would happily transcribe
        # a 200 ms blip into garbage tokens which then derails the
        # agent loop.
        if samples.size == 0 or sample_rate <= 0:
            return None, "_Waiting for your next recording..._", state, None
        duration_s = samples.shape[0] / float(sample_rate)
        if duration_s < _MIN_RECORDING_SECONDS:
            return (
                None,
                f"_Recording was {duration_s:.1f}s - hold and speak for at "
                f"least {_MIN_RECORDING_SECONDS:.1f}s._",
                state,
                None,
            )

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
                # The polished demo-mode bubble - recruiters arriving
                # without an OPENAI_API_KEY should see what the tab
                # would do, not a stack trace.
                return None, _DEMO_BUBBLE_TEXT, state, None
            return None, f"**Voice backend error**: {msg}", state, None

        out_audio_path = _write_temp_audio(result.audio_bytes, result.audio_format)
        markdown = (
            f"**You said:** {result.transcript or '_(empty transcript)_'}\n\n"
            f"**Clarion:** {result.assistant_text or '_(empty reply)_'}\n\n"
            f"---\n"
            f"_Latency_ - STT: {result.latency_ms_stt} ms · "
            f"Agent: {result.latency_ms_agent} ms · "
            f"TTS: {result.latency_ms_tts} ms · "
            f"_session: `{state.session_id}`_\n\n"
            "_Tap the mic again whenever you're ready for the next turn._"
        )
        # Returning None for the mic component clears it so the
        # next record-tap starts from a blank slate.
        return out_audio_path, markdown, state, None

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
