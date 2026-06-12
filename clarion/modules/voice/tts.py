"""Text-to-speech adapters for Module M5.

Two implementations sit behind ``SpeakerProtocol``:

* ``OpenAITtsSpeaker`` — production path. Calls the OpenAI audio
  speech endpoint (``tts-1`` / ``tts-1-hd``). The ``openai``
  client is already a core dep so we don't lazy-import here —
  but the call only happens if a real speaker is constructed.
* ``SineWaveSpeaker`` — deterministic test stub. Produces a tiny
  WAV-shaped PCM buffer (44.1 kHz, ~50 ms) whose duration scales
  with text length. Lets the orchestrator + API tests assert on
  audio_metadata without burning an OpenAI call.

The Protocol is one method: ``synthesize(text) -> (bytes, AudioMetadata)``.
Returning the metadata alongside the bytes means the orchestrator
doesn't have to sniff the payload — the speaker is the source of
truth for format + sample rate.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Literal, Protocol

from clarion.schemas import AudioMetadata

OpenAITtsFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class SpeakerProtocol(Protocol):
    """Convert assistant text into audio bytes + metadata."""

    @property
    def version(self) -> str:
        ...

    def synthesize(self, text: str) -> tuple[bytes, AudioMetadata]:
        ...


@dataclass(frozen=True)
class SineWaveSpeaker:
    """Deterministic test stub: ~50 ms 440 Hz tone per word.

    Generates a real WAV byte stream (RIFF header + PCM samples) so
    callers that downstream-decode the audio still see a valid file.
    Duration scales with text length so tests can sanity-check the
    audio_metadata.duration_ms field.
    """

    sample_rate_hz: int = 16000
    ms_per_word: int = 50
    speaker_version: str = "sine-1.0"

    @property
    def version(self) -> str:
        return self.speaker_version

    def synthesize(self, text: str) -> tuple[bytes, AudioMetadata]:
        n_words = max(1, len(text.split()))
        duration_ms = self.ms_per_word * n_words
        n_samples = int(self.sample_rate_hz * duration_ms / 1000)
        pcm = _sine_pcm16(n_samples, freq_hz=440.0, sample_rate_hz=self.sample_rate_hz)
        wav = _wrap_wav_pcm16(pcm, sample_rate_hz=self.sample_rate_hz)

        meta = AudioMetadata(
            format="wav",
            sample_rate_hz=self.sample_rate_hz,
            duration_ms=duration_ms,
            n_bytes=len(wav),
        )
        return wav, meta


class OpenAITtsSpeaker:
    """OpenAI ``audio.speech`` speaker.

    Defaults to the small ``tts-1`` model + alloy voice + mp3 format
    so the response is small enough for JSON round-trips after base64
    framing. ``duration_ms`` is approximated from the byte count
    (mp3 averages ~16 kbps at this preset) because the API doesn't
    surface duration directly and we don't want to pull a decoder
    in just to measure it.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: OpenAITtsFormat = "mp3",
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._voice = voice
        self._format = response_format

    @property
    def version(self) -> str:
        return f"openai-tts:{self._model}:{self._voice}"

    def synthesize(self, text: str) -> tuple[bytes, AudioMetadata]:
        resp = self._client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format=self._format,
        )
        audio = resp.read()
        # tts-1 mp3 hits ~16 kbps at default quality; close enough
        # for a duration estimate without pulling in a decoder.
        approx_ms = int(len(audio) * 8 / 16) if self._format == "mp3" else 0

        meta = AudioMetadata(
            format=_to_audio_format(self._format),
            sample_rate_hz=24000,  # OpenAI tts-1 native
            duration_ms=approx_ms,
            n_bytes=len(audio),
        )
        return audio, meta


def _to_audio_format(fmt: OpenAITtsFormat) -> Literal["wav", "mp3", "ogg", "webm"]:
    """Map OpenAI's response_format set to the narrower AudioFormat
    union. OpenAI's pcm / opus / flac / aac responses fall back to
    "wav" / "ogg" / "ogg" / "mp3" for the wire metadata so downstream
    decoders only have to handle four cases."""
    if fmt in {"mp3", "aac"}:
        return "mp3"
    if fmt in {"opus", "flac"}:
        return "ogg"
    return "wav"


# ---------- WAV helpers ----------


def _sine_pcm16(n_samples: int, *, freq_hz: float, sample_rate_hz: int) -> bytes:
    """Generate n_samples of int16 PCM at the given frequency."""
    out = bytearray()
    amplitude = 0.3 * 32767  # leave headroom
    for i in range(n_samples):
        sample = int(amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate_hz))
        out += struct.pack("<h", sample)
    return bytes(out)


def _wrap_wav_pcm16(pcm: bytes, *, sample_rate_hz: int) -> bytes:
    """Wrap raw int16 PCM in a minimal RIFF/WAVE header.

    Single channel, 16-bit. The header sizes are computed from the
    payload so the resulting bytes parse cleanly in audio libraries
    that round-trip the test stub.
    """
    n_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate_hz * n_channels * bits_per_sample // 8
    block_align = n_channels * bits_per_sample // 8
    data_size = len(pcm)
    riff_size = 36 + data_size

    header = (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)  # PCM fmt chunk size
        + struct.pack("<H", 1)  # AudioFormat = PCM
        + struct.pack("<H", n_channels)
        + struct.pack("<I", sample_rate_hz)
        + struct.pack("<I", byte_rate)
        + struct.pack("<H", block_align)
        + struct.pack("<H", bits_per_sample)
        + b"data"
        + struct.pack("<I", data_size)
    )
    return header + pcm
