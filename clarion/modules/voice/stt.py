"""Speech-to-text adapters for Module M5.

Two implementations sit behind ``TranscriberProtocol``:

* ``FasterWhisperTranscriber`` — production path. faster-whisper is
  ~1GB once you count ctranslate2 + the model weights, so we
  lazy-import it; the module's own import stays cheap when the
  voice feature is off.
* ``EchoTranscriber`` — deterministic test stub. The "audio" is
  treated as a UTF-8 text payload (the orchestrator's tests
  encode short strings as bytes); whatever shows up in is what
  the transcription claims it heard.

The Protocol intentionally has one method. Adding any retry /
chunking logic here would couple the orchestrator to STT
internals; we'd rather the orchestrator stay thin and bring
its own retry policy if it ever needs one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from clarion.schemas import TranscriptionResult


class TranscriberProtocol(Protocol):
    """Convert audio bytes into a TranscriptionResult."""

    @property
    def version(self) -> str:
        ...

    def transcribe(self, audio: bytes, *, sample_rate_hz: int) -> TranscriptionResult:
        ...


@dataclass(frozen=True)
class EchoTranscriber:
    """UTF-8 echo transcriber for tests + dry-run demos.

    Whatever bytes come in get decoded as UTF-8 and returned as the
    transcription text. ``language`` and ``confidence`` are pinned at
    "en" / None so downstream code never sees fabricated metadata.
    """

    transcriber_version: str = "echo-1.0"

    @property
    def version(self) -> str:
        return self.transcriber_version

    def transcribe(self, audio: bytes, *, sample_rate_hz: int) -> TranscriptionResult:
        text = audio.decode("utf-8", errors="replace")
        return TranscriptionResult(
            text=text,
            language="en",
            confidence=None,
            duration_ms=len(audio),
            transcriber_version=self.transcriber_version,
        )


class FasterWhisperTranscriber:
    """faster-whisper-backed transcriber.

    Imports the heavy dep at construct time so:
      - module imports stay cheap for callers that never instantiate
        a real transcriber (every test path, every customer with
        modules.voice = False)
      - a missing dep raises a clear error at the point of use, not
        at the top of the import graph
    """

    def __init__(self, *, model_size: str = "base.en", device: str = "cpu") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper is not installed — install the optional voice extra "
                "(`poetry install --with voice`) before constructing "
                "FasterWhisperTranscriber"
            ) from e
        self._model = WhisperModel(model_size, device=device, compute_type="int8")
        self._model_size = model_size

    @property
    def version(self) -> str:
        return f"faster-whisper:{self._model_size}"

    def transcribe(self, audio: bytes, *, sample_rate_hz: int) -> TranscriptionResult:
        import io

        start = time.perf_counter()
        # faster-whisper accepts a file-like; an in-memory buffer keeps
        # us off disk and works for the API's base64-decoded bytes.
        buf = io.BytesIO(audio)
        segments, info = self._model.transcribe(buf, beam_size=5)
        text = " ".join(seg.text for seg in segments).strip()
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            confidence=getattr(info, "language_probability", None),
            duration_ms=elapsed_ms,
            transcriber_version=self.version,
        )
