"""Speech-to-text adapters for Module M5.

Three implementations sit behind ``TranscriberProtocol``:

* ``OpenAIWhisperTranscriber`` — managed-service path. Calls OpenAI's
  ``audio.transcriptions`` endpoint (whisper-1). Zero extra deps —
  ``openai`` is already in the core install — and zero on-disk model
  weights, so it suits constrained deployments (HF Spaces' CPU
  basic tier) where faster-whisper's ~1GB is a non-starter.
* ``FasterWhisperTranscriber`` — self-hosted path. faster-whisper is
  ~1GB once you count ctranslate2 + the model weights, so we
  lazy-import it; the module's own import stays cheap when the
  voice feature is off. Right for deployments that need no
  third-party calls.
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
from typing import ClassVar, Protocol

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


class OpenAIWhisperTranscriber:
    """OpenAI Whisper-1 transcriber — managed STT over HTTPS.

    Why this is the default for the live Space:

    * No extra runtime dep — the ``openai`` client is already pinned
      for the agent's LLM calls. faster-whisper would add ~1 GB to
      the image and exceed HF Spaces' free-tier disk.
    * Whisper-1 is multilingual out of the box; the agent path can
      stay English-only without the transcriber forcing it.
    * Lazy import of the SDK so this module's own import stays
      cheap when no real transcriber is constructed.
    """

    _MIME_BY_FORMAT: ClassVar[dict[str, str]] = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
        "webm": "audio/webm",
        "m4a": "audio/mp4",
    }

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "whisper-1",
        audio_format: str = "wav",
        language: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover — openai is a core dep
            raise RuntimeError(
                "openai is not installed — install the core deps before "
                "constructing OpenAIWhisperTranscriber"
            ) from e
        import os

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OpenAIWhisperTranscriber requires OPENAI_API_KEY "
                "(env or constructor arg)."
            )
        self._client = OpenAI(api_key=key)
        self._model = model
        self._format = audio_format
        self._language = language
        self._mime = self._MIME_BY_FORMAT.get(audio_format, "application/octet-stream")
        self._filename = f"audio.{audio_format}"

    @property
    def version(self) -> str:
        return f"openai-whisper:{self._model}"

    def transcribe(self, audio: bytes, *, sample_rate_hz: int) -> TranscriptionResult:
        start = time.perf_counter()
        # The SDK accepts a (filename, bytes, mime) tuple for the file
        # argument — that matches multipart upload semantics without
        # ever touching disk. We pass call args inline (not via dict
        # unpack) so the SDK's overload resolution narrows to the
        # non-streaming variant; kwargs-spread would force the
        # union-return type and confuse mypy.
        file_tuple = (self._filename, audio, self._mime)
        if self._language:
            resp = self._client.audio.transcriptions.create(
                model=self._model,
                file=file_tuple,
                language=self._language,
            )
        else:
            resp = self._client.audio.transcriptions.create(
                model=self._model,
                file=file_tuple,
            )
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        text = getattr(resp, "text", "") or ""
        # The transcriptions endpoint doesn't surface a confidence
        # score on whisper-1; leave it None rather than fabricate.
        return TranscriptionResult(
            text=text.strip(),
            language=self._language,
            confidence=None,
            duration_ms=elapsed_ms,
            transcriber_version=self.version,
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
