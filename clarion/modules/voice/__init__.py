"""Module M5: Voice Layer.

Round-trip: inbound audio -> STT -> agent.chat -> TTS -> outbound
audio. The orchestrator chains those three adapters; the API layer
exposes a single POST /voice/turn endpoint.

Heavy deps (faster-whisper, openai.audio) are lazy-imported so
clients with ``modules.voice = False`` don't pay the cold-start cost.

Public surface (built up across commits):
  TranscriberProtocol, EchoTranscriber, FasterWhisperTranscriber  (commit 2)
  SpeakerProtocol, SineWaveSpeaker, OpenAITtsSpeaker              (commit 3)
  VoiceOrchestrator                                               (commit 4)
"""

from clarion.modules.voice.stt import (
    EchoTranscriber,
    FasterWhisperTranscriber,
    TranscriberProtocol,
)

__all__ = [
    "EchoTranscriber",
    "FasterWhisperTranscriber",
    "TranscriberProtocol",
]
