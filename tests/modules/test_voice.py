"""Module M5 tests — schemas, adapters, orchestrator, endpoint."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from clarion.agents.llm import FakeLLM, LLMClient, LLMResponse
from clarion.config import Settings
from clarion.modules.voice import (
    EchoTranscriber,
    SineWaveSpeaker,
    VoiceOrchestrator,
)
from clarion.schemas import (
    AudioMetadata,
    TranscriptionResult,
    VoiceTurnRequest,
    VoiceTurnResponse,
)
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.app import create_app
from api.sessions import make_session_manager

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


# ---------- schemas ----------


def test_audio_metadata_rejects_bad_sample_rate() -> None:
    with pytest.raises(ValidationError):
        AudioMetadata(format="wav", sample_rate_hz=1000, duration_ms=100, n_bytes=10)


def test_voice_turn_request_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        VoiceTurnRequest(  # type: ignore[call-arg]
            customer_id="ophthalmology",
            session_id="sess_001",
            audio_b64="aGVsbG8=",
            audio_metadata=AudioMetadata(
                format="wav", sample_rate_hz=16000, duration_ms=100, n_bytes=5
            ),
            wat="hi",
        )


# ---------- adapters ----------


def test_echo_transcriber_round_trips_utf8() -> None:
    t = EchoTranscriber()
    audio = b"book me a slot please"
    out = t.transcribe(audio, sample_rate_hz=16000)
    assert isinstance(out, TranscriptionResult)
    assert out.text == "book me a slot please"
    assert out.language == "en"
    assert out.confidence is None
    assert out.transcriber_version == "echo-1.0"


def test_sine_speaker_produces_valid_wav_with_scaling_duration() -> None:
    s = SineWaveSpeaker()
    short, short_meta = s.synthesize("hi")
    long_, long_meta = s.synthesize("this is a much longer utterance to synthesize")

    # RIFF header sanity.
    assert short.startswith(b"RIFF") and short[8:12] == b"WAVE"
    assert long_.startswith(b"RIFF") and long_[8:12] == b"WAVE"

    # Format metadata matches the wire shape.
    assert short_meta.format == "wav"
    assert short_meta.sample_rate_hz == 16000

    # Duration scales with word count (50 ms / word).
    assert long_meta.duration_ms > short_meta.duration_ms
    assert long_meta.n_bytes > short_meta.n_bytes


# ---------- orchestrator ----------


def test_orchestrator_chains_stt_agent_tts(
    fake_agent_with_reply: object,
) -> None:
    agent = fake_agent_with_reply  # type: ignore[assignment]
    orch = VoiceOrchestrator(transcriber=EchoTranscriber(), speaker=SineWaveSpeaker())

    inbound = b"book me an eye exam"
    resp = orch.turn(
        agent,  # type: ignore[arg-type]
        inbound,
        customer_id="ophthalmology",
        session_id="sess_voice_001",
        sample_rate_hz=16000,
    )

    assert isinstance(resp, VoiceTurnResponse)
    assert resp.transcription.text == "book me an eye exam"
    assert resp.assistant_text == "AGENT_REPLY:book me an eye exam"
    assert resp.audio_metadata.format == "wav"
    assert resp.latency_ms_stt >= 0
    assert resp.latency_ms_agent >= 0
    assert resp.latency_ms_tts >= 0
    # Outbound audio is already base64-framed.
    decoded = base64.b64decode(resp.audio_b64)
    assert decoded.startswith(b"RIFF")
    assert len(decoded) == resp.audio_metadata.n_bytes


class _StubAgent:
    """Stand-in for clarion.agents.Agent — minimum surface chat() needs."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.last_trace_id = "trace_stub"

    def chat(self, user_message: str) -> str:
        self.calls.append(user_message)
        return f"AGENT_REPLY:{user_message}"


@pytest.fixture
def fake_agent_with_reply() -> _StubAgent:
    return _StubAgent()


# ---------- /voice/turn endpoint ----------


@pytest.fixture
def voice_client(tmp_path: Path) -> Iterator[TestClient]:
    """A TestClient with the voice orchestrator wired in."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Seed the structured store so /voice/turn doesn't crash on first use.
    from clarion.pipelines.structured import StructuredStore
    from clarion.schemas import AvailabilitySlot, EligibilityRecord, Provider

    seeds_dir = REPO_ROOT / "data" / "seeds"
    payload = json.loads((seeds_dir / "ophthalmology.json").read_text(encoding="utf-8"))
    store = StructuredStore.for_customer("ophthalmology", data_dir)
    for p in payload["providers"]:
        store.upsert_provider(Provider(**p))
    for s in payload["availability"]:
        store.upsert_slot(AvailabilitySlot(**s))
    for e in payload["eligibility"]:
        store.upsert_eligibility(EligibilityRecord(**e))

    settings = Settings(customer="ophthalmology", config_dir=CONFIGS_DIR, data_dir=data_dir)
    fake = FakeLLM(
        responses=[
            LLMResponse(content="Sure — I can help with that booking.", tool_calls=[]),
        ]
    )

    def factory() -> LLMClient:
        return fake

    sessions = make_session_manager(settings, llm_factory=factory)
    orch = VoiceOrchestrator(transcriber=EchoTranscriber(), speaker=SineWaveSpeaker())
    app = create_app(settings=settings, sessions=sessions, voice_orchestrator=orch)
    with TestClient(app) as c:
        yield c


def test_voice_endpoint_returns_503_when_no_orchestrator(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings = Settings(customer="ophthalmology", config_dir=CONFIGS_DIR, data_dir=data_dir)
    sessions = make_session_manager(settings, llm_factory=lambda: FakeLLM(responses=[]))
    app = create_app(settings=settings, sessions=sessions)  # no voice_orchestrator
    with TestClient(app) as c:
        payload = b"x"
        body = {
            "customer_id": "ophthalmology",
            "session_id": "sess_001",
            "audio_b64": base64.b64encode(payload).decode("ascii"),
            "audio_metadata": {
                "format": "wav",
                "sample_rate_hz": 16000,
                "duration_ms": 10,
                "n_bytes": len(payload),
            },
        }
        r = c.post("/voice/turn", json=body)
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["code"] == "voice_not_configured"


def test_voice_endpoint_rejects_length_mismatch(voice_client: TestClient) -> None:
    payload = b"x"
    body = {
        "customer_id": "ophthalmology",
        "session_id": "sess_002",
        "audio_b64": base64.b64encode(payload).decode("ascii"),
        "audio_metadata": {
            "format": "wav",
            "sample_rate_hz": 16000,
            "duration_ms": 10,
            "n_bytes": 999,  # wrong!
        },
    }
    r = voice_client.post("/voice/turn", json=body)
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "audio_length_mismatch"


def test_voice_endpoint_happy_path_round_trips(voice_client: TestClient) -> None:
    payload = b"book me an eye exam"
    body = {
        "customer_id": "ophthalmology",
        "session_id": "sess_003",
        "audio_b64": base64.b64encode(payload).decode("ascii"),
        "audio_metadata": {
            "format": "wav",
            "sample_rate_hz": 16000,
            "duration_ms": 100,
            "n_bytes": len(payload),
        },
    }
    r = voice_client.post("/voice/turn", json=body)
    assert r.status_code == 200, r.json()
    resp = r.json()
    assert resp["customer_id"] == "ophthalmology"
    assert resp["session_id"] == "sess_003"
    assert resp["transcription"]["text"] == "book me an eye exam"
    assert resp["assistant_text"] == "Sure — I can help with that booking."
    assert resp["audio_metadata"]["format"] == "wav"
    assert resp["audio_metadata"]["n_bytes"] > 0
    # Outbound audio is a valid base64-encoded WAV.
    decoded = base64.b64decode(resp["audio_b64"])
    assert decoded.startswith(b"RIFF")
