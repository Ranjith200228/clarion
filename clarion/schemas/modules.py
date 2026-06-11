"""Wire shapes for the post-launch module system.

Each module owns its own subdir under ``clarion/modules/`` and writes
its outputs to ``<data_dir>/<customer_id>/<module_name>/...``. The
schemas here are the on-disk contracts those outputs must satisfy.

Module M1 (PMS Writeback): two files per completed conversation
  summary.json  -> ConversationSummary
  task.json     -> PmsTaskWriteback

Module M3 (No-Show Prediction): per-appointment risk scores plus a
metadata block describing the trained model.
  predictions.jsonl  -> one NoShowPrediction per line
  metadata.json      -> NoShowModelMetadata for the persisted model

Other modules (M5 voice) will add their own shapes here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PMS_WRITEBACK_SCHEMA_VERSION = "1.0.0"
NO_SHOW_PREDICTION_SCHEMA_VERSION = "1.0.0"
VOICE_SCHEMA_VERSION = "1.0.0"


# ---------- M1: PMS Writeback ----------


SummaryOutcome = Literal[
    "booked",
    "rescheduled",
    "cancelled",
    "info_provided",
    "escalated_emergency",
    "refused_clinical",
    "task_created",
    "unresolved",
]


class ConversationSummary(BaseModel):
    """One conversation's structured summary, written to ``summary.json``.

    Every field except ``customer_id`` + ``conversation_id`` +
    ``schema_version`` + ``generated_at`` + ``outcome`` is optional —
    the extractor surfaces what it can and leaves null when the
    conversation didn't supply enough signal.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=PMS_WRITEBACK_SCHEMA_VERSION, pattern=r"^\d+\.\d+\.\d+$")
    customer_id: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    conversation_id: str = Field(min_length=1)
    generated_at: datetime

    # Patient + intent (best-effort extraction).
    patient_id: str | None = None
    caller_name: str | None = None
    intent: str | None = None
    appointment_type: str | None = None
    appointment_time: datetime | None = None
    payer: str | None = None

    # Outcome bucket (deterministic — derived from agent outcome).
    outcome: SummaryOutcome
    escalated: bool = False

    # Free-text fields, capped so PMS systems with text-column limits
    # don't choke on a verbose conversation.
    notes: str = Field(default="", max_length=2000)
    transcript_preview: str = Field(default="", max_length=500)


WritebackTaskPriority = Literal["normal", "urgent"]
WritebackTaskStatus = Literal["open", "in_progress", "closed"]


class PmsTaskWriteback(BaseModel):
    """One PMS task derived from a conversation, written to ``task.json``.

    Mirrors the existing in-engine PmsTask shape but adds Module M1
    fields the writeback contract demands:
      - schema_version for downstream consumers
      - summary_ref pointing at the sibling summary.json so the PMS
        system can pivot in one click
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=PMS_WRITEBACK_SCHEMA_VERSION, pattern=r"^\d+\.\d+\.\d+$")
    customer_id: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    conversation_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    generated_at: datetime

    # Standard task fields.
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(default="", max_length=4000)
    priority: WritebackTaskPriority = "normal"
    status: WritebackTaskStatus = "open"
    patient_id: str | None = None
    assignee_group: str = Field(default="front_desk", max_length=64)

    # Cross-link back to the sibling summary.
    summary_ref: str = Field(default="summary.json")


# ---------- M3: No-Show Prediction ----------


NoShowRiskBand = Literal["low", "medium", "high"]


class NoShowPrediction(BaseModel):
    """One scored appointment.

    Produced by ``NoShowPredictor.predict_one`` and written to the
    module's ``predictions.jsonl`` stream. The risk band is a coarse
    bucket derived from ``p_no_show`` so downstream front-desk UIs
    can colour-code without re-deriving thresholds.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default=NO_SHOW_PREDICTION_SCHEMA_VERSION, pattern=r"^\d+\.\d+\.\d+$"
    )
    customer_id: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    appointment_id: str = Field(min_length=1)
    patient_id: str | None = None
    generated_at: datetime
    model_version: str = Field(min_length=1)

    p_no_show: float = Field(ge=0.0, le=1.0)
    risk_band: NoShowRiskBand


class NoShowModelMetadata(BaseModel):
    """Persisted alongside the joblib bundle so we can audit deploys.

    The trainer writes this whenever it persists a model; the
    predictor reads it to surface ``model_version`` on every
    prediction. ``feature_columns`` is the post-one-hot column
    order — the predictor uses it to align incoming feature dicts
    with the booster's expected layout.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default=NO_SHOW_PREDICTION_SCHEMA_VERSION, pattern=r"^\d+\.\d+\.\d+$"
    )
    model_version: str = Field(min_length=1)
    trained_at: datetime
    n_train: int = Field(ge=1)
    n_features: int = Field(ge=1)
    roc_auc_cv: float = Field(ge=0.0, le=1.0)
    top_decile_lift_cv: float = Field(ge=0.0)
    feature_columns: list[str] = Field(min_length=1)
    seed: int


# ---------- M5: Voice Layer ----------


AudioFormat = Literal["wav", "mp3", "ogg", "webm"]


class AudioMetadata(BaseModel):
    """Container metadata for a chunk of audio carried over the API.

    The API layer ferries audio as base64 strings to keep the JSON
    transport simple — this block tells the decoder what it's
    decoding without sniffing the payload.
    """

    model_config = ConfigDict(extra="forbid")

    format: AudioFormat
    sample_rate_hz: int = Field(ge=8000, le=48000)
    duration_ms: int = Field(ge=0)
    n_bytes: int = Field(ge=0)


class TranscriptionResult(BaseModel):
    """One STT pass, structured.

    ``confidence`` is null for transcribers that don't surface one
    (the echo-mode test stub). The orchestrator passes the ``text``
    field straight into the agent loop.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=VOICE_SCHEMA_VERSION, pattern=r"^\d+\.\d+\.\d+$")
    text: str
    language: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    duration_ms: int = Field(ge=0)
    transcriber_version: str = Field(min_length=1)


class VoiceTurnRequest(BaseModel):
    """One inbound voice turn.

    Audio rides as base64 because the JSON wire stays opaque to
    intermediaries (proxies, gateway log scrubbers) — the alternative
    multipart upload bypasses every middleware we care about.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=VOICE_SCHEMA_VERSION, pattern=r"^\d+\.\d+\.\d+$")
    customer_id: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    session_id: str = Field(min_length=1)
    audio_b64: str = Field(min_length=1)
    audio_metadata: AudioMetadata


class VoiceTurnResponse(BaseModel):
    """One outbound voice turn — STT input, agent text, TTS audio."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=VOICE_SCHEMA_VERSION, pattern=r"^\d+\.\d+\.\d+$")
    customer_id: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    session_id: str = Field(min_length=1)
    transcription: TranscriptionResult
    assistant_text: str
    audio_b64: str = Field(min_length=1)
    audio_metadata: AudioMetadata
    latency_ms_stt: int = Field(ge=0)
    latency_ms_agent: int = Field(ge=0)
    latency_ms_tts: int = Field(ge=0)
