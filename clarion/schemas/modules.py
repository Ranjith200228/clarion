"""Wire shapes for the post-launch module system.

Each module owns its own subdir under ``clarion/modules/`` and writes
its outputs to ``<data_dir>/<customer_id>/<module_name>/...``. The
schemas here are the on-disk contracts those outputs must satisfy.

Module M1 (PMS Writeback): two files per completed conversation
  summary.json  -> ConversationSummary
  task.json     -> PmsTaskWriteback

Other modules (M3 no-show, M5 voice) will add their own shapes here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PMS_WRITEBACK_SCHEMA_VERSION = "1.0.0"


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
