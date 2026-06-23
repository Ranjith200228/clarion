"""Tool input/output schemas.

Every Clarion tool follows the same shape:

* a Pydantic ``*Input`` model — what the LLM passes in
* a Pydantic ``*Output`` model — always carries an ``ok`` + ``error`` so
  the agent can recover from failures without exception handling
* both are ``extra="forbid"`` so the LLM can't smuggle unknown fields

These schemas live in ``clarion.schemas.tools`` (not next to the tool
implementations) so they're importable from the API layer and the agent
layer without dragging in tool execution code.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from clarion.schemas.domain import (
    Appointment,
    AvailabilitySlot,
    EligibilityRecord,
)

TaskPriority = Literal["normal", "urgent"]

# Pydantic regex for "looks like an identifier" — must start with a
# letter, then letters/digits/underscores/dashes, max 64 chars. This
# rejects free-form text the LLM might hallucinate into an ID slot
# (e.g. a patient's full name with spaces), which previously slipped
# straight into the structured store and corrupted downstream views.
# Update only with a careful migration — the orthopedics db has had
# at least one row wiped by this guard before.
_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_\-]{0,63}$"


class ToolOutput(BaseModel):
    """Base for every tool's output. Tools never raise to the agent — they
    return ``ok=False`` with a human-readable error string so the LLM can
    decide what to do next (retry with different args, escalate, etc).
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool
    error: str | None = None


# ---------- search_slots ----------


class SearchSlotsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    appointment_type: str = Field(min_length=1)
    on_or_after: date
    provider_id: str | None = None
    limit: int = Field(default=5, ge=1, le=20)


class SearchSlotsOutput(ToolOutput):
    slots: list[AvailabilitySlot] = Field(default_factory=list)


# ---------- book_appointment ----------


class BookAppointmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_id: str = Field(min_length=1, pattern=_ID_PATTERN)
    patient_id: str = Field(min_length=1, pattern=_ID_PATTERN)
    notes: str | None = Field(default=None, max_length=2000)


class BookAppointmentOutput(ToolOutput):
    appointment: Appointment | None = None


# ---------- cancel_appointment ----------


class CancelAppointmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    appointment_id: str = Field(min_length=1, pattern=_ID_PATTERN)


class CancelAppointmentOutput(ToolOutput):
    cancelled: bool = False


# ---------- check_eligibility ----------


class CheckEligibilityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(min_length=1, pattern=_ID_PATTERN)


class CheckEligibilityOutput(ToolOutput):
    record: EligibilityRecord | None = None
    on_file: bool = False


# ---------- create_pms_task ----------


class CreatePmsTaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=4000)
    patient_id: str | None = None
    priority: TaskPriority = "normal"


class CreatePmsTaskOutput(ToolOutput):
    task_id: str | None = None
