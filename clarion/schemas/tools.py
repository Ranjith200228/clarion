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

# ---------------------------------------------------------------
# Patient-detail validators
#
# These three patterns guard the LLM's hand-typed fields against
# obvious junk before the tool round-trips into the store. They
# accept legitimate input formats (international names with
# accents/apostrophes, US + international phone formats, normal
# email addresses) and reject the failure modes we've actually
# seen in production traces (empty strings, single words, "n/a",
# pasted role labels like "Customer").
#
# Strict enough to catch hallucinations, lenient enough to not
# block real callers. Lift to a richer parse step (libphonenumber,
# email-validator) once we have signal that these false-positive.
# ---------------------------------------------------------------

# Two or more name parts: each part is letters/apostrophe/hyphen,
# separated by spaces. Allows "Mary O'Brien", "Jean-Luc Picard",
# "José García". Rejects "Customer", "TBD", "Ranjit" (single word).
_NAME_PATTERN = r"^[A-Za-zÀ-ÖØ-öø-ÿ'\-]{2,}(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'\-]{1,}){1,4}$"

# 7-20 chars of digits + common phone glyphs. Must contain at
# least 7 digits in total. Accepts "(617) 584-1139", "+1 617 584
# 1139", "6175841139". Rejects "ask me", "see notes".
_PHONE_PATTERN = r"^[\d\s\-\(\)\+\.]{7,25}$"

# Minimal RFC-5321-ish email check: local@domain.tld with no
# whitespace and at least one dot in the domain. Good enough to
# catch the LLM dropping "n/a" or "tbd" into the email slot.
_EMAIL_PATTERN = r"^[^\s@]+@[^\s@]+\.[^\s@]{2,}$"


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
    """Inputs the booking specialist passes to ``book_appointment``.

    The three patient-detail fields (``patient_name``, ``patient_phone``,
    ``patient_email``) are required so the LLM cannot complete a
    booking without collecting them - the regex validators ensure
    they look like real values, not hallucinated placeholders like
    "n/a" or single-word free text.

    The booking specialist's persona prompt explicitly requires
    asking the caller for each of these and confirming back before
    the tool call; the schema is the safety net behind that prompt.
    """

    model_config = ConfigDict(extra="forbid")

    slot_id: str = Field(min_length=1, pattern=_ID_PATTERN)
    patient_id: str = Field(min_length=1, pattern=_ID_PATTERN)
    patient_name: str = Field(
        min_length=3,
        max_length=120,
        pattern=_NAME_PATTERN,
        description="Caller's full legal name as confirmed back to them.",
    )
    patient_phone: str = Field(
        min_length=7,
        max_length=25,
        pattern=_PHONE_PATTERN,
        description="Reachable phone number; any common format accepted.",
    )
    patient_email: str = Field(
        min_length=5,
        max_length=254,
        pattern=_EMAIL_PATTERN,
        description="Confirmation email; must look like a real address.",
    )
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
