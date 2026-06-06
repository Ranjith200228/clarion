"""Domain models for the structured pipeline.

These are the shapes the agent and tools traffic in (providers, slots,
eligibility records, booked appointments). They live in ``clarion.schemas``
because they're shared across pipelines, tools, and the API layer.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AppointmentStatus = Literal["booked", "cancelled", "completed", "no_show"]
EligibilityStatus = Literal["active", "inactive", "pending", "unknown"]


class Provider(BaseModel):
    """A clinician / resource that can be booked."""

    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(min_length=1)
    full_name: str = Field(min_length=1)
    specialties: list[str] = Field(min_length=1)
    location: str = Field(min_length=1)
    accepts_new_patients: bool = True


class AvailabilitySlot(BaseModel):
    """One bookable slot for a provider on a given day.

    The agent searches over these via ``StructuredStore.search_slots``.
    """

    model_config = ConfigDict(extra="forbid")

    slot_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    appointment_type: str = Field(min_length=1)
    slot_date: date
    start_time: time
    duration_minutes: int = Field(gt=0, le=240)
    is_booked: bool = False


class Appointment(BaseModel):
    """A confirmed booking — written by ``book_appointment``."""

    model_config = ConfigDict(extra="forbid")

    appointment_id: str = Field(min_length=1)
    slot_id: str = Field(min_length=1)
    patient_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    appointment_type: str = Field(min_length=1)
    starts_at: datetime
    duration_minutes: int = Field(gt=0, le=240)
    status: AppointmentStatus = "booked"
    notes: str | None = None


class EligibilityRecord(BaseModel):
    """Coverage / insurance eligibility for one patient.

    Looked up by ``check_eligibility`` before booking insurance-dependent
    appointment types.
    """

    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(min_length=1)
    payer: str = Field(min_length=1)
    member_id: str = Field(min_length=1)
    status: EligibilityStatus = "unknown"
    plan_name: str | None = None
    effective_date: date | None = None
    termination_date: date | None = None
