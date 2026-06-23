"""Tests for the cancel_appointment tool."""

from __future__ import annotations

import pytest
from clarion.schemas.tools import BookAppointmentInput, CancelAppointmentInput
from clarion.tools.base import ToolContext
from clarion.tools.book_appointment import BookAppointmentTool
from clarion.tools.cancel_appointment import CancelAppointmentTool
from pydantic import ValidationError


def _book_first(ctx: ToolContext) -> str:
    out = BookAppointmentTool().run(
        BookAppointmentInput(
            slot_id="slot_demo_1",
            patient_id="pat_demo",
            patient_name="Sasha Petrova",
            patient_phone="(617) 584-1139",
            patient_email="sasha.petrova@example.invalid",
        ),
        ctx,
    )
    assert out.ok and out.appointment is not None
    return out.appointment.appointment_id


def test_cancels_existing_appointment(ctx: ToolContext) -> None:
    appt_id = _book_first(ctx)
    out = CancelAppointmentTool().run(CancelAppointmentInput(appointment_id=appt_id), ctx)
    assert out.ok is True
    assert out.cancelled is True


def test_second_cancel_is_ok_but_cancelled_false(ctx: ToolContext) -> None:
    """Idempotent — the LLM may retry without a hard error."""
    appt_id = _book_first(ctx)
    CancelAppointmentTool().run(CancelAppointmentInput(appointment_id=appt_id), ctx)
    out = CancelAppointmentTool().run(CancelAppointmentInput(appointment_id=appt_id), ctx)
    assert out.ok is True
    assert out.cancelled is False


def test_unknown_appointment_returns_ok_false_flag(ctx: ToolContext) -> None:
    out = CancelAppointmentTool().run(CancelAppointmentInput(appointment_id="ghost"), ctx)
    assert out.ok is True
    assert out.cancelled is False


def test_input_rejects_empty_appointment_id() -> None:
    with pytest.raises(ValidationError):
        CancelAppointmentInput(appointment_id="")


def test_input_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        CancelAppointmentInput(
            appointment_id="appt_x",
            secret="oops",  # type: ignore[call-arg]
        )
