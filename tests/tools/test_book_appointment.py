"""Tests for the book_appointment tool."""

from __future__ import annotations

import pytest
from clarion.schemas.tools import BookAppointmentInput
from clarion.tools.base import ToolContext
from clarion.tools.book_appointment import BookAppointmentTool
from pydantic import ValidationError


def test_books_an_open_slot(ctx: ToolContext) -> None:
    out = BookAppointmentTool().run(
        BookAppointmentInput(slot_id="slot_demo_1", patient_id="pat_demo", notes="callback ok"),
        ctx,
    )
    assert out.ok is True
    assert out.error is None
    assert out.appointment is not None
    assert out.appointment.slot_id == "slot_demo_1"
    assert out.appointment.patient_id == "pat_demo"
    assert out.appointment.status == "booked"
    assert out.appointment.notes == "callback ok"


def test_double_book_returns_structured_error(ctx: ToolContext) -> None:
    first = BookAppointmentTool().run(
        BookAppointmentInput(slot_id="slot_demo_1", patient_id="pat_alpha"),
        ctx,
    )
    assert first.ok is True

    second = BookAppointmentTool().run(
        BookAppointmentInput(slot_id="slot_demo_1", patient_id="pat_beta"),
        ctx,
    )
    assert second.ok is False
    assert second.error is not None
    assert "slot unavailable" in second.error
    assert second.appointment is None


def test_unknown_slot_returns_structured_error(ctx: ToolContext) -> None:
    out = BookAppointmentTool().run(
        BookAppointmentInput(slot_id="ghost_slot", patient_id="pat_demo"),
        ctx,
    )
    assert out.ok is False
    assert out.error is not None
    assert "slot unavailable" in out.error


def test_input_rejects_empty_slot_id() -> None:
    with pytest.raises(ValidationError):
        BookAppointmentInput(slot_id="", patient_id="pat_demo")


def test_input_rejects_empty_patient_id() -> None:
    with pytest.raises(ValidationError):
        BookAppointmentInput(slot_id="slot_demo_1", patient_id="")


def test_input_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        BookAppointmentInput(
            slot_id="slot_demo_1",
            patient_id="pat_demo",
            secret="oops",  # type: ignore[call-arg]
        )


def test_input_rejects_overlong_notes() -> None:
    with pytest.raises(ValidationError):
        BookAppointmentInput(
            slot_id="slot_demo_1",
            patient_id="pat_demo",
            notes="x" * 2001,
        )
