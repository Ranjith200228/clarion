"""Tests for the book_appointment tool."""

from __future__ import annotations

import json

import pytest
from clarion.schemas.tools import BookAppointmentInput
from clarion.tools.base import ToolContext
from clarion.tools.book_appointment import BookAppointmentTool
from pydantic import ValidationError

# Shared good-citizen patient details used across the booking tests
# (kept here so each test reads as a single intent, not a fixture
# dump).
_GOOD_DETAILS = {
    "patient_name": "Sasha Petrova",
    "patient_phone": "(617) 584-1139",
    "patient_email": "sasha.petrova@example.invalid",
}


def _input(**overrides: object) -> BookAppointmentInput:
    """Build a BookAppointmentInput with sensible defaults so each
    test can highlight only the field it cares about."""
    base = {
        "slot_id": "slot_demo_1",
        "patient_id": "pat_demo",
        **_GOOD_DETAILS,
    }
    base.update(overrides)
    return BookAppointmentInput(**base)  # type: ignore[arg-type]


def test_books_an_open_slot(ctx: ToolContext) -> None:
    out = BookAppointmentTool().run(_input(notes="callback ok"), ctx)
    assert out.ok is True
    assert out.error is None
    assert out.appointment is not None
    assert out.appointment.slot_id == "slot_demo_1"
    assert out.appointment.patient_id == "pat_demo"
    assert out.appointment.status == "booked"
    # The store-side notes column now carries a JSON blob with the
    # captured patient details so the Patient 360 confirmation card
    # and any downstream PMS sync read the same confirmed values.
    assert out.appointment.notes is not None
    persisted = json.loads(out.appointment.notes)
    assert persisted["patient_name"] == "Sasha Petrova"
    assert persisted["patient_phone"] == "(617) 584-1139"
    assert persisted["patient_email"] == "sasha.petrova@example.invalid"
    assert persisted["caller_notes"] == "callback ok"


def test_double_book_returns_structured_error(ctx: ToolContext) -> None:
    first = BookAppointmentTool().run(_input(patient_id="pat_alpha"), ctx)
    assert first.ok is True

    second = BookAppointmentTool().run(_input(patient_id="pat_beta"), ctx)
    assert second.ok is False
    assert second.error is not None
    assert "slot unavailable" in second.error
    assert second.appointment is None


def test_unknown_slot_returns_structured_error(ctx: ToolContext) -> None:
    out = BookAppointmentTool().run(_input(slot_id="ghost_slot"), ctx)
    assert out.ok is False
    assert out.error is not None
    assert "slot unavailable" in out.error


def test_input_rejects_empty_slot_id() -> None:
    with pytest.raises(ValidationError):
        _input(slot_id="")


def test_input_rejects_empty_patient_id() -> None:
    with pytest.raises(ValidationError):
        _input(patient_id="")


def test_input_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        BookAppointmentInput(
            slot_id="slot_demo_1",
            patient_id="pat_demo",
            secret="oops",  # type: ignore[call-arg]
            **_GOOD_DETAILS,
        )


def test_input_rejects_overlong_notes() -> None:
    with pytest.raises(ValidationError):
        _input(notes="x" * 2001)


# ---------- patient-detail validation ----------


def test_input_rejects_single_word_name() -> None:
    with pytest.raises(ValidationError):
        _input(patient_name="Customer")


def test_input_rejects_phone_with_words() -> None:
    with pytest.raises(ValidationError):
        _input(patient_phone="ask me later")


def test_input_rejects_bad_email() -> None:
    with pytest.raises(ValidationError):
        _input(patient_email="n/a")


def test_input_accepts_international_name() -> None:
    out = _input(patient_name="Mira O'Brien-Khoury")
    assert out.patient_name == "Mira O'Brien-Khoury"


def test_input_accepts_e164_phone() -> None:
    out = _input(patient_phone="+1 617 584 1139")
    assert out.patient_phone == "+1 617 584 1139"


def test_book_persists_patient_details_in_notes(ctx: ToolContext) -> None:
    """Round-trip: caller-confirmed details land in the appointment's
    notes JSON so downstream views can render the captured values."""
    out = BookAppointmentTool().run(_input(), ctx)
    assert out.ok is True
    assert out.appointment is not None
    persisted = json.loads(out.appointment.notes or "{}")
    assert persisted == {
        "patient_name": "Sasha Petrova",
        "patient_phone": "(617) 584-1139",
        "patient_email": "sasha.petrova@example.invalid",
    }
