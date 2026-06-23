"""``book_appointment`` — atomically book one slot for one patient.

Wraps ``StructuredStore.book_slot``. The store treats double-book as an
exceptional case (``SlotAlreadyBookedError``); we surface that to the
agent as a structured ``ok=False`` so the LLM can apologise and offer a
different slot.

Patient details (name / phone / email) come in validated via
``BookAppointmentInput`` and are persisted into the appointment's
``notes`` column as JSON so the Patient 360 confirmation card,
downstream PMS sync, and audit log all read the same captured values
the caller actually confirmed during the conversation.
"""

from __future__ import annotations

import json
from typing import ClassVar

from clarion.pipelines.structured.store import SlotAlreadyBookedError
from clarion.schemas.tools import (
    BookAppointmentInput,
    BookAppointmentOutput,
)
from clarion.tools.base import ToolContext, run_with_retry


class BookAppointmentTool:
    """Atomically reserve one slot for one patient."""

    name: ClassVar[str] = "book_appointment"
    input_model: ClassVar[type[BookAppointmentInput]] = BookAppointmentInput
    output_model: ClassVar[type[BookAppointmentOutput]] = BookAppointmentOutput

    def run(self, input: BookAppointmentInput, ctx: ToolContext) -> BookAppointmentOutput:
        notes_payload = {
            "patient_name": input.patient_name,
            "patient_phone": input.patient_phone,
            "patient_email": input.patient_email,
        }
        if input.notes:
            notes_payload["caller_notes"] = input.notes
        serialised_notes = json.dumps(notes_payload, separators=(",", ":"))

        try:
            appt = run_with_retry(
                lambda: ctx.structured.book_slot(
                    slot_id=input.slot_id,
                    patient_id=input.patient_id,
                    notes=serialised_notes,
                )
            )
        except SlotAlreadyBookedError as e:
            return BookAppointmentOutput(
                ok=False,
                error=f"slot unavailable: {e}",
                appointment=None,
            )
        except Exception as e:
            return BookAppointmentOutput(
                ok=False,
                error=f"book_appointment failed: {e}",
                appointment=None,
            )
        return BookAppointmentOutput(ok=True, appointment=appt)
