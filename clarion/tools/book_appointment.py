"""``book_appointment`` — atomically book one slot for one patient.

Wraps ``StructuredStore.book_slot``. The store treats double-book as an
exceptional case (``SlotAlreadyBookedError``); we surface that to the
agent as a structured ``ok=False`` so the LLM can apologise and offer a
different slot.
"""

from __future__ import annotations

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
        try:
            appt = run_with_retry(
                lambda: ctx.structured.book_slot(
                    slot_id=input.slot_id,
                    patient_id=input.patient_id,
                    notes=input.notes,
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
