"""``cancel_appointment`` — cancel one appointment by id.

Wraps ``StructuredStore.cancel_appointment``. Idempotent: cancelling an
already-cancelled (or never-existed) appointment returns ``ok=True`` with
``cancelled=False`` so the agent can read it as "nothing to do" rather
than a hard error.
"""

from __future__ import annotations

from typing import ClassVar

from clarion.schemas.tools import (
    CancelAppointmentInput,
    CancelAppointmentOutput,
)
from clarion.tools.base import ToolContext, run_with_retry


class CancelAppointmentTool:
    """Cancel an appointment and free its slot for re-booking."""

    name: ClassVar[str] = "cancel_appointment"
    input_model: ClassVar[type[CancelAppointmentInput]] = CancelAppointmentInput
    output_model: ClassVar[type[CancelAppointmentOutput]] = CancelAppointmentOutput

    def run(self, input: CancelAppointmentInput, ctx: ToolContext) -> CancelAppointmentOutput:
        try:
            cancelled = run_with_retry(
                lambda: ctx.structured.cancel_appointment(input.appointment_id)
            )
        except Exception as e:
            return CancelAppointmentOutput(
                ok=False,
                error=f"cancel_appointment failed: {e}",
                cancelled=False,
            )
        return CancelAppointmentOutput(ok=True, cancelled=cancelled)
