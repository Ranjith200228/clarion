"""Cancel specialist — cancel an existing appointment."""

from __future__ import annotations

from typing import ClassVar

from clarion.multiagent.specialists.base import Specialist
from clarion.multiagent.state import SpecialistIntent


class CancelSpecialist(Specialist):
    """cancel_appointment + create_pms_task (for ambiguous cancellations)."""

    intent: ClassVar[SpecialistIntent] = "cancel"
    allowed_tools: ClassVar[frozenset[str]] = frozenset(
        {"cancel_appointment", "create_pms_task"}
    )
    persona: ClassVar[str] = (
        "You are the cancellation specialist. Confirm the appointment "
        "id + patient before cancelling. If the caller is vague ('cancel "
        "everything next week'), ask for the specific appointment first; "
        "if they refuse to disambiguate, create a front-desk task and "
        "explain you can't cancel without confirmation."
    )
