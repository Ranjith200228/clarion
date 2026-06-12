"""Booking specialist — search + book + reschedule.

Owns the slot-search + appointment-create flow. Reschedule is folded
in here because it's "cancel then book" with shared context; the
router doesn't need to know that distinction.
"""

from __future__ import annotations

from typing import ClassVar

from clarion.multiagent.specialists.base import Specialist
from clarion.multiagent.state import SpecialistIntent


class BookingSpecialist(Specialist):
    """search_slots + book_appointment + cancel_appointment."""

    intent: ClassVar[SpecialistIntent] = "booking"
    allowed_tools: ClassVar[frozenset[str]] = frozenset(
        {"search_slots", "book_appointment", "cancel_appointment"}
    )
    persona: ClassVar[str] = (
        "You are the booking specialist for a specialty medical practice. "
        "Your job is to find slots that fit the caller's request, confirm "
        "the choice, and book it. For reschedules: cancel the existing "
        "appointment first, then book the new one. Ask one clarifying "
        "question at a time when the request is ambiguous (date range, "
        "appointment type, provider preference). Never claim a slot is "
        "available without calling search_slots first."
    )
