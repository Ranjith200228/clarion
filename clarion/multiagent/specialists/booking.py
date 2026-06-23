"""Booking specialist — search + book + reschedule.

Owns the slot-search + appointment-create flow. Reschedule is folded
in here because it's "cancel then book" with shared context; the
router doesn't need to know that distinction.
"""

from __future__ import annotations

import threading
from typing import ClassVar

from clarion.multiagent.booking_fastpath import BookingFastPath
from clarion.multiagent.specialists.base import Specialist
from clarion.multiagent.state import SpecialistIntent

# Lazily-trained module-level singleton — cheap to train (<1 s),
# deterministic, and shared across all specialist instances so the
# first call in a process pays the training cost once.
_fastpath: BookingFastPath | None = None
_fastpath_lock = threading.Lock()


def _get_fastpath() -> BookingFastPath:
    global _fastpath
    if _fastpath is None:
        with _fastpath_lock:
            if _fastpath is None:
                _fastpath = BookingFastPath.train_default()
    return _fastpath


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
        "available without calling search_slots first.\n\n"
        "Before calling book_appointment you MUST have ALL THREE of:\n"
        "  1. The caller's full legal name (first + last, spelled out).\n"
        "  2. A working phone number where they can be reached.\n"
        "  3. An email address for the confirmation.\n"
        "Ask for each missing piece explicitly - 'Could I get your full "
        "name, please?', 'What's the best phone number to reach you?', "
        "'And an email for the confirmation?'. Read the values back to "
        "the caller and ask them to confirm BEFORE you call book_appointment. "
        "Never invent, guess, or substitute these fields - if the caller "
        "refuses to share one, apologise and escalate to a human; do not "
        "book without all three confirmed values. The book_appointment "
        "tool will reject anything that doesn't look like a real name, "
        "phone number, or email."
    )

    def _build_system_prompt(self, *, user_message: str) -> str:
        """Base prompt + optional fast-path hint from the classifier."""
        base = super()._build_system_prompt(user_message=user_message)
        hint = _get_fastpath().hint_for(user_message)
        if hint:
            return f"{base}\n\n{hint}"
        return base
