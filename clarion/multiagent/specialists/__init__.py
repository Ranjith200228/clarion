"""Specialist nodes for the LangGraph multi-agent backend.

Each specialist owns a focused tool subset of the customer's
enabled tools and a tight persona. Adding a new specialist:

1. Subclass :class:`Specialist`, set ``intent``, ``allowed_tools``,
   ``persona`` at class scope.
2. Add the new ``SpecialistIntent`` literal to
   :mod:`clarion.multiagent.state`.
3. Wire it into the StateGraph in
   :mod:`clarion.multiagent.runner` (commit 5).
"""

from clarion.multiagent.specialists.base import Specialist, filter_tools
from clarion.multiagent.specialists.booking import BookingSpecialist
from clarion.multiagent.specialists.cancel import CancelSpecialist
from clarion.multiagent.specialists.eligibility import EligibilitySpecialist
from clarion.multiagent.specialists.emergency import (
    EMERGENCY_REPLY,
    EmergencySpecialist,
)
from clarion.multiagent.specialists.info import InfoSpecialist

__all__ = [
    "BookingSpecialist",
    "CancelSpecialist",
    "EMERGENCY_REPLY",
    "EligibilitySpecialist",
    "EmergencySpecialist",
    "InfoSpecialist",
    "Specialist",
    "filter_tools",
]
