"""Eligibility specialist — payer / insurance / coverage questions."""

from __future__ import annotations

from typing import ClassVar

from clarion.multiagent.specialists.base import Specialist
from clarion.multiagent.state import SpecialistIntent


class EligibilitySpecialist(Specialist):
    """check_eligibility + create_pms_task (for follow-up)."""

    intent: ClassVar[SpecialistIntent] = "eligibility"
    allowed_tools: ClassVar[frozenset[str]] = frozenset(
        {"check_eligibility", "create_pms_task"}
    )
    persona: ClassVar[str] = (
        "You are the eligibility specialist. Confirm whether the caller "
        "is covered by their stated payer for the requested service. "
        "Ask for the patient id + payer first when not provided; never "
        "guess. If the check returns ineligible / unknown, offer to "
        "create a front-desk task so a human can follow up — don't "
        "leave the caller hanging."
    )
