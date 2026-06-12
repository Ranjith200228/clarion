"""Info specialist — rules-grounded factual questions.

Reads-only specialist: no booking tools, may create a follow-up task
when the rules corpus doesn't carry the answer.
"""

from __future__ import annotations

from typing import ClassVar

from clarion.multiagent.specialists.base import Specialist
from clarion.multiagent.state import SpecialistIntent


class InfoSpecialist(Specialist):
    """Tool subset: create_pms_task only (for unanswerable questions)."""

    intent: ClassVar[SpecialistIntent] = "info"
    allowed_tools: ClassVar[frozenset[str]] = frozenset({"create_pms_task"})
    persona: ClassVar[str] = (
        "You are the info specialist. Answer questions grounded in the "
        "practice's rules and policies. If the rules corpus doesn't cover "
        "the answer, say so directly and offer to create a follow-up task "
        "instead of guessing. Never invent hours, prices, or policy "
        "details. Never give clinical advice — even a casual 'should I "
        "be worried?' is out of scope."
    )
