"""Emergency specialist — short-circuit handoff.

No LLM call, no tools. The router's emergency classification is the
gate; once we're here, the response is deterministic.
"""

from __future__ import annotations

from typing import ClassVar, cast

from clarion.agents.llm import Message
from clarion.multiagent.specialists.base import Specialist
from clarion.multiagent.state import MultiAgentState, SpecialistIntent

EMERGENCY_REPLY = (
    "This sounds like an emergency. Please call 911 immediately. "
    "If you're already in our clinic, please stay on the line and "
    "I'll connect you with a clinician right away."
)


class EmergencySpecialist(Specialist):
    """Short-circuit specialist — overrides __call__ entirely."""

    intent: ClassVar[SpecialistIntent] = "emergency"
    allowed_tools: ClassVar[frozenset[str]] = frozenset()
    persona: ClassVar[str] = ""

    def __call__(self, state: MultiAgentState) -> MultiAgentState:
        # No LLM, no tools, no ambiguity. Set the assistant text + the
        # escalation flag; the supervisor (commit 4) will see
        # escalated=True and FINISH.
        return cast(
            MultiAgentState,
            {
                "assistant_text": EMERGENCY_REPLY,
                "messages": [Message.assistant(text=EMERGENCY_REPLY)],
                "escalated": True,
                "escalation_reasons": ["emergency_intent_classified"],
            },
        )
