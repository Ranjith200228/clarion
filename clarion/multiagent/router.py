"""Intent router for the LangGraph multi-agent backend.

One node, one job: read the latest user message + transcript and
emit ``state["intent"] = "<SpecialistIntent>"``. The conditional
edge in :mod:`clarion.multiagent.runner` consumes that field and
dispatches to the matching specialist.

Why a separate router instead of letting the supervisor decide:

* Classification is a small, focused task; gpt-4o-mini handles it
  reliably with a tight prompt + structured output.
* Keeping it separate means we can swap in a cheaper / faster
  classifier (DistilBERT, a small open-weights model) later
  without touching specialist code.
* The router is stateless across turns — it sees the rolling
  transcript but doesn't carry routing decisions forward. That's
  what the visit counter on state is for (the supervisor flips
  the loop).

The router is LLM-backed for the production path and rule-based
for tests / sub-second smoke runs. Both implement the same
``Router`` Protocol so the graph wiring doesn't care which one is
plugged in.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol, cast, get_args

from clarion.agents.llm import LLMClient, Message, ToolSpec
from clarion.multiagent.state import MultiAgentState, SpecialistIntent

log = logging.getLogger(__name__)


# A literal-equivalent tuple so we can iterate / check membership
# at runtime without re-typing the strings.
_INTENT_VALUES: tuple[str, ...] = get_args(SpecialistIntent)


_ROUTER_TOOL_SPEC = ToolSpec(
    name="route",
    description=(
        "Classify the user's latest message into exactly one specialist "
        "queue: booking (book / reschedule a slot), eligibility "
        "(insurance / payer / coverage), info (rules-grounded factual "
        "question — hours, policies), cancel (cancel an existing "
        "appointment), or emergency (the user describes an acute "
        "medical situation). Pick exactly one."
    ),
    parameters={
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": list(_INTENT_VALUES),
                "description": "The single best-fit specialist queue.",
            },
            "rationale": {
                "type": "string",
                "description": "One sentence justifying the choice. For audit only.",
            },
        },
        "required": ["intent"],
        "additionalProperties": False,
    },
)


_ROUTER_SYSTEM = (
    "You are the intent router for a healthcare scheduling assistant. "
    "Read the user's latest message and call the `route` function exactly "
    "once with the best-fit specialist queue. Do NOT answer the user; "
    "another agent will handle the conversation. If the user describes "
    "any medical emergency (chest pain, stroke symptoms, severe bleeding, "
    "vision loss, suspected fracture, anything time-critical) you MUST "
    "pick `emergency` — do not second-guess."
)


class Router(Protocol):
    """Read the state, emit ``intent``. Pure (State) -> State."""

    def __call__(self, state: MultiAgentState) -> MultiAgentState: ...


@dataclass
class LLMIntentRouter:
    """Production router — tool-call structured output via gpt-4o-mini.

    Falls back to ``info`` on a malformed response. We deliberately do
    NOT fall back to ``emergency`` on parse error — that would be a
    safety-theater move (false-positive escalations from a flaky
    parser don't make anyone safer; they just train ops to ignore the
    signal).
    """

    llm: LLMClient

    def __call__(self, state: MultiAgentState) -> MultiAgentState:
        user_message = state.get("user_message", "")
        transcript = state.get("messages", [])

        prompt: list[Message] = [
            Message.system(_ROUTER_SYSTEM),
            *transcript[-6:],  # the recent few turns are enough context
            Message.user(user_message),
        ]
        resp = self.llm.complete(prompt, tools=[_ROUTER_TOOL_SPEC])

        intent = _extract_intent(resp.tool_calls)
        if intent is None:
            log.warning(
                "router: LLM returned no usable tool call; defaulting to info",
                extra={"router_content": resp.content},
            )
            intent = "info"
        return cast(MultiAgentState, {"intent": intent})


@dataclass
class HeuristicIntentRouter:
    """Deterministic router for tests + offline smoke runs.

    Word-anchored substring scan over a small keyword catalog. The
    ordering matters — emergency is checked first so a phrase like
    "I had a stroke yesterday, want to reschedule" routes to
    emergency, not booking. ``info`` is the fallback when nothing
    else matches.
    """

    _PATTERNS: tuple[tuple[SpecialistIntent, tuple[str, ...]], ...] = (
        (
            "emergency",
            (
                r"\bstroke\b",
                r"\bheart attack\b",
                r"\bchest pain\b",
                r"\bbleeding\b",
                r"\bvision loss\b",
                r"\bcan(?:'|no)?t see\b",
                r"\b911\b",
                r"\bemergency\b",
            ),
        ),
        (
            "cancel",
            (
                r"\bcancel(?:l(?:ing|ed))?\b",
                r"\bdrop my appointment\b",
            ),
        ),
        (
            "booking",
            (
                r"\bbook\b",
                r"\bschedule\b",
                r"\breschedul\w*\b",
                r"\bappointment\b",
                r"\bavailab\w+\b",
                r"\bslot\b",
                r"\b(?:next|this)\s+(?:week|month|tuesday|wednesday|thursday|friday)\b",
            ),
        ),
        (
            "eligibility",
            (
                r"\binsurance\b",
                r"\bcoverage\b",
                r"\beligib\w+\b",
                r"\bcovered\b",
                r"\bcopay\b",
                r"\baetna\b",
                r"\bcigna\b",
                r"\bbcbs\b",
                r"\bmedicare\b",
            ),
        ),
    )

    def __call__(self, state: MultiAgentState) -> MultiAgentState:
        text = state.get("user_message", "").lower()
        for intent, patterns in self._PATTERNS:
            if any(re.search(p, text) for p in patterns):
                return cast(MultiAgentState, {"intent": intent})
        # Default — rules / hours / general "what is" questions.
        return cast(MultiAgentState, {"intent": "info"})


def _extract_intent(tool_calls: tuple) -> SpecialistIntent | None:  # type: ignore[type-arg]
    """Pull ``intent`` from the LLM's ``route(...)`` tool call.

    Tolerant: accepts the first call to ``route`` with a known
    ``intent`` arg, regardless of how many tool calls came back.
    """
    for tc in tool_calls:
        if tc.name != "route":
            continue
        candidate = tc.arguments.get("intent")
        if isinstance(candidate, str) and candidate in _INTENT_VALUES:
            return cast(SpecialistIntent, candidate)
    return None
