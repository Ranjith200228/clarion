"""Shared state for the LangGraph multi-agent backend.

LangGraph nodes are pure ``(State) -> State`` callables — adding a
node never modifies the schema. This module owns the schema; every
node imports ``MultiAgentState`` from here.

Why a TypedDict + manual merge instead of a Pydantic model:

* LangGraph's reducer convention (``Annotated[list, add]`` for
  append-only fields) is built on plain dicts.
* Pydantic copies on every mutation; in a multi-node graph that's
  one copy per edge. TypedDicts are flat and cheap.
* The schema is small enough that we don't need runtime validation
  at every hop — the wire boundary stays Pydantic
  (``Message`` / ``ConversationSummary`` / etc.); state is
  in-process scratch.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from clarion.agents.llm import Message
from clarion.rag.retriever import RetrievalHit

# The router maps a user message to exactly one specialist. The
# Literal makes mypy + the dashboard categorize cleanly, and the
# enumerated set is small enough that a 3B model can pick reliably.
SpecialistIntent = Literal[
    "booking",       # search slots + book / reschedule one
    "eligibility",   # payer / insurance / coverage question
    "info",          # rules-grounded question (hours, policies)
    "emergency",     # safety short-circuit — emit canned reply
    "cancel",        # cancel an existing appointment
]


# ``decision`` from the supervisor — what to do after the specialist
# returns. FINISH ends the run; ROUTE sends back to the router for a
# different specialist; ESCALATE marks the conversation for handoff.
SupervisorDecision = Literal["finish", "route", "escalate"]


class MultiAgentState(TypedDict, total=False):
    """In-flight state for one user turn.

    Fields marked ``Annotated[..., operator.add]`` are LangGraph
    reducers — concurrent updates from parallel nodes merge by
    concatenation. We don't run parallel nodes yet, but the
    reducers cost nothing and future-proof the schema.

    All fields are optional (``total=False``) because LangGraph
    initializes the state from the runner's input dict and
    progressively fills in fields as nodes execute.
    """

    # Conversation transcript — append-only. The router and every
    # specialist see the full transcript; nodes only ever append
    # their assistant message (plus tool messages from a specialist's
    # ReAct loop).
    messages: Annotated[list[Message], operator.add]

    # Latest user text, redacted for PHI. Set once at graph entry by
    # the runner so every downstream node reads from one source of
    # truth.
    user_message: str

    # Router output. Set by ``router`` once per turn; consumed by
    # the conditional edge that picks the specialist.
    intent: SpecialistIntent

    # Retrieval — populated lazily by specialists that need it
    # (Info almost always; Booking sometimes for "what's a pre-op?").
    retrieved_chunks: Annotated[list[RetrievalHit], operator.add]

    # Supervisor decision after the specialist returns.
    decision: SupervisorDecision

    # The assistant text that will be returned to the user. The
    # last node to set this wins; the supervisor is responsible for
    # the final value.
    assistant_text: str

    # Escalation signal — set when the supervisor or the
    # EmergencySpecialist decides to hand off.
    escalated: bool
    escalation_reasons: Annotated[list[str], operator.add]

    # Step counter — guards against ROUTE loops. The supervisor
    # bumps it on every visit; ``max_steps`` in the runner trips
    # ESCALATE when exceeded.
    visits: int


def initial_state(*, user_message: str, transcript: list[Message]) -> MultiAgentState:
    """Build the entry-state for one user turn.

    The runner calls this once per ``chat()`` invocation. The
    transcript is the rolling conversation history (excluding the
    new user message, which the runner adds via ``user_message``
    so nodes can read either side without ambiguity).
    """
    return MultiAgentState(
        messages=list(transcript),
        user_message=user_message,
        retrieved_chunks=[],
        escalated=False,
        escalation_reasons=[],
        visits=0,
        assistant_text="",
    )
