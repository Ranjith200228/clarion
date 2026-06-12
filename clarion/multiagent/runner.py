"""Assemble the StateGraph and expose an Agent-compatible runner.

The graph:

::

    START -> router -> {booking | eligibility | info | cancel | emergency}
                                                |
                                                v
                                            supervisor
                                                |
                       finish / escalate ------+------> END
                              route ------------+----> back to router

Each specialist is a node; the router emits ``state["intent"]``
which a conditional edge maps to the matching specialist; the
supervisor emits ``state["decision"]`` which a second conditional
edge maps to END (finish / escalate) or back to ``router`` (route).

The runner exposes :class:`MultiAgentRunner` with a ``chat(message)``
signature matching :class:`clarion.agents.Agent.chat`, so the
session manager can swap in either backend without per-call
branches in the route handlers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from clarion.agents.llm import LLMClient, Message
from clarion.config import CustomerConfig
from clarion.multiagent.router import (
    HeuristicIntentRouter,
    LLMIntentRouter,
    Router,
)
from clarion.multiagent.specialists import (
    BookingSpecialist,
    CancelSpecialist,
    EligibilitySpecialist,
    EmergencySpecialist,
    InfoSpecialist,
    Specialist,
)
from clarion.multiagent.state import (
    MultiAgentState,
    SpecialistIntent,
    initial_state,
)
from clarion.multiagent.supervisor import (
    DEFAULT_MAX_VISITS,
    Supervisor,
    route_after_supervisor,
)
from clarion.observability import Tracer
from clarion.sentinel.escalation import EscalationScorer
from clarion.tools.base import ToolContext

log = logging.getLogger(__name__)


@dataclass
class MultiAgentRunner:
    """Agent-compatible runner backed by a LangGraph StateGraph.

    Construct one per conversation (same lifecycle as
    :class:`clarion.agents.Agent`). The compiled graph + Sentinel
    scorer + tracer are reusable across turns; the rolling
    transcript lives on the runner instance.
    """

    customer: CustomerConfig
    llm: LLMClient
    ctx: ToolContext
    use_heuristic_router: bool = False
    """When True, swap in the deterministic HeuristicIntentRouter
    instead of the LLM-backed one. Tests + offline smoke runs use
    this so they don't burn LLM calls on classification."""

    escalation_scorer: EscalationScorer | None = None
    tracer: Tracer | None = None
    max_visits: int = DEFAULT_MAX_VISITS

    # Rolling transcript — mirrors clarion.agents.Agent's contract
    # so a session can swap backends mid-conversation without losing
    # state.
    transcript: list[Message] = field(default_factory=list)

    last_trace_id: str = ""

    def __post_init__(self) -> None:
        self._scorer = self.escalation_scorer or EscalationScorer()
        self._graph = self._build_graph()

    # ---------- public entry point ----------

    def chat(self, user_message: str) -> str:
        """Advance one user turn through the multi-agent graph.

        Signature matches :class:`clarion.agents.Agent.chat`. Returns
        the assistant's reply text; appends the new user + assistant
        turns to ``self.transcript``.
        """
        tracer = self.tracer
        if tracer is not None:
            self.last_trace_id = tracer.trace_id
            with tracer.span(
                "multiagent.chat",
                user_chars=len(user_message),
            ) as span:
                reply = self._run(user_message)
                span.set("reply_chars", len(reply))
                return reply
        return self._run(user_message)

    # ---------- internals ----------

    def _run(self, user_message: str) -> str:
        state = initial_state(
            user_message=user_message,
            transcript=list(self.transcript),
        )
        result = cast(MultiAgentState, self._graph.invoke(state))
        reply = result.get("assistant_text", "")

        # Append both the user turn and the final assistant turn to
        # the rolling transcript. The intermediate Specialist message
        # already lives in result["messages"] but it would double-up
        # with the supervisor's override on escalate, so we trust
        # ``assistant_text`` as the authoritative final reply.
        self.transcript.append(Message.user(user_message))
        self.transcript.append(Message.assistant(text=reply))
        return reply

    def _build_graph(self) -> Any:
        """Wire the StateGraph once and compile it.

        LangGraph's compile() returns a Pregel-style runnable with
        an ``.invoke(state)`` method we use in ``_run``.
        """
        router = self._router_instance()
        supervisor = Supervisor(
            scorer=self._scorer,
            max_visits=self.max_visits,
        )
        specialists = self._build_specialists()

        graph = StateGraph(MultiAgentState)
        graph.add_node("router", router)
        for intent, node in specialists.items():
            graph.add_node(intent, node)
        graph.add_node("supervisor", supervisor)

        graph.add_edge(START, "router")
        # Conditional dispatch from router -> specialist by intent.
        graph.add_conditional_edges(
            "router",
            _route_to_specialist,
            {intent: intent for intent in specialists},
        )
        # Every specialist hands off to the supervisor.
        for intent in specialists:
            graph.add_edge(intent, "supervisor")
        # Supervisor either finishes the turn or routes back.
        graph.add_conditional_edges(
            "supervisor",
            route_after_supervisor,
            {"router": "router", "END": END},
        )
        return graph.compile()

    def _router_instance(self) -> Router:
        if self.use_heuristic_router:
            return HeuristicIntentRouter()
        return LLMIntentRouter(llm=self.llm)

    def _build_specialists(self) -> dict[SpecialistIntent, Specialist]:
        common = {
            "llm": self.llm,
            "customer": self.customer,
            "ctx": self.ctx,
            "tracer": self.tracer,
        }
        return {
            "booking": BookingSpecialist(**common),  # type: ignore[arg-type]
            "eligibility": EligibilitySpecialist(**common),  # type: ignore[arg-type]
            "info": InfoSpecialist(**common),  # type: ignore[arg-type]
            "cancel": CancelSpecialist(**common),  # type: ignore[arg-type]
            "emergency": EmergencySpecialist(**common),  # type: ignore[arg-type]
        }


def _route_to_specialist(state: MultiAgentState) -> str:
    """Conditional-edge predicate from router -> specialist."""
    intent = state.get("intent", "info")
    return intent
