"""Supervisor node — completion check + escalation routing.

Runs AFTER each specialist. Three decisions on the table:

* **finish** — the specialist's reply is sufficient; return to the
  user. Default outcome.
* **route** — the conversation needs a different specialist (e.g.
  Info found a payer question hiding inside a hours request). Loop
  back to the router. Bounded by ``max_visits`` so we can't bounce
  forever.
* **escalate** — hand off to a human. Triggers:
    1. ``state["escalated"]`` already set by a specialist (Emergency)
    2. ``state["visits"] >= max_visits`` — the router loop bounced
       too many times
    3. The configured ``EscalationScorer`` flags the assistant
       turn (low-confidence, frustration, repeated refusals)

The supervisor is intentionally rule-based, not LLM-backed. The
decision is small (3 options), the cost matters (a single extra LLM
call per turn doubles latency), and the rules are auditable in a
way an LLM judgment isn't. The Sentinel ``EscalationScorer`` is the
shared escalation brain — the same heuristic the single-agent
backend uses, so escalation behavior stays consistent across
backends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

from clarion.multiagent.state import MultiAgentState, SupervisorDecision
from clarion.sentinel.escalation import (
    ConversationFacts,
    EscalationScorer,
)

log = logging.getLogger(__name__)

DEFAULT_MAX_VISITS = 3

# Friendly handoff text emitted on the escalate path. Mirrors the
# single-agent backend's handoff message so a caller migrating
# between backends doesn't notice a copy difference.
ESCALATION_HANDOFF_TEXT = (
    "Let me have a teammate call you back to finish this — they'll be "
    "in touch shortly. Is there anything else I can note for them?"
)


@dataclass
class Supervisor:
    """Decide what happens after a specialist runs.

    The Supervisor is constructed once per graph and reused for every
    turn — it carries the EscalationScorer + max_visits config but no
    per-turn state.
    """

    scorer: EscalationScorer | None = None
    max_visits: int = DEFAULT_MAX_VISITS
    handoff_text: str = ESCALATION_HANDOFF_TEXT

    # Optional list of escalation-reason strings the runner can read
    # after each turn (for tracing). The supervisor itself appends to
    # state["escalation_reasons"] which is the canonical signal.
    _last_reasons: list[str] = field(default_factory=list)

    def __call__(self, state: MultiAgentState) -> MultiAgentState:
        """LangGraph node: read the specialist's output, emit a decision."""
        visits = state.get("visits", 0) + 1
        assistant_text = state.get("assistant_text", "")
        already_escalated = state.get("escalated", False)

        # 1. Specialist (Emergency) already said "escalate". Honor it.
        if already_escalated:
            return self._escalate(
                state,
                visits=visits,
                reasons=state.get("escalation_reasons", []) or ["specialist_set_escalated"],
                # Specialist already wrote the assistant_text (e.g.
                # EMERGENCY_REPLY); don't overwrite it with handoff text.
                override_text=False,
            )

        # 2. Bounced too many times in the router loop.
        if visits > self.max_visits:
            return self._escalate(
                state,
                visits=visits,
                reasons=[f"router_loop_exceeded_max_visits={self.max_visits}"],
                override_text=True,
            )

        # 3. EscalationScorer flags the assistant turn.
        if self.scorer is not None and assistant_text:
            facts = self._facts_for(state)
            score = self.scorer.score(facts)
            if score.should_escalate:
                return self._escalate(
                    state,
                    visits=visits,
                    reasons=list(score.reasons),
                    override_text=True,
                )

        # 4. Default — specialist's reply is sufficient.
        finish: SupervisorDecision = "finish"
        return cast(
            MultiAgentState,
            {
                "decision": finish,
                "visits": visits,
            },
        )

    # ---------- helpers ----------

    def _escalate(
        self,
        state: MultiAgentState,
        *,
        visits: int,
        reasons: list[str],
        override_text: bool,
    ) -> MultiAgentState:
        escalate: SupervisorDecision = "escalate"
        delta: dict[str, object] = {
            "decision": escalate,
            "escalated": True,
            "escalation_reasons": reasons,
            "visits": visits,
        }
        if override_text:
            delta["assistant_text"] = self.handoff_text
            from clarion.agents.llm import Message  # local import; circular guard

            delta["messages"] = [Message.assistant(text=self.handoff_text)]
        log.info(
            "supervisor escalating",
            extra={"reasons": reasons, "visits": visits},
        )
        self._last_reasons = list(reasons)
        return cast(MultiAgentState, delta)

    def _facts_for(self, state: MultiAgentState) -> ConversationFacts:
        """Build the ConversationFacts the shared scorer consumes.

        The multi-agent backend doesn't track judge / refusal /
        confidence the same way scripted evaluation does, so we
        surface what the supervisor actually has visibility into:

        * ``user_messages``   — pulled from transcript role=user
        * ``agent_replies``   — transcript role=assistant + the new
          specialist text in ``assistant_text``
        * ``tools_called``    — names from assistant ``tool_calls``
          entries across the transcript
        * ``judge``           — None; the multi-agent runner doesn't
          run a per-turn judge
        * ``expected_outcome_is_task`` — False; the runner doesn't
          know the scenario's ground truth at serve time
        * ``already_escalated`` — read from state["escalated"]
        """
        from clarion.agents.llm import Message  # local import; circular guard

        transcript = state.get("messages", [])
        user_messages: list[str] = []
        agent_replies: list[str] = []
        tools_called: list[str] = []
        for msg in transcript:
            if not isinstance(msg, Message):
                continue
            if msg.role == "user" and msg.content:
                user_messages.append(msg.content)
            elif msg.role == "assistant":
                if msg.content:
                    agent_replies.append(msg.content)
                for tc in msg.tool_calls:
                    tools_called.append(tc.name)
        # Include the just-emitted text from the specialist in the
        # reply list (it may not have been pushed into state["messages"]
        # before the supervisor reads).
        text = state.get("assistant_text", "")
        if text and (not agent_replies or agent_replies[-1] != text):
            agent_replies.append(text)
        # Ensure user_messages includes the latest turn.
        user_msg = state.get("user_message", "")
        if user_msg and (not user_messages or user_messages[-1] != user_msg):
            user_messages.append(user_msg)
        return ConversationFacts(
            user_messages=user_messages,
            agent_replies=agent_replies,
            tools_called=tools_called,
            judge=None,
            expected_outcome_is_task=False,
            already_escalated=state.get("escalated", False),
        )


def route_after_supervisor(state: MultiAgentState) -> str:
    """Conditional-edge predicate the runner uses to wire the graph.

    Returns the literal LangGraph node name to dispatch to next:
    ``"END"`` for finish + escalate (both end the run), ``"router"``
    for the route-back-to-classifier loop.
    """
    decision = state.get("decision", "finish")
    if decision == "route":
        return "router"
    # finish + escalate both terminate the graph.
    return "END"
