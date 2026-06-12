"""Shared scaffolding for specialist nodes.

Every specialist:

* Is a ``(MultiAgentState) -> MultiAgentState`` callable so it
  drops into the LangGraph as a node.
* Runs the existing :func:`clarion.agents.react.react_loop` over a
  **restricted tool subset** (filter applied per specialist).
* Surfaces its assistant text on ``state["assistant_text"]`` —
  the supervisor (commit 4) decides whether that text is the
  final reply or whether we route to a different specialist.

The specialist Protocol is intentionally thin. Each implementation
sets two class attributes:

* ``intent`` — the SpecialistIntent it handles
* ``allowed_tools`` — frozenset of tool names it may call

Everything else (system-prompt building, ReAct invocation, state
update) lives in :class:`Specialist` so individual specialists are
~10 lines of declarative config plus an optional ``finalize`` hook
for short-circuit paths (emergency).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from clarion.agents.llm import LLMClient, Message
from clarion.agents.react import DEFAULT_MAX_STEPS, react_loop
from clarion.config import CustomerConfig
from clarion.multiagent.state import MultiAgentState, SpecialistIntent
from clarion.observability import Tracer
from clarion.tools.base import Tool, ToolContext
from clarion.tools.registry import available_tools

log = logging.getLogger(__name__)


@dataclass
class Specialist:
    """Base class for tool-using specialists.

    Subclasses set ``intent``, ``allowed_tools``, and ``persona`` at
    class scope; the runtime stitch (``__call__`` on the node) lives
    here so subclasses stay short.
    """

    # Class-level config — overridden per specialist.
    intent: ClassVar[SpecialistIntent] = "info"
    allowed_tools: ClassVar[frozenset[str]] = frozenset()
    persona: ClassVar[str] = ""

    # Instance dependencies — injected at runner-assembly time.
    llm: LLMClient
    customer: CustomerConfig
    ctx: ToolContext
    tracer: Tracer | None = None
    max_steps: int = DEFAULT_MAX_STEPS

    # ---------- node entry ----------

    def __call__(self, state: MultiAgentState) -> MultiAgentState:
        """LangGraph node entry: read state, run ReAct, return delta."""
        user_message = state.get("user_message", "")
        prior = state.get("messages", [])

        system = self._build_system_prompt(user_message=user_message)
        messages: list[Message] = [
            Message.system(system),
            *prior,
            Message.user(user_message),
        ]

        # Bind a restricted CustomerConfig so the ReAct loop's tool
        # registry returns only this specialist's allowed tools.
        scoped = self._scoped_customer()
        result = react_loop(
            llm=self.llm,
            messages=messages,
            customer=scoped,
            ctx=self.ctx,
            max_steps=self.max_steps,
            tracer=self.tracer,
        )

        # Hand the supervisor a single source of truth.
        return cast(
            MultiAgentState,
            {
                "assistant_text": result.final_text,
                # Append ONLY the new assistant turn — the existing
                # transcript is in state["messages"] already, and
                # operator.add will concatenate.
                "messages": [Message.assistant(text=result.final_text)],
            },
        )

    # ---------- overridable hooks ----------

    def _build_system_prompt(self, *, user_message: str) -> str:
        """Compose the system prompt for this specialist's ReAct loop.

        Default: persona + tool-scoping line + standard safety footer.
        Subclasses override for special handling (emergency).
        """
        tool_line = (
            f"You may only call these tools: {sorted(self.allowed_tools)}. "
            "Refuse to attempt anything outside that set; instead, ask the "
            "user a clarifying question or explain the limitation."
        )
        return (
            f"{self.persona}\n\n"
            f"{tool_line}\n\n"
            "Be concise. Never provide clinical advice. If the user "
            "describes an emergency, stop and respond with: 'This sounds "
            "like an emergency. Please call 911 immediately.'"
        )

    # ---------- internals ----------

    def _scoped_customer(self) -> CustomerConfig:
        """Return a CustomerConfig copy with enabled_tools intersected.

        We don't mutate the caller's customer — both backends share it,
        and a downstream node could be running with the unrestricted
        set. ``model_copy(update=...)`` is the Pydantic v2 idiom.
        """
        scoped_tools = [t for t in self.customer.enabled_tools if t in self.allowed_tools]
        return self.customer.model_copy(update={"enabled_tools": scoped_tools})


def filter_tools(
    customer: CustomerConfig, allowed: frozenset[str]
) -> list[Tool[Any, Any]]:
    """Helper: customer's enabled tools intersected with ``allowed``.

    Used by tests + the runner when it wants to advertise the tool list
    a specialist will actually see (e.g. on a /capabilities endpoint).
    """
    return [t for t in available_tools(customer) if t.name in allowed]
