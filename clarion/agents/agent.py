"""High-level Agent — the public entry point Phase 8's API and Phase 5's
tests both go through.

Usage::

    agent = Agent.from_customer(
        customer=cfg,
        llm=OpenAIClient(),
        structured=store,
        retriever=retriever,
    )
    text = agent.chat("I'd like to book a cataract consult next Monday")
    print(text)

Conversation memory is just ``list[Message]`` kept on the instance. Each
``chat`` call:

1. Rebuilds the system prompt for the current turn (so retrieval is
   focused on the *new* user message)
2. Appends the user message
3. Runs the ReAct loop
4. Returns the final assistant text

The system prompt is rebuilt every turn — Phase 7's observability will
record the system prompt token count per turn from this list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from clarion.agents.llm import LLMClient, Message
from clarion.agents.prompt import PromptContext, build_system_prompt
from clarion.agents.react import DEFAULT_MAX_STEPS, ReactResult, react_loop
from clarion.config import CustomerConfig
from clarion.pipelines.structured import StructuredStore
from clarion.rag.builder import load_customer_retriever
from clarion.rag.retriever import Retriever
from clarion.tools.base import ToolContext

log = logging.getLogger(__name__)


@dataclass
class Agent:
    """One conversation. Holds the rolling transcript + the runtime context.

    Instantiate per call (each phone call is one Agent). Inside the call
    you can ``chat`` repeatedly to advance the conversation; the LLM sees
    the full transcript each turn.
    """

    customer: CustomerConfig
    llm: LLMClient
    ctx: ToolContext
    retriever: Retriever | None = None
    max_steps: int = DEFAULT_MAX_STEPS

    # Rolling transcript — excludes the system message (rebuilt per turn)
    # but includes every user / assistant / tool message so the LLM has
    # full context on each turn.
    transcript: list[Message] = field(default_factory=list)

    # ---------- public entry points ----------

    def chat(self, user_message: str) -> str:
        """Advance the conversation by one user turn; return the agent's reply."""
        prompt_ctx = PromptContext(customer=self.customer, retriever=self.retriever)
        system_prompt = build_system_prompt(prompt_ctx, user_message=user_message)

        # Each LLM call sees a freshly built system message followed by the
        # full prior transcript and the new user turn.
        self.transcript.append(Message.user(user_message))
        loop_messages: list[Message] = [Message.system(system_prompt), *self.transcript]

        result: ReactResult = react_loop(
            llm=self.llm,
            messages=loop_messages,
            customer=self.customer,
            ctx=self.ctx,
            max_steps=self.max_steps,
        )
        # Replay the loop-side mutations back onto our transcript,
        # skipping the system message we prepended.
        self.transcript = loop_messages[1:]
        log.debug(
            "agent chat: steps=%d, max_steps_hit=%s",
            len(result.steps),
            result.stopped_for_max_steps,
        )
        return result.final_text

    # ---------- convenience constructors ----------

    @classmethod
    def from_customer(
        cls,
        *,
        customer: CustomerConfig,
        llm: LLMClient,
        structured: StructuredStore,
        retriever: Retriever | None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> Agent:
        ctx = ToolContext(customer=customer, structured=structured)
        return cls(
            customer=customer,
            llm=llm,
            ctx=ctx,
            retriever=retriever,
            max_steps=max_steps,
        )

    @classmethod
    def build(
        cls,
        *,
        customer: CustomerConfig,
        llm: LLMClient,
        data_dir: Path,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> Agent:
        """Construct everything from the customer config + data dir.

        Loads the customer's structured store and prebuilt RAG index;
        raises if either is missing (run the ingest CLI first).
        """
        structured = StructuredStore.for_customer(customer.customer_id, data_dir)
        retriever = load_customer_retriever(customer, data_dir=data_dir)
        return cls.from_customer(
            customer=customer,
            llm=llm,
            structured=structured,
            retriever=retriever,
            max_steps=max_steps,
        )
