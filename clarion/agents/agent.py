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
from typing import Any

from clarion.agents.llm import LLMClient, Message
from clarion.agents.prompt import PromptContext, build_system_prompt
from clarion.agents.react import DEFAULT_MAX_STEPS, ReactResult, react_loop
from clarion.config import CustomerConfig
from clarion.observability import Tracer, TraceWriter
from clarion.pipelines.structured import StructuredStore
from clarion.rag.builder import load_customer_retriever
from clarion.rag.retriever import Retriever
from clarion.sentinel import (
    AuditLog,
    AuditTurn,
    detect_clinical_advice_request,
    detect_emergency,
)
from clarion.tools.base import ToolContext

# Canned guardrail responses. Short, neutral, customer-friendly.
_EMERGENCY_REPLY = (
    "This sounds like an emergency. Please call 911 or go to the nearest "
    "emergency department right now. I've also flagged this for our care team "
    "so they can follow up."
)
_CLINICAL_ADVICE_REPLY = (
    "That's a clinical question I'm not able to answer over the phone. I'll "
    "take a message for one of our clinicians to call you back."
)

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

    # Optional audit log. When set, every chat turn (including guardrail
    # short-circuits) is appended with PHI redaction.
    audit: AuditLog | None = None

    # Optional trace writer. When set, every chat turn emits a JSONL
    # trace with full span hierarchy (agent.chat → retrieval → react.step
    # → llm.complete + tool.<name>).
    traces: TraceWriter | None = None

    # Guardrails are on by default. The agent NEVER lets a flagged user
    # message reach the LLM — short-circuits with a canned reply and an
    # urgent task for emergencies. Set to False only in tests that need
    # to validate the ReAct loop in isolation.
    guardrails_enabled: bool = True

    # Rolling transcript — excludes the system message (rebuilt per turn)
    # but includes every user / assistant / tool message so the LLM has
    # full context on each turn.
    transcript: list[Message] = field(default_factory=list)

    # Trace id of the most recent chat() turn. Set every turn; populated
    # whether or not a TraceWriter is attached, so the API layer can
    # echo it back to the client without re-reading any file.
    last_trace_id: str = ""

    # Last completed turn's spans, kept in memory for one cycle so the
    # API layer can build LastTurnMetrics without a file read. List of
    # the Tracer's spans after the with-block closes; overwritten on
    # every chat() call. None before the first call.
    _last_turn_spans: list[Any] | None = None

    # ---------- public entry points ----------

    def chat(self, user_message: str) -> str:
        """Advance the conversation by one user turn; return the agent's reply."""
        tracer = Tracer(
            customer_id=self.customer.customer_id,
            conversation_id=self.audit.conversation_id if self.audit else None,
        )
        self.last_trace_id = tracer.trace_id
        with tracer.span("agent.chat", user_chars=len(user_message)) as root:
            reply = self._chat_inner(user_message, tracer)
            root.set("reply_chars", len(reply))
        if self.traces is not None:
            self.traces.write(tracer.emit())
        # Capture the just-closed span list. ``Tracer.emit().spans`` is
        # already a snapshot (list of Span objects), safe to retain.
        self._last_turn_spans = list(tracer.emit().spans)
        return reply

    def _chat_inner(self, user_message: str, tracer: Tracer) -> str:
        # 1. Guardrails run BEFORE the LLM sees the message. They are
        #    cheap regex matches, so the cost is negligible and the
        #    safety story is unambiguous: an emergency phrase never
        #    reaches the model.
        if self.guardrails_enabled:
            with tracer.span("guardrails.check") as g_span:
                short_circuit = self._check_guardrails(user_message)
                g_span.set("fired", short_circuit is not None)
            if short_circuit is not None:
                return short_circuit

        # 2. Retrieval span (even when there's no retriever, so the
        #    dashboard can spot configs that skipped RAG).
        with tracer.span("retrieval") as r_span:
            prompt_ctx = PromptContext(customer=self.customer, retriever=self.retriever)
            hits = self.retriever.retrieve(user_message, k=4) if self.retriever is not None else []
            r_span.update(
                {
                    "k": 4,
                    "hit_count": len(hits),
                    "top_score": hits[0].score if hits else None,
                    "top_source": hits[0].chunk.source if hits else None,
                }
            )

        # 3. System prompt + ReAct loop.
        system_prompt = build_system_prompt(prompt_ctx, user_message=user_message)
        self.transcript.append(Message.user(user_message))
        loop_messages: list[Message] = [Message.system(system_prompt), *self.transcript]

        result: ReactResult = react_loop(
            llm=self.llm,
            messages=loop_messages,
            customer=self.customer,
            ctx=self.ctx,
            max_steps=self.max_steps,
            tracer=tracer,
        )
        self.transcript = loop_messages[1:]
        log.debug(
            "agent chat: steps=%d, max_steps_hit=%s",
            len(result.steps),
            result.stopped_for_max_steps,
        )

        # 4. Audit-log this turn (PHI-redacted inside AuditLog.write).
        if self.audit is not None:
            tool_calls = self._summarize_tool_calls(result)
            self.audit.write(
                AuditTurn(
                    user_message=user_message,
                    agent_reply=result.final_text,
                    guardrail="safe",
                    tool_calls=tool_calls,
                    steps=len(result.steps),
                    extra={
                        "stopped_for_max_steps": result.stopped_for_max_steps,
                        "trace_id": tracer.trace_id,
                    },
                )
            )
        return result.final_text

    # ---------- guardrail short-circuits ----------

    def _check_guardrails(self, user_message: str) -> str | None:
        """If a guardrail fires, return the canned reply and log it.
        Otherwise return None so chat() falls through to the LLM."""
        emergency = detect_emergency(user_message)
        if emergency.fired:
            self._handle_emergency(user_message)
            return _EMERGENCY_REPLY
        clinical = detect_clinical_advice_request(user_message)
        if clinical.fired:
            self._handle_clinical_advice(user_message)
            return _CLINICAL_ADVICE_REPLY
        return None

    def _handle_emergency(self, user_message: str) -> None:
        """File an urgent PMS task and append to the transcript + audit."""
        # File the urgent task. Tools never raise to the agent, so this
        # is best-effort: failure to file is logged and ignored.
        task_id: str | None = None
        try:
            from clarion.schemas.tools import CreatePmsTaskInput
            from clarion.tools.create_pms_task import CreatePmsTaskTool

            out = CreatePmsTaskTool().run(
                CreatePmsTaskInput(
                    subject="EMERGENCY caller — escalate immediately",
                    body=(
                        "Caller described a possible emergency. The agent "
                        "advised 911 / ED and did not attempt to book. "
                        "Verbatim message (PHI-redacted in audit log)."
                    ),
                    priority="urgent",
                ),
                self.ctx,
            )
            task_id = out.task_id
        except Exception:
            log.exception("emergency task creation failed")

        self.transcript.append(Message.user(user_message))
        self.transcript.append(Message.assistant(text=_EMERGENCY_REPLY))
        if self.audit is not None:
            self.audit.write(
                AuditTurn(
                    user_message=user_message,
                    agent_reply=_EMERGENCY_REPLY,
                    guardrail="emergency",
                    extra={"escalation_task_id": task_id},
                )
            )

    def _handle_clinical_advice(self, user_message: str) -> None:
        self.transcript.append(Message.user(user_message))
        self.transcript.append(Message.assistant(text=_CLINICAL_ADVICE_REPLY))
        if self.audit is not None:
            self.audit.write(
                AuditTurn(
                    user_message=user_message,
                    agent_reply=_CLINICAL_ADVICE_REPLY,
                    guardrail="clinical_advice",
                )
            )

    @staticmethod
    def _summarize_tool_calls(result: ReactResult) -> list[dict[str, object]]:
        """Flatten react steps into a list of tool-call dicts the audit
        log can persist."""
        out: list[dict[str, object]] = []
        for step in result.steps:
            for call, reply in zip(step.tool_calls, step.tool_results, strict=False):
                out.append(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "ok": reply.get("ok"),
                        "error": reply.get("error"),
                    }
                )
        return out

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
