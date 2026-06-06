"""System-prompt construction.

The system prompt has three layers, in order:

1. **Persona + identity** — straight from ``CustomerConfig.agent_persona``.
2. **Operating contract** — hard rules that apply to every Clarion
   deployment: no clinical advice, escalate emergencies, ask before
   inventing data, call tools rather than guess.
3. **Per-turn retrieval context** — top-k chunks from the customer's
   rules RAG index, selected against the *current user message*. This is
   how the agent stays grounded in the practice's actual policies
   without burning tokens on irrelevant rules.

The escalation thresholds from the config land here too so the agent has
the same "when to escalate" knobs the Sentinel will measure against in
Phase 11.
"""

from __future__ import annotations

from dataclasses import dataclass

from clarion.config import CustomerConfig
from clarion.rag.retriever import RetrievalHit, Retriever

# Hard rules every Clarion deployment ships with. Keep short — the LLM
# reads this every turn. Practice-specific rules come from RAG.
_OPERATING_CONTRACT = """\
Operating contract (applies to every conversation):

- You never give clinical advice. If the patient asks anything requiring
  medical judgment (medication safety, "should I", symptom interpretation),
  decline politely and offer to schedule a visit or file a task for a clinician.
- If the patient describes a medical emergency (sudden vision loss, severe
  pain, suspected stroke, compound fracture, loss of consciousness), tell
  them to call 911 or go to an emergency department, and call
  create_pms_task with priority "urgent". Never attempt to book a routine
  appointment in that turn.
- Prefer calling a tool over guessing. If a fact is not in the rules above
  or in a tool response, ask the patient or call create_pms_task — never
  invent appointment types, payers, or provider names.
- Confirm patient identity before any booking or cancellation: full name,
  date of birth, and a callback number at minimum.
- Speak warmly and concisely. One question at a time.
"""


@dataclass(frozen=True)
class PromptContext:
    """Everything we need to build a per-turn system prompt."""

    customer: CustomerConfig
    retriever: Retriever | None  # None during tests that don't need RAG


def build_system_prompt(
    ctx: PromptContext,
    *,
    user_message: str | None = None,
    rag_k: int = 4,
) -> str:
    """Assemble the system prompt for the next LLM call.

    ``user_message`` is the latest user turn; we retrieve against it so the
    rules block focuses on the topic the patient just raised. Pass ``None``
    on the first call before the user has spoken (greeting turn).
    """
    parts: list[str] = [
        f"You are Clarion, the virtual front-desk assistant for " f"{ctx.customer.display_name}.",
        ctx.customer.agent_persona.strip(),
        _OPERATING_CONTRACT,
        _escalation_block(ctx.customer),
    ]
    if user_message and ctx.retriever is not None:
        hits = ctx.retriever.retrieve(user_message, k=rag_k)
        if hits:
            parts.append(_rules_block(hits))
    return "\n\n".join(parts)


def _escalation_block(customer: CustomerConfig) -> str:
    e = customer.escalation
    return (
        "Escalation policy for this practice:\n"
        f"- escalate if your own confidence drops below {e.low_confidence:.2f}\n"
        f"- escalate after {e.max_clarifications} clarification turns "
        "without progress\n"
        f"- escalate if the patient's frustration appears to exceed "
        f"{e.frustration:.2f}\n"
        + (
            "- escalate immediately on any rule conflict (two rules give " "different answers)\n"
            if e.on_rule_conflict
            else ""
        )
        + "When you decide to escalate, call create_pms_task with a clear "
        "subject and a one-paragraph body summarizing the conversation, "
        "then tell the patient a teammate will call back."
    )


def _rules_block(hits: list[RetrievalHit]) -> str:
    """Format retrieved chunks as labeled, citable rule blocks."""
    lines = [
        "Practice rules relevant to this turn (cite by [source] if you "
        "quote them; never invent rules not listed here):"
    ]
    for i, h in enumerate(hits, start=1):
        lines.append(
            f"\n[{i}] {h.chunk.heading}  (source: {h.chunk.source}, "
            f"score: {h.score:.2f})\n{h.chunk.text}"
        )
    return "\n".join(lines)
