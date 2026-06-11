"""PMS-writeback extractors.

Two implementations sharing the same Protocol so callers don't know
(or care) which one is plugged in:

* ``HeuristicExtractor``  — regex / keyword extraction from transcript +
  scenario ground truth. Deterministic, no LLM key needed, used in CI
  and as the default in the harness.
* (future) ``LLMExtractor`` — JSON-mode LLM call. Sharper but costs money.

Both produce a ``ConversationSummary``. The writer in commit 3 builds
the matching ``PmsTaskWriteback`` from the same context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from clarion.schemas import (
    ConversationSummary,
    HarnessResult,
    Scenario,
    SummaryOutcome,
)

# Common intent keywords -> normalized labels.
_INTENT_KEYWORDS: dict[str, str] = {
    "book": "booking",
    "schedule": "booking",
    "appointment": "booking",
    "cancel": "cancellation",
    "reschedule": "reschedule",
    "eligibility": "eligibility_check",
    "insurance": "eligibility_check",
    "coverage": "eligibility_check",
    "refill": "refill_request",
    "hours": "faq",
    "location": "faq",
    "directions": "faq",
}

# Outcome mapping from harness's Outcome enum to PMS-writeback enum.
_OUTCOME_TO_SUMMARY: dict[str, SummaryOutcome] = {
    "booked": "booked",
    "cancelled": "cancelled",
    "task_created": "task_created",
    "escalated_emergency": "escalated_emergency",
    "refused_clinical": "refused_clinical",
    "info_provided": "info_provided",
}

_PHONE_RE = re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
_PAT_ID_RE = re.compile(r"\bpat[_-]?\d+\b", re.IGNORECASE)
_PAYER_KEYWORDS = (
    "aetna",
    "anthem",
    "blue cross",
    "blue shield",
    "cigna",
    "kaiser",
    "medicaid",
    "medicare",
    "united",
    "uhc",
    "workers comp",
)


@dataclass(frozen=True)
class ExtractionContext:
    """Everything an extractor needs to produce a ConversationSummary.

    Built once per scenario by the harness; passed to both the extractor
    (for ConversationSummary) and the writer (for PmsTaskWriteback).
    """

    customer_id: str
    conversation_id: str
    scenario: Scenario
    result: HarnessResult


class Extractor(Protocol):
    """Anything that turns an ExtractionContext into a summary."""

    def extract(self, ctx: ExtractionContext) -> ConversationSummary: ...


class HeuristicExtractor:
    """Regex + keyword extractor — no LLM call.

    Uses the scenario's ground truth as a strong prior (a real
    deployment would not have ground truth; the heuristic extractor is
    paired with the deterministic harness so we always have it). For
    fields the ground truth does not cover (caller_name, payer), falls
    back to regex over the user messages.
    """

    def extract(self, ctx: ExtractionContext) -> ConversationSummary:
        scenario = ctx.scenario
        result = ctx.result
        gt = scenario.ground_truth

        user_text = "\n".join(scenario.messages)

        patient_id = self._extract_patient_id(user_text)
        payer = self._extract_payer(user_text)
        caller_name = self._extract_caller_name(scenario.messages)
        intent = self._extract_intent(scenario.intent, user_text)

        outcome = _OUTCOME_TO_SUMMARY.get(result.actual_outcome, "unresolved")
        escalated = bool(result.escalated)

        # Build short transcript preview from the last user message +
        # the last agent reply.
        preview_parts: list[str] = []
        if scenario.messages:
            preview_parts.append(f"user: {scenario.messages[-1][:200]}")
        if result.agent_replies:
            preview_parts.append(f"agent: {result.agent_replies[-1][:200]}")
        transcript_preview = "\n".join(preview_parts)[:500]

        notes = self._build_notes(scenario, result)

        return ConversationSummary(
            customer_id=ctx.customer_id,
            conversation_id=ctx.conversation_id,
            generated_at=datetime.now(UTC),
            patient_id=patient_id,
            caller_name=caller_name,
            intent=intent,
            appointment_type=gt.expected_appointment_type,
            appointment_time=None,  # heuristic doesn't reliably parse times
            payer=payer,
            outcome=outcome,
            escalated=escalated,
            notes=notes,
            transcript_preview=transcript_preview,
        )

    # ---------- field extractors ----------

    def _extract_patient_id(self, text: str) -> str | None:
        m = _PAT_ID_RE.search(text)
        return m.group(0).lower() if m else None

    def _extract_payer(self, text: str) -> str | None:
        lower = text.lower()
        for keyword in _PAYER_KEYWORDS:
            if keyword in lower:
                # Title-case for downstream consumption.
                return keyword.title()
        return None

    def _extract_caller_name(self, messages: list[str]) -> str | None:
        """Match patterns like 'Hi, this is John Doe' or 'I'm Jane Smith'."""
        for msg in messages:
            m = re.search(
                r"(?:this\s+is|I'?m|my\s+name\s+is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
                msg,
            )
            if m:
                return m.group(1).strip()
        return None

    def _extract_intent(self, scenario_intent: str, text: str) -> str:
        """Prefer the scenario's labeled intent; fall back to keyword scan."""
        if scenario_intent and scenario_intent != "faq":
            return scenario_intent
        lower = text.lower()
        for keyword, label in _INTENT_KEYWORDS.items():
            if keyword in lower:
                return label
        return scenario_intent or "unknown"

    def _build_notes(self, scenario: Scenario, result: HarnessResult) -> str:
        """Free-text summary of what happened. Capped to 2000 chars."""
        parts: list[str] = []
        if result.passed:
            parts.append(f"Outcome={result.actual_outcome}; passed=True.")
        else:
            reasons = "; ".join(result.failure_reasons) or "(no reasons listed)"
            parts.append(f"Outcome={result.actual_outcome}; passed=False. Reasons: {reasons}")
        if scenario.ground_truth.expected_tools:
            parts.append("Expected tools: " + ", ".join(scenario.ground_truth.expected_tools))
        if result.actual_tools:
            parts.append("Actual tools: " + ", ".join(result.actual_tools))
        if scenario.ground_truth.notes:
            parts.append("Scenario note: " + scenario.ground_truth.notes)
        return ("\n".join(parts))[:2000]
