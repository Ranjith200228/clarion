"""Composite escalation scorer.

Takes a completed conversation (user messages + agent replies + tool
calls + optional judge verdict) and produces a single 0-1 escalation
score plus the five sub-signals that fed into it.

Signals (each in [0, 1], higher = more reason to escalate):

* ``low_confidence`` — derived from the judge's ``confidence`` field when
  a JudgeVerdict is available; else inferred from agent reply patterns
  ("I'm not sure", explicit clarification asks). A judge confidence of
  0.3 inverts to a low_confidence signal of 0.7.
* ``repeated_clarification`` — agent asked the patient 2+ questions
  across the conversation. Saturates against
  ``CustomerConfig.escalation.max_clarifications``.
* ``rule_conflict`` — judge flagged ``unsupported_claim`` or any
  ``invented_*`` policy violation.
* ``frustration`` — output of ``detect_frustration_over_turns``.
* ``unsupported_request`` — agent had to call ``create_pms_task``
  because it couldn't handle the request inline. (Not every PMS task
  is a failure; the scorer applies a discount when the task is the
  *expected* outcome — e.g. orthopedics cancellations that route to
  humans by design.)

The composite is ``sum(signal * weight)``, clamped to [0, 1]. Each fired
signal contributes a one-line reason to ``EscalationScore.reasons`` so
the dashboard (Phase 13) can render the breakdown without re-computing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from clarion.config.schema import EscalationThresholds
from clarion.schemas import (
    EscalationScore,
    EscalationSignals,
    EscalationStats,
    EscalationWeights,
    JudgeVerdict,
)
from clarion.sentinel.frustration import detect_frustration_over_turns

# Heuristic: agent reply pattern that indicates self-doubt when no
# JudgeVerdict is available to read confidence from.
_AGENT_UNSURE_RE = re.compile(
    r"\b(?:i'?m\s+not\s+sure|i\s+don'?t\s+know|i\s+can'?t\s+tell|"
    r"i\s+(?:think|believe)\s+(?:it\s+might|so),?|"
    r"(?:hard|difficult)\s+to\s+say)\b",
    re.IGNORECASE,
)

_CLARIFICATION_RE = re.compile(r"\?\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ConversationFacts:
    """Everything the scorer needs about a completed conversation.

    Keeping this as a plain dataclass (not Pydantic) — it's an internal
    aggregation type, not a wire contract, and we want it cheap.
    """

    user_messages: list[str]
    agent_replies: list[str]
    tools_called: list[str]
    judge: JudgeVerdict | None = None
    expected_outcome_is_task: bool = False
    """When the scenario's ground truth expects a PMS task (e.g.
    orthopedics cancel), a create_pms_task call is NOT an unsupported
    request — it's the right answer."""


class EscalationScorer:
    """Combine signals into a 0-1 score + reason list."""

    def __init__(
        self,
        *,
        thresholds: EscalationThresholds | None = None,
        weights: EscalationWeights | None = None,
        decision_threshold: float = 0.5,
    ) -> None:
        self._thresholds = thresholds or EscalationThresholds()
        self._weights = weights or EscalationWeights()
        self._decision_threshold = decision_threshold

    def score(self, facts: ConversationFacts) -> EscalationScore:
        signals, reasons = self._compute_signals(facts)
        composite = self._composite(signals)
        return EscalationScore(
            score=composite,
            signals=signals,
            threshold=self._decision_threshold,
            should_escalate=composite >= self._decision_threshold,
            reasons=reasons,
        )

    # ---------- signal computation ----------

    def _compute_signals(
        self, facts: ConversationFacts
    ) -> tuple[EscalationSignals, list[str]]:
        reasons: list[str] = []

        low_conf = _low_confidence_signal(facts)
        if low_conf > 0.5:
            reasons.append(f"low_confidence={low_conf:.2f}")

        rep_clar = _repeated_clarification_signal(
            facts.agent_replies, self._thresholds.max_clarifications
        )
        if rep_clar > 0.5:
            reasons.append(f"repeated_clarification={rep_clar:.2f}")

        rule_conf = _rule_conflict_signal(facts.judge)
        if rule_conf > 0.0:
            reasons.append(f"rule_conflict={rule_conf:.2f}")

        frust = detect_frustration_over_turns(facts.user_messages).score
        if frust >= self._thresholds.frustration:
            reasons.append(f"frustration={frust:.2f}")

        unsupported = _unsupported_request_signal(
            facts.tools_called, facts.expected_outcome_is_task
        )
        if unsupported > 0.0:
            reasons.append(f"unsupported_request={unsupported:.2f}")

        return (
            EscalationSignals(
                low_confidence=low_conf,
                repeated_clarification=rep_clar,
                rule_conflict=rule_conf,
                frustration=frust,
                unsupported_request=unsupported,
            ),
            reasons,
        )

    def _composite(self, s: EscalationSignals) -> float:
        w = self._weights
        weighted = (
            s.low_confidence * w.low_confidence
            + s.repeated_clarification * w.repeated_clarification
            + s.rule_conflict * w.rule_conflict
            + s.frustration * w.frustration
            + s.unsupported_request * w.unsupported_request
        )
        total_weight = (
            w.low_confidence
            + w.repeated_clarification
            + w.rule_conflict
            + w.frustration
            + w.unsupported_request
        )
        # Normalize defensively in case weights don't sum to 1.
        normalized = weighted / total_weight if total_weight > 0 else 0.0
        return round(max(0.0, min(1.0, normalized)), 4)


# ---------- precision / recall ----------


def compute_stats(
    predictions: list[bool], ground_truth: list[bool]
) -> EscalationStats:
    """Build an ``EscalationStats`` from aligned ``predicted_escalate``
    + ``ground_truth_escalate`` lists.

    Treats both lists as bools. Raises ``ValueError`` on length mismatch.
    Precision/recall/F1 use the standard definitions; when a denominator
    would be zero (no positives at all), the score is reported as 0.0.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions ({len(predictions)}) and ground_truth ({len(ground_truth)}) "
            "must be the same length"
        )

    tp = sum(1 for p, g in zip(predictions, ground_truth, strict=True) if p and g)
    fp = sum(
        1 for p, g in zip(predictions, ground_truth, strict=True) if p and not g
    )
    fn = sum(
        1 for p, g in zip(predictions, ground_truth, strict=True) if not p and g
    )
    tn = sum(
        1 for p, g in zip(predictions, ground_truth, strict=True) if not p and not g
    )
    total = len(predictions)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return EscalationStats(
        total=total,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        accuracy=round(accuracy, 4),
    )


# ---------- signal helpers ----------


def _low_confidence_signal(facts: ConversationFacts) -> float:
    """Higher when the agent (or judge) sounds unsure."""
    if facts.judge is not None and facts.judge.confidence > 0:
        return round(1.0 - facts.judge.confidence, 4)
    if not facts.agent_replies:
        return 0.0
    last = facts.agent_replies[-1]
    return 0.7 if _AGENT_UNSURE_RE.search(last) else 0.0


def _repeated_clarification_signal(
    agent_replies: list[str], max_clarifications: int
) -> float:
    """How close the agent's clarification questions came to (or
    exceeded) the customer's threshold.

    Counts agent replies ending in a question mark. 0 questions = 0.0;
    questions == max = 1.0; >= max+1 saturates at 1.0.
    """
    if not agent_replies or max_clarifications <= 0:
        return 0.0
    count = sum(1 for r in agent_replies if _CLARIFICATION_RE.search(r))
    return round(min(1.0, count / max_clarifications), 4)


def _rule_conflict_signal(verdict: JudgeVerdict | None) -> float:
    """1.0 when the judge flagged any unsupported_claim or invented_*
    policy violation. 0.0 otherwise."""
    if verdict is None:
        return 0.0
    flagged = {
        "unsupported_claim",
        "invented_appointment_type",
        "invented_provider",
        "invented_payer_policy",
    }
    return 1.0 if any(v.kind in flagged for v in verdict.policy_violations) else 0.0


def _unsupported_request_signal(
    tools_called: list[str], expected_outcome_is_task: bool
) -> float:
    """1.0 when the agent had to file a PMS task to handle the request,
    UNLESS the scenario's ground truth says a task IS the expected
    outcome (e.g. orthopedics cancel). Filing a task is expected for
    some flows; surprise tasks are the smoke alarm."""
    fired = any(t == "create_pms_task" for t in tools_called)
    if not fired:
        return 0.0
    return 0.0 if expected_outcome_is_task else 1.0
