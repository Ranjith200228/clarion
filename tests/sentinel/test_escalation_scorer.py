"""Tests for the composite escalation scorer."""

from __future__ import annotations

from clarion.schemas import (
    EscalationSignals,
    EscalationWeights,
    JudgeVerdict,
    PolicyViolation,
)
from clarion.sentinel.escalation import (
    ConversationFacts,
    EscalationScorer,
    compute_stats,
)


def _facts(**overrides: object) -> ConversationFacts:
    base: dict[str, object] = {
        "user_messages": ["Hi, I'd like an appointment."],
        "agent_replies": ["Sure, what type?"],
        "tools_called": [],
        "judge": None,
        "expected_outcome_is_task": False,
    }
    base.update(overrides)
    return ConversationFacts(**base)  # type: ignore[arg-type]


# ---------- score boundaries ----------


def test_clean_conversation_scores_low() -> None:
    scorer = EscalationScorer()
    result = scorer.score(
        _facts(
            user_messages=["Book me a cataract consult"],
            agent_replies=["Done — see you June 15."],
            tools_called=["search_slots", "book_appointment"],
        )
    )
    assert result.score < 0.5
    assert result.should_escalate is False
    assert result.signals.frustration == 0.0
    assert result.signals.rule_conflict == 0.0


def test_one_signal_per_threshold_is_in_score() -> None:
    scorer = EscalationScorer()
    # Heuristic-only low-confidence: agent reply contains "I'm not sure"
    result = scorer.score(
        _facts(
            agent_replies=["I'm not sure about that, let me check."],
        )
    )
    assert result.signals.low_confidence > 0.0
    # One mild signal alone should NOT cross the default 0.5 threshold.
    assert result.score < 0.5


# ---------- per-signal: low_confidence ----------


def test_judge_confidence_inverts_into_low_confidence_signal() -> None:
    judge = JudgeVerdict(
        booking_correct=1.0,
        hallucination=0.0,
        confidence=0.2,  # judge is very unsure
    )
    scorer = EscalationScorer()
    result = scorer.score(_facts(judge=judge))
    assert result.signals.low_confidence == 0.8


def test_judge_high_confidence_yields_low_signal() -> None:
    judge = JudgeVerdict(
        booking_correct=1.0,
        hallucination=0.0,
        confidence=0.95,
    )
    scorer = EscalationScorer()
    result = scorer.score(_facts(judge=judge))
    assert result.signals.low_confidence < 0.1


# ---------- per-signal: repeated_clarification ----------


def test_repeated_clarification_saturates_at_max() -> None:
    scorer = EscalationScorer()
    # 3 clarification questions; default max_clarifications=3 → signal=1.0
    result = scorer.score(
        _facts(
            agent_replies=[
                "What appointment type?",
                "Which provider would you like?",
                "What date works for you?",
            ],
        )
    )
    assert result.signals.repeated_clarification == 1.0


def test_no_clarifications_is_zero() -> None:
    scorer = EscalationScorer()
    result = scorer.score(
        _facts(agent_replies=["You're all booked for June 15."]),
    )
    assert result.signals.repeated_clarification == 0.0


# ---------- per-signal: rule_conflict ----------


def test_judge_unsupported_claim_triggers_rule_conflict() -> None:
    judge = JudgeVerdict(
        hallucination=0.5,
        policy_violations=[PolicyViolation(kind="unsupported_claim", description="X")],
        confidence=0.8,
    )
    scorer = EscalationScorer()
    result = scorer.score(_facts(judge=judge))
    assert result.signals.rule_conflict == 1.0


def test_judge_invented_provider_triggers_rule_conflict() -> None:
    judge = JudgeVerdict(
        hallucination=0.9,
        policy_violations=[PolicyViolation(kind="invented_provider", description="Dr. Random")],
        confidence=0.9,
    )
    scorer = EscalationScorer()
    result = scorer.score(_facts(judge=judge))
    assert result.signals.rule_conflict == 1.0


def test_other_violation_does_not_trigger_rule_conflict() -> None:
    """Non-rule-conflict violations (e.g. clinical_advice_given) still
    surface in the judge, but the rule_conflict signal is specifically
    for fabrication."""
    judge = JudgeVerdict(
        hallucination=0.0,
        policy_violations=[PolicyViolation(kind="clinical_advice_given", description="X")],
        confidence=0.9,
    )
    scorer = EscalationScorer()
    result = scorer.score(_facts(judge=judge))
    assert result.signals.rule_conflict == 0.0


# ---------- per-signal: unsupported_request ----------


def test_pms_task_when_unexpected_fires_unsupported() -> None:
    scorer = EscalationScorer()
    result = scorer.score(
        _facts(
            tools_called=["create_pms_task"],
            expected_outcome_is_task=False,
        )
    )
    assert result.signals.unsupported_request == 1.0


def test_pms_task_when_expected_does_not_fire_unsupported() -> None:
    """Orthopedics cancel-as-task: PMS task is the right answer."""
    scorer = EscalationScorer()
    result = scorer.score(
        _facts(
            tools_called=["create_pms_task"],
            expected_outcome_is_task=True,
        )
    )
    assert result.signals.unsupported_request == 0.0


# ---------- per-signal: frustration ----------


def test_frustration_aggregates_across_user_turns() -> None:
    scorer = EscalationScorer()
    result = scorer.score(
        _facts(
            user_messages=[
                "I already told you my patient id",
                "Let me speak to a manager",
            ],
        )
    )
    assert result.signals.frustration > 0.5


# ---------- composite + threshold ----------


def test_score_clamps_to_unit_interval() -> None:
    # Force every signal to 1.0 and weights summing to > 1; verify clamp.
    scorer = EscalationScorer(
        weights=EscalationWeights(
            low_confidence=1.0,
            repeated_clarification=1.0,
            rule_conflict=1.0,
            frustration=1.0,
            unsupported_request=1.0,
        )
    )
    # Worst-case facts: judge confidence=0 → low_confidence=1.0;
    # rule_conflict=1.0; 3 clarification questions → repeated=1.0;
    # PMS task when not expected → unsupported=1.0; multiple frustration
    # cues across many turns → frustration saturates near 1.0.
    judge = JudgeVerdict(
        hallucination=1.0,
        policy_violations=[PolicyViolation(kind="unsupported_claim", description="X")],
        confidence=0.0,
    )
    result = scorer.score(
        _facts(
            user_messages=[
                "I already told you my id",
                "Let me speak to a manager",
                "This is ridiculous",
                "Are you serious",
                "Stop asking me",
                "Forget it",
                "Never mind",
                "PLEASE LISTEN TO ME!!!",
            ],
            agent_replies=["What?", "What?", "What?"],
            tools_called=["create_pms_task"],
            judge=judge,
            expected_outcome_is_task=False,
        )
    )
    assert 0.0 <= result.score <= 1.0
    # With all five signals at or near 1.0 and equal weights, the
    # normalized composite should be at the top of the range.
    assert result.score >= 0.95
    assert result.should_escalate is True


def test_decision_threshold_is_configurable() -> None:
    # Score sits around 0.18 with one mild signal.
    base = _facts(agent_replies=["I'm not sure, give me a moment."])
    high = EscalationScorer(decision_threshold=0.9).score(base)
    low = EscalationScorer(decision_threshold=0.1).score(base)
    assert high.should_escalate is False
    assert low.should_escalate is True


def test_signals_pydantic_constraints() -> None:
    # Defensive: signals are clamped via Pydantic validation.
    s = EscalationSignals(
        low_confidence=0.0,
        repeated_clarification=0.0,
        rule_conflict=0.0,
        frustration=0.0,
        unsupported_request=0.0,
    )
    assert s.low_confidence == 0.0


# ---------- precision/recall ----------


def test_compute_stats_perfect_predictions() -> None:
    stats = compute_stats([True, False, True, False], [True, False, True, False])
    assert stats.precision == 1.0
    assert stats.recall == 1.0
    assert stats.f1 == 1.0
    assert stats.accuracy == 1.0
    assert stats.false_positives == 0
    assert stats.false_negatives == 0


def test_compute_stats_all_wrong() -> None:
    stats = compute_stats([True, True, False, False], [False, False, True, True])
    assert stats.precision == 0.0
    assert stats.recall == 0.0
    assert stats.f1 == 0.0
    assert stats.accuracy == 0.0


def test_compute_stats_mixed_outcomes() -> None:
    # 2 TP, 1 FP, 1 FN, 0 TN
    stats = compute_stats(
        [True, True, True, False],
        [True, True, False, True],
    )
    assert stats.true_positives == 2
    assert stats.false_positives == 1
    assert stats.false_negatives == 1
    assert stats.true_negatives == 0
    # The schema rounds to 4 decimals; allow a tight tolerance.
    assert abs(stats.precision - 2 / 3) < 1e-3
    assert abs(stats.recall - 2 / 3) < 1e-3
    assert abs(stats.f1 - 2 / 3) < 1e-3


def test_compute_stats_length_mismatch_raises() -> None:
    try:
        compute_stats([True, False], [True])
    except ValueError as e:
        assert "same length" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_compute_stats_no_positives_safe() -> None:
    """When ground truth has no positives, recall denominator is zero —
    we report 0.0 instead of crashing."""
    stats = compute_stats([False, False, False], [False, False, False])
    assert stats.precision == 0.0
    assert stats.recall == 0.0
    assert stats.f1 == 0.0
    assert stats.accuracy == 1.0
