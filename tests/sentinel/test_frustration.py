"""Tests for the frustration text-pattern detector."""

from __future__ import annotations

import pytest
from clarion.sentinel.frustration import (
    detect_frustration,
    detect_frustration_over_turns,
)

# ---------- positives — should fire ----------


@pytest.mark.parametrize(
    "msg",
    [
        # "I already told you" family
        "I already told you I want a cataract consult",
        "I already said this twice",
        "As I told you before, my patient id is pat_001",
        "You're not listening to me",
        "You are not listening at all",
        # Repetition / impatience
        "Just answer my question",
        "Why can't you understand",
        "Why won't you do it",
        "That's not what I asked",
        # Human demand
        "Let me speak to a human",
        "Put me through to a manager",
        "I want to speak to a supervisor",
        "I need to speak to someone in charge",
        # Anger markers
        "This is ridiculous",
        "Are you kidding me right now",
        "Are you serious",
        # Frustrated commands
        "Stop asking me the same thing",
        "Forget it",
        "Never mind, just hang up",
    ],
)
def test_frustration_patterns_fire(msg: str) -> None:
    result = detect_frustration(msg)
    assert result.score > 0.0, f"no fire on {msg!r}"
    assert result.patterns, f"no matched patterns on {msg!r}"


def test_shouting_counts() -> None:
    result = detect_frustration("PLEASE LISTEN TO ME RIGHT NOW")
    assert result.has_shouting is True
    assert result.score > 0.0


def test_excessive_punctuation_counts() -> None:
    result = detect_frustration("really??? are you serious???")
    assert result.excessive_punctuation is True
    assert result.score > 0.0


# ---------- negatives — clean traffic ----------


@pytest.mark.parametrize(
    "msg",
    [
        "Hi, I'd like to book a cataract consult.",
        "Could you tell me your hours?",
        "My patient id is pat_001.",
        "Do you accept Aetna insurance?",
        "Please send me to room 101.",  # 'send me' is innocuous
        "I want to know my appointment date.",
        "USA",  # capitalized, but short — not shouting
        "ER",
        "MRI scheduled for Tuesday.",
    ],
)
def test_clean_traffic_does_not_fire(msg: str) -> None:
    result = detect_frustration(msg)
    assert result.score == 0.0, f"false fire on {msg!r} (patterns: {result.patterns})"
    assert not result.has_shouting


# ---------- saturation behavior ----------


def test_one_hit_below_default_threshold() -> None:
    """A single frustration cue shouldn't trigger the default 0.5
    escalation threshold by itself — that takes two."""
    result = detect_frustration("This is ridiculous")
    assert result.score < 0.5


def test_two_hits_cross_threshold() -> None:
    """Two cues together cross the default 0.5 threshold."""
    result = detect_frustration("This is ridiculous! I already told you my patient id!!!")
    # Patterns: "this is ridiculous" + "I already told"
    # Plus excessive_punctuation → 3 raw hits → ~0.78
    assert result.score >= 0.5


def test_score_saturates_below_one() -> None:
    """Even maximally angry input doesn't exceed 1.0."""
    msg = (
        "This is RIDICULOUS!!! I already told you, I already said, "
        "let me speak to a manager, are you serious, stop asking, "
        "forget it!!!"
    )
    result = detect_frustration(msg)
    assert result.score <= 1.0
    assert result.score > 0.9


# ---------- multi-turn accumulation ----------


def test_multi_turn_score_higher_than_single_turn() -> None:
    """Two angry turns should aggregate to a higher score than one
    of them alone."""
    msg = "This is ridiculous"
    single = detect_frustration(msg).score
    aggregated = detect_frustration_over_turns([msg, msg]).score
    assert aggregated > single


def test_multi_turn_empty_returns_zero() -> None:
    result = detect_frustration_over_turns([])
    assert result.score == 0.0
    assert result.patterns == []


def test_multi_turn_collects_all_patterns() -> None:
    result = detect_frustration_over_turns(["I already told you", "let me speak to a manager"])
    assert len(result.patterns) >= 2
