"""Tests for clarion.multiagent.booking_fastpath and its integration
with BookingSpecialist._build_system_prompt.
"""

from __future__ import annotations

import pytest

from clarion.multiagent.booking_fastpath import (
    BookingFastPath,
    FastPathDecision,
    generate_training_data,
)


# ---------- unit: FastPathDecision ----------


def test_decision_is_confident_above_threshold() -> None:
    d = FastPathDecision(intent="book", confidence=0.80)
    assert d.is_confident(threshold=0.65)


def test_decision_not_confident_below_threshold() -> None:
    d = FastPathDecision(intent="search", confidence=0.50)
    assert not d.is_confident(threshold=0.65)


def test_fallback_intent_never_confident() -> None:
    # Even with probability 1.0, the fallback bucket is not "confident".
    d = FastPathDecision(intent="fallback", confidence=1.0)
    assert not d.is_confident()


# ---------- unit: generate_training_data ----------


def test_training_data_is_deterministic() -> None:
    t1, l1 = generate_training_data(seed=42, n_per_intent=10)
    t2, l2 = generate_training_data(seed=42, n_per_intent=10)
    assert t1 == t2
    assert l1 == l2


def test_training_data_sizes() -> None:
    texts, labels = generate_training_data(seed=0, n_per_intent=20)
    # 5 real intents + fallback = 6 * 20
    assert len(texts) == 120
    assert len(labels) == 120


def test_training_data_label_distribution() -> None:
    _, labels = generate_training_data(seed=0, n_per_intent=30)
    from collections import Counter
    counts = Counter(labels)
    for intent in ("search", "book", "reschedule", "cancel", "check_eligibility", "fallback"):
        assert counts[intent] == 30, f"expected 30 for {intent}, got {counts[intent]}"


# ---------- integration: BookingFastPath.train_default ----------


@pytest.fixture(scope="module")
def fastpath() -> BookingFastPath:
    return BookingFastPath.train_default(seed=42, n_per_intent=80)


def test_train_default_accuracy_above_floor(fastpath: BookingFastPath) -> None:
    # Synthetic data is clean; expect > 90% on the held-out split.
    assert fastpath.accuracy > 0.90, f"held-out acc={fastpath.accuracy:.3f}"


def test_train_default_metadata(fastpath: BookingFastPath) -> None:
    # 6 intents * 80 examples = 480 total; 80% train, 20% test.
    assert fastpath.n_train + fastpath.n_test == 480
    assert fastpath.n_test == 96  # 20% of 480
    assert set(fastpath.classes_) == {
        "book", "cancel", "check_eligibility", "fallback", "reschedule", "search"
    }


def test_predict_empty_string_returns_fallback(fastpath: BookingFastPath) -> None:
    d = fastpath.predict("")
    assert d.intent == "fallback"
    assert d.confidence == 0.0


def test_predict_whitespace_returns_fallback(fastpath: BookingFastPath) -> None:
    d = fastpath.predict("   \t\n  ")
    assert d.intent == "fallback"
    assert d.confidence == 0.0


@pytest.mark.parametrize("message,expected_intent", [
    ("what's available next week?", "search"),
    ("do you have any openings Monday?", "search"),
    ("book me a slot at 2pm with Dr. Smith", "book"),
    ("yes, book it", "book"),
    ("I need to reschedule my appointment", "reschedule"),
    ("can I move my Tuesday appointment?", "reschedule"),
    ("I need to cancel my Friday consult", "cancel"),
    ("please cancel my visit", "cancel"),
    ("do you take Aetna?", "check_eligibility"),
    ("is BCBS accepted here?", "check_eligibility"),
])
def test_predict_correct_intent(
    fastpath: BookingFastPath, message: str, expected_intent: str
) -> None:
    """Classifier picks the right intent class; calibration (confidence)
    is not asserted here — that depends on softmax spread across 6 classes."""
    d = fastpath.predict(message)
    assert d.intent == expected_intent, (
        f"message={message!r} → got {d.intent!r} (conf={d.confidence:.2f}), "
        f"expected {expected_intent!r}"
    )


def test_predict_noise_returns_fallback_or_low_confidence(
    fastpath: BookingFastPath,
) -> None:
    # These are from the noise bucket; the model should either emit
    # fallback or be below the 0.65 threshold for any real intent.
    noise = ["hi", "good morning", "where are you located?", "thanks"]
    for msg in noise:
        d = fastpath.predict(msg)
        # Either the intent is fallback, or the model is not confident.
        assert not d.is_confident() or d.intent == "fallback", (
            f"noise message {msg!r} was classified as {d.intent!r} with "
            f"confidence {d.confidence:.2f}"
        )


def test_hint_for_returns_none_on_noise(fastpath: BookingFastPath) -> None:
    assert fastpath.hint_for("hi") is None
    assert fastpath.hint_for("") is None


def test_hint_for_returns_string_when_confident(fastpath: BookingFastPath) -> None:
    """hint_for returns non-None when above threshold; use threshold=0 to
    bypass calibration and test the routing logic itself."""
    # At threshold=0, any non-fallback prediction yields a hint.
    hint = fastpath.hint_for("book me a slot at 2pm", threshold=0.0)
    assert hint is not None
    assert "BOOK" in hint


def test_hint_for_returns_none_for_fallback_at_threshold_zero(
    fastpath: BookingFastPath,
) -> None:
    """Even at threshold=0, the fallback intent never yields a hint."""
    hint = fastpath.hint_for("", threshold=0.0)
    assert hint is None


def test_hint_for_cancel_at_zero_threshold(fastpath: BookingFastPath) -> None:
    hint = fastpath.hint_for("please cancel my visit", threshold=0.0)
    assert hint is not None
    assert "CANCEL" in hint


def test_hint_for_custom_threshold(fastpath: BookingFastPath) -> None:
    # Setting threshold to 0.0 means every non-fallback prediction yields a hint.
    hint = fastpath.hint_for("book me something", threshold=0.0)
    assert hint is not None


# ---------- integration: BookingSpecialist._build_system_prompt ----------


def test_booking_specialist_prompt_contains_hint_when_fastpath_fires() -> None:
    """When the classifier returns a confident verdict, the hint appears in the
    system prompt. We mock the singleton to control confidence precisely."""
    from unittest.mock import MagicMock, patch

    from clarion.multiagent.specialists.booking import BookingSpecialist
    from clarion.config import CustomerConfig

    specialist = BookingSpecialist(
        llm=MagicMock(),
        customer=MagicMock(spec=CustomerConfig),
        ctx=MagicMock(),
    )

    fixed_hint = "Hint: the caller most likely wants to BOOK a specific slot."

    with patch(
        "clarion.multiagent.specialists.booking._get_fastpath"
    ) as mock_gfp:
        mock_fp = MagicMock()
        mock_fp.hint_for.return_value = fixed_hint
        mock_gfp.return_value = mock_fp

        prompt = specialist._build_system_prompt(user_message="book me a slot Monday")

    assert "Hint:" in prompt
    assert "BOOK" in prompt
    # Ensure the base persona is still present.
    assert "booking specialist" in prompt.lower()


def test_booking_specialist_prompt_no_hint_for_noise() -> None:
    """When the classifier returns None (not confident), no hint is appended."""
    from unittest.mock import MagicMock, patch

    from clarion.multiagent.specialists.booking import BookingSpecialist
    from clarion.config import CustomerConfig

    specialist = BookingSpecialist(
        llm=MagicMock(),
        customer=MagicMock(spec=CustomerConfig),
        ctx=MagicMock(),
    )

    with patch(
        "clarion.multiagent.specialists.booking._get_fastpath"
    ) as mock_gfp:
        mock_fp = MagicMock()
        mock_fp.hint_for.return_value = None
        mock_gfp.return_value = mock_fp

        prompt = specialist._build_system_prompt(user_message="hello there")

    assert "Hint:" not in prompt


def test_booking_specialist_prompt_base_always_present() -> None:
    """The persona and tool-scoping line are always present regardless of hint."""
    from unittest.mock import MagicMock, patch

    from clarion.multiagent.specialists.booking import BookingSpecialist
    from clarion.config import CustomerConfig

    specialist = BookingSpecialist(
        llm=MagicMock(),
        customer=MagicMock(spec=CustomerConfig),
        ctx=MagicMock(),
    )

    with patch(
        "clarion.multiagent.specialists.booking._get_fastpath"
    ) as mock_gfp:
        mock_fp = MagicMock()
        mock_fp.hint_for.return_value = None
        mock_gfp.return_value = mock_fp

        prompt = specialist._build_system_prompt(user_message="anything")

    assert "booking specialist" in prompt.lower()
    assert "search_slots" in prompt
    assert "book_appointment" in prompt
