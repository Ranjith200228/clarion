"""Tests for clinical advice request detection."""

from __future__ import annotations

import pytest
from clarion.sentinel.guardrails import detect_clinical_advice_request

# ---------- positives — agent must refuse ----------


@pytest.mark.parametrize(
    "patient_says",
    [
        # Should I X?
        "Should I take my drops tonight?",
        "Should I stop my blood thinners before surgery?",
        "Should I skip my morning dose?",
        "Should I double my dose if I missed yesterday?",
        # Is it safe / ok / dangerous?
        "Is it safe to drive after dilation?",
        "Is this safe to take with my heart medication?",
        "Is it ok to ice it before the appointment?",
        "Is this normal for someone with glaucoma?",
        # Can I take X with Y?
        "Can I take ibuprofen with my eye drops?",
        "Can I take Tylenol while I'm on amoxicillin?",
        # Dose questions.
        "What dose of acetaminophen should I take?",
        "How much ibuprofen can I take for this pain?",
        "Should I double the dose if it didn't work?",
        # Diagnostic.
        "Can you diagnose me over the phone?",
        "Is this cancer?",
        "Should I be worried about this lump?",
        # Refill / prescription.
        "I need a refill for my latanoprost drops",
        "Can you refill my prescription?",
        "Please refill my medication",
    ],
)
def test_detects_clinical_advice_request(patient_says: str) -> None:
    hit = detect_clinical_advice_request(patient_says)
    assert hit.fired, f"expected clinical_advice fire on: {patient_says}"
    assert hit.kind == "clinical_advice"
    assert hit.matched_pattern is not None


# ---------- negatives — must NOT refuse normal requests ----------


@pytest.mark.parametrize(
    "patient_says",
    [
        "I'd like to book a cataract pre-op consult",
        "Should I bring my insurance card?",  # logistical, not clinical
        "Should I bring sunglasses?",  # logistical
        "Is it ok if I bring my husband to the appointment?",
        "Is it ok to reschedule to next week?",
        "Do you accept Aetna?",
        "When is the next available slot?",
        "Can you tell me my appointment date?",
        "Is the parking free at the Downtown location?",
        "Should I park in the garage or the lot?",
        "I'd like to refill my coffee while I wait",  # 'refill' but not Rx
    ],
)
def test_does_not_fire_on_normal_traffic(patient_says: str) -> None:
    hit = detect_clinical_advice_request(patient_says)
    assert not hit.fired, (
        f"unexpected match on: {patient_says!r} " f"(pattern: {hit.matched_pattern})"
    )
    assert hit.kind == "safe"


def test_case_insensitive() -> None:
    assert detect_clinical_advice_request("SHOULD I TAKE MY DROPS").fired
    assert detect_clinical_advice_request("Should I Take My Drops").fired
