"""Tests for emergency detection."""

from __future__ import annotations

import pytest
from clarion.sentinel.guardrails import detect_emergency

# ---------- positives ----------


@pytest.mark.parametrize(
    "patient_says",
    [
        # General.
        "I think I'm having a stroke, please help",
        "Tell my husband I might be having a heart attack",
        "He just passed out on the floor",
        "She's unconscious, what do I do",
        "I can't breathe properly",
        "I called 911 already, what now",
        "He's in the emergency room right now",
        "this is an emergency",
        "I'm in excruciating pain",
        # Ophthalmology.
        "I suddenly lost my sight in my right eye",
        "my vision went black for a few seconds",
        "I can't see out of my left eye",
        "I have eye trauma from yesterday",
        "I had a chemical splash in my eye",
        # Orthopedic.
        "I think I have a compound fracture",
        "the bone is sticking out of my leg",
        "I can't feel my foot",
        "my leg is cold and blue",
        "I had a head injury this morning",
        "I lost control of my bladder",
    ],
)
def test_detects_emergency(patient_says: str) -> None:
    hit = detect_emergency(patient_says)
    assert hit.fired
    assert hit.kind == "emergency"
    assert hit.matched_pattern is not None
    assert "911" in hit.reason or "emergency" in hit.reason.lower()


# ---------- negatives — legitimate booking / FAQ traffic ----------


@pytest.mark.parametrize(
    "patient_says",
    [
        "I'd like to book a cataract pre-op consult",
        "Do you accept Kaiser insurance?",
        "Can I reschedule my appointment?",
        "What's the duration of a glaucoma follow-up?",
        "I'm a new patient looking for a joint consult",
        "I have some mild pain in my knee, just want to be seen",  # vague, not severe
        "I broke my glasses",  # 'broke' triggered nothing
        "I see better with my contacts",  # 'see' triggered nothing
        "I'd like to see Dr. Smith",
        "We can call back later",
    ],
)
def test_does_not_fire_on_routine_traffic(patient_says: str) -> None:
    hit = detect_emergency(patient_says)
    assert not hit.fired, f"unexpected match on routine text: {hit.matched_pattern}"
    assert hit.kind == "safe"


def test_case_insensitive() -> None:
    assert detect_emergency("THIS IS AN EMERGENCY").fired
    assert detect_emergency("This Is An Emergency").fired
