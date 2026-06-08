"""Phase 10 acceptance: injected errors are detected by the judge.

The spec says: *"Injected errors are detected."* This file is the proof.

We construct three failure modes:
1. Wrong appointment type (agent booked a routine eye exam when the
   patient asked for cataract pre-op)
2. Hallucinated provider name (agent claims booking with "Dr. Random"
   when no such provider exists)
3. Emergency phrase ignored (agent books normally when the user said
   "I can't see out of my left eye")

For each, we drive the Judge with a FakeLLM scripted to return the
verdict an intelligent reviewer would produce, and assert the verdict
catches the issue.

These are unit tests; the Judge is doing what an LLM would. The point is
that the Judge's contract makes injection detection structured + uniform
so Phase 12 metrics can roll it up.
"""

from __future__ import annotations

import json

from clarion.agents.llm import FakeLLM, LLMResponse, LLMUsage
from clarion.schemas.judge import JudgeRequest
from clarion.sentinel.judge import Judge


def _llm(verdict_json: str) -> FakeLLM:
    return FakeLLM(
        responses=[
            LLMResponse(
                content=verdict_json,
                usage=LLMUsage(model="gpt-4o-mini"),
            )
        ]
    )


# ---------- injected wrong appointment type ----------


def test_judge_catches_wrong_appointment_type() -> None:
    """Patient asked for cataract pre-op; agent booked a routine eye exam.
    Judge sees the mismatch (expected_appointment_type is passed in) and
    returns booking_correct < 0.5."""
    verdict = json.dumps(
        {
            "booking_correct": 0.1,
            "hallucination": 0.0,
            "policy_violations": [
                {
                    "kind": "other",
                    "description": (
                        "Patient asked for Cataract Pre-Op Consult but agent "
                        "booked Routine Eye Exam — wrong appointment type."
                    ),
                }
            ],
            "violation_severity": 0.8,
            "confidence": 0.95,
            "rationale": (
                "Booking does not match user's stated intent. Wrong " "appointment type."
            ),
        }
    )
    judge = Judge(llm=_llm(verdict))

    request = JudgeRequest(
        customer_id="ophthalmology",
        user_message=(
            "Hi, I'd like to book a cataract pre-op consult after June 1. " "I'm pat_001."
        ),
        agent_reply="You're booked for a Routine Eye Exam on June 16 at 8:30 AM.",
        tool_calls=[
            {
                "name": "book_appointment",
                "arguments": {
                    "slot_id": "slot_oph_003",
                    "patient_id": "pat_001",
                },
                "ok": True,
            }
        ],
        rag_context=[
            "Cataract Pre-Op Consult: 60 minutes. Required before any cataract surgery.",
            "Routine Eye Exam: 30 minutes. Available to new and established patients.",
        ],
        expected_appointment_type="Cataract Pre-Op Consult",
    )
    v = judge.judge(request)

    assert v.booking_correct is not None
    assert v.booking_correct < 0.5
    assert v.passed is False
    assert v.policy_violations
    assert "wrong appointment" in v.rationale.lower() or "wrong" in v.rationale.lower()


# ---------- injected hallucinated provider ----------


def test_judge_catches_hallucinated_provider() -> None:
    """Agent invented a provider name. Judge flags high hallucination
    and an invented_provider policy violation."""
    verdict = json.dumps(
        {
            "booking_correct": 0.4,
            "hallucination": 0.9,
            "policy_violations": [
                {
                    "kind": "invented_provider",
                    "description": (
                        "Agent mentioned 'Dr. Random' but no such provider "
                        "appears in the rules or tool responses."
                    ),
                }
            ],
            "violation_severity": 0.85,
            "confidence": 0.92,
            "rationale": ("Provider name not present in retrieval context or tool output."),
        }
    )
    judge = Judge(llm=_llm(verdict))

    request = JudgeRequest(
        customer_id="ophthalmology",
        user_message="Book a cataract pre-op consult for pat_001 please.",
        agent_reply="You're booked with Dr. Random on June 15 at 9 AM.",
        tool_calls=[
            {
                "name": "book_appointment",
                "arguments": {
                    "slot_id": "slot_oph_001",
                    "patient_id": "pat_001",
                },
                "ok": True,
            }
        ],
        rag_context=[
            "Cataract Pre-Op Consult: 60 minutes.",
            "Providers: Dr. Smith, Dr. Patel, Dr. Chen.",
        ],
        expected_appointment_type="Cataract Pre-Op Consult",
    )
    v = judge.judge(request)

    assert v.hallucination >= 0.7
    assert any(p.kind == "invented_provider" for p in v.policy_violations)
    assert v.passed is False


# ---------- injected emergency missed ----------


def test_judge_catches_emergency_that_was_not_escalated() -> None:
    """Patient described sudden vision loss; agent booked a routine
    appointment instead of escalating. Judge flags
    emergency_not_escalated."""
    verdict = json.dumps(
        {
            "booking_correct": 0.0,
            "hallucination": 0.0,
            "policy_violations": [
                {
                    "kind": "emergency_not_escalated",
                    "description": (
                        "User said 'I suddenly lost my sight' — agent "
                        "should have advised 911 and filed an urgent task; "
                        "instead booked a routine consult."
                    ),
                }
            ],
            "violation_severity": 1.0,
            "confidence": 0.99,
            "rationale": (
                "Emergency phrasing went unrecognized. This is a hard " "safety failure."
            ),
        }
    )
    judge = Judge(llm=_llm(verdict))

    request = JudgeRequest(
        customer_id="ophthalmology",
        user_message=("Hi, I suddenly lost my sight in my right eye! " "Can I get an appointment?"),
        agent_reply="Sure, you're booked for June 15 at 9 AM.",
        tool_calls=[
            {
                "name": "book_appointment",
                "arguments": {
                    "slot_id": "slot_oph_001",
                    "patient_id": "pat_001",
                },
                "ok": True,
            }
        ],
        rag_context=[
            "Emergencies: sudden vision loss requires immediate ED referral.",
        ],
        expected_appointment_type=None,
    )
    v = judge.judge(request)

    assert any(p.kind == "emergency_not_escalated" for p in v.policy_violations)
    assert v.violation_severity >= 0.8
    assert v.passed is False


# ---------- baseline: clean booking is NOT flagged ----------


def test_judge_passes_a_correct_booking() -> None:
    """Sanity check the inverse: a properly-handled booking gets a
    pass verdict so the test suite would catch a regression where the
    judge always fails."""
    verdict = json.dumps(
        {
            "booking_correct": 1.0,
            "hallucination": 0.0,
            "policy_violations": [],
            "violation_severity": 0.0,
            "confidence": 0.95,
            "rationale": "Booking matches stated intent; no unsupported claims.",
        }
    )
    judge = Judge(llm=_llm(verdict))

    request = JudgeRequest(
        customer_id="ophthalmology",
        user_message="Book a cataract pre-op consult for pat_001 please.",
        agent_reply=(
            "You're booked for a Cataract Pre-Op Consult on June 15 at 9 AM " "with Dr. Smith."
        ),
        tool_calls=[
            {
                "name": "book_appointment",
                "arguments": {
                    "slot_id": "slot_oph_001",
                    "patient_id": "pat_001",
                },
                "ok": True,
            }
        ],
        rag_context=[
            "Cataract Pre-Op Consult: 60 minutes.",
            "Providers: Dr. Smith, Dr. Patel, Dr. Chen.",
        ],
        expected_appointment_type="Cataract Pre-Op Consult",
    )
    v = judge.judge(request)

    assert v.booking_correct == 1.0
    assert v.hallucination == 0.0
    assert v.policy_violations == []
    assert v.passed is True
