"""Unit tests for the Sentinel LLM-as-judge."""

from __future__ import annotations

import json

import pytest
from clarion.agents.llm import FakeLLM, LLMResponse, LLMUsage
from clarion.schemas.judge import JudgeRequest, JudgeVerdict
from clarion.sentinel.judge import Judge


def _verdict_json(**overrides: object) -> str:
    payload = {
        "booking_correct": 1.0,
        "hallucination": 0.0,
        "policy_violations": [],
        "violation_severity": 0.0,
        "confidence": 0.9,
        "rationale": "Looks fine.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _llm(*responses: str) -> FakeLLM:
    return FakeLLM(
        responses=[LLMResponse(content=r, usage=LLMUsage(model="gpt-4o-mini")) for r in responses]
    )


def _req(**overrides: object) -> JudgeRequest:
    base: dict[str, object] = {
        "customer_id": "ophthalmology",
        "user_message": "Book me a cataract consult",
        "agent_reply": "Booked for June 15.",
        "tool_calls": [{"name": "book_appointment", "arguments": {}, "ok": True}],
        "rag_context": ["Cataract Pre-Op Consult: 60 minutes."],
    }
    base.update(overrides)
    return JudgeRequest(**base)  # type: ignore[arg-type]


# ---------- happy path ----------


def test_clean_pass_returns_passed_verdict() -> None:
    judge = Judge(llm=_llm(_verdict_json()))
    verdict = judge.judge(_req())
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.booking_correct == 1.0
    assert verdict.hallucination == 0.0
    assert verdict.policy_violations == []
    assert verdict.passed is True


# ---------- correctness, hallucination, policy ----------


def test_low_booking_correctness_fails_passed_property() -> None:
    judge = Judge(llm=_llm(_verdict_json(booking_correct=0.2)))
    verdict = judge.judge(_req())
    assert verdict.booking_correct == 0.2
    assert verdict.passed is False


def test_high_hallucination_fails_passed_property() -> None:
    judge = Judge(llm=_llm(_verdict_json(hallucination=0.8)))
    verdict = judge.judge(_req())
    assert verdict.hallucination == 0.8
    assert verdict.passed is False


def test_policy_violation_fails_passed_property() -> None:
    judge = Judge(
        llm=_llm(
            _verdict_json(
                policy_violations=[
                    {
                        "kind": "clinical_advice_given",
                        "description": "Suggested dose change.",
                    }
                ],
                violation_severity=0.8,
            )
        )
    )
    verdict = judge.judge(_req())
    assert len(verdict.policy_violations) == 1
    assert verdict.policy_violations[0].kind == "clinical_advice_given"
    assert verdict.violation_severity == 0.8
    assert verdict.passed is False


def test_booking_correct_null_means_non_booking_turn() -> None:
    judge = Judge(llm=_llm(_verdict_json(booking_correct=None)))
    verdict = judge.judge(_req(tool_calls=[]))
    assert verdict.booking_correct is None
    assert verdict.passed is True  # non-booking turn can still pass


# ---------- defensive parsing ----------


def test_handles_markdown_fenced_json() -> None:
    fenced = "```json\n" + _verdict_json(rationale="fenced") + "\n```"
    judge = Judge(llm=_llm(fenced))
    verdict = judge.judge(_req())
    assert verdict.rationale == "fenced"


def test_handles_malformed_json_with_parse_failure_verdict() -> None:
    judge = Judge(llm=_llm("this is not json at all"))
    verdict = judge.judge(_req())
    assert verdict.confidence == 0.0
    assert "parse failure" in verdict.rationale.lower()
    # Doesn't blow up, doesn't accidentally pass.
    assert verdict.passed is True  # No violations registered, hallucination=0
    # But confidence==0 is the low-trust signal callers should check.


def test_handles_non_object_payload() -> None:
    judge = Judge(llm=_llm('"just a string"'))
    verdict = judge.judge(_req())
    assert verdict.confidence == 0.0
    assert "json object" in verdict.rationale.lower()


def test_clamps_out_of_range_floats() -> None:
    judge = Judge(
        llm=_llm(
            _verdict_json(
                hallucination=1.5,  # above 1
                violation_severity=-0.3,  # below 0
                confidence=42,  # way out
            )
        )
    )
    verdict = judge.judge(_req())
    assert verdict.hallucination == 1.0
    assert verdict.violation_severity == 0.0
    assert verdict.confidence == 1.0


def test_handles_missing_optional_fields() -> None:
    partial = json.dumps({"hallucination": 0.1})  # everything else missing
    judge = Judge(llm=_llm(partial))
    verdict = judge.judge(_req())
    assert verdict.hallucination == 0.1
    assert verdict.booking_correct is None
    assert verdict.policy_violations == []


# ---------- prompt assembly ----------


def test_reflect_uses_second_person_voice() -> None:
    judge_called: list[str] = []

    class CapturingLLM(FakeLLM):
        def complete(self, messages, *, tools=None):  # type: ignore[no-untyped-def]
            judge_called.append(messages[-1].content or "")
            return super().complete(messages, tools=tools)

    judge = Judge(llm=CapturingLLM(responses=[LLMResponse(content=_verdict_json())]))
    judge.reflect(_req())
    assert "you (Clarion)" in judge_called[0]
    assert "Clarion replied" not in judge_called[0]


def test_judge_uses_third_person_voice() -> None:
    captured: list[str] = []

    class CapturingLLM(FakeLLM):
        def complete(self, messages, *, tools=None):  # type: ignore[no-untyped-def]
            captured.append(messages[-1].content or "")
            return super().complete(messages, tools=tools)

    judge = Judge(llm=CapturingLLM(responses=[LLMResponse(content=_verdict_json())]))
    judge.judge(_req())
    # Third-person framing — "Clarion replied" appears in the rendered
    # prompt.
    assert "Clarion replied" in captured[0]


def test_request_validates_required_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        JudgeRequest(  # type: ignore[call-arg]
            customer_id="oph",
            user_message="",  # min_length=1
            agent_reply="",
        )
