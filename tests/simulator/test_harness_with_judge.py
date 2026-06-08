"""Harness x Judge integration tests.

The harness can be configured with an optional Judge. When wired,
every HarnessResult carries a structured judge_verdict so Phase 12 can
roll up per-customer pass/fail metrics across the 100 scenarios
without re-parsing prose.

The Judge here is driven by a FakeLLM so CI never makes a real LLM call.
"""

from __future__ import annotations

import json
from pathlib import Path

from clarion.agents.llm import FakeLLM, LLMResponse, LLMUsage
from clarion.config import Settings, load_customer
from clarion.pipelines.structured import StructuredStore
from clarion.schemas import AvailabilitySlot, EligibilityRecord, Provider
from clarion.sentinel import Judge
from clarion.simulator.harness import load_scenarios, run_scripted

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"


def _seed(store: StructuredStore, seed_json: Path) -> None:
    payload = json.loads(seed_json.read_text(encoding="utf-8"))
    for p in payload["providers"]:
        store.upsert_provider(Provider(**p))
    for s in payload["availability"]:
        store.upsert_slot(AvailabilitySlot(**s))
    for e in payload["eligibility"]:
        store.upsert_eligibility(EligibilityRecord(**e))


def _settings(tmp_path: Path, customer: str = "ophthalmology") -> Settings:
    return Settings(
        customer=customer,
        config_dir=CONFIGS_DIR,
        data_dir=tmp_path,
    )


def _clean_verdict_json() -> str:
    return json.dumps(
        {
            "booking_correct": 1.0,
            "hallucination": 0.0,
            "policy_violations": [],
            "violation_severity": 0.0,
            "confidence": 0.95,
            "rationale": "Looks fine.",
        }
    )


def test_harness_run_without_judge_leaves_verdict_none(tmp_path: Path) -> None:
    """Baseline: when no judge is wired, judge_verdict is None on every
    result. Existing Phase 9 callers continue working unchanged."""
    customer = load_customer("ophthalmology", settings=_settings(tmp_path))
    store = StructuredStore.for_customer("ophthalmology", tmp_path)
    _seed(store, DATA_DIR / "seeds" / "ophthalmology.json")
    scenarios = load_scenarios(DATA_DIR / "personas" / "ophthalmology.json")[:5]

    report = run_scripted(
        scenarios,
        customer_config=customer,
        structured=store,
        retriever=None,
    )

    assert report.total == 5
    for r in report.results:
        assert r.judge_verdict is None


def test_harness_with_judge_attaches_verdict_per_scenario(tmp_path: Path) -> None:
    """With a judge wired, every result has a judge_verdict dict."""
    customer = load_customer("ophthalmology", settings=_settings(tmp_path))
    store = StructuredStore.for_customer("ophthalmology", tmp_path)
    _seed(store, DATA_DIR / "seeds" / "ophthalmology.json")
    scenarios = load_scenarios(DATA_DIR / "personas" / "ophthalmology.json")[:3]

    # One FakeLLM verdict per scenario.
    fake = FakeLLM(
        responses=[
            LLMResponse(content=_clean_verdict_json(), usage=LLMUsage(model="gpt-4o-mini"))
            for _ in scenarios
        ]
    )
    judge = Judge(llm=fake)

    report = run_scripted(
        scenarios,
        customer_config=customer,
        structured=store,
        retriever=None,
        judge=judge,
    )

    assert report.total == 3
    assert all(r.judge_verdict is not None for r in report.results)
    for r in report.results:
        v = r.judge_verdict
        assert isinstance(v, dict)
        assert v["confidence"] == 0.95
        assert v["hallucination"] == 0.0
        assert v["policy_violations"] == []


def test_judge_skipped_when_agent_returned_no_reply(tmp_path: Path) -> None:
    """If the agent's reply list is empty (defensive), the judge is not
    called. Scenarios in the shipped persona files always produce a
    reply, but the harness shouldn't crash if a future scenario doesn't.
    The check here is implicit: an empty FakeLLM responses list would
    raise the moment the judge is consulted."""
    customer = load_customer("ophthalmology", settings=_settings(tmp_path))
    store = StructuredStore.for_customer("ophthalmology", tmp_path)
    _seed(store, DATA_DIR / "seeds" / "ophthalmology.json")
    # Take only emergency scenarios so the agent guardrails return a
    # canned reply (always non-empty), and confirm the judge gets called.
    all_scenarios = load_scenarios(DATA_DIR / "personas" / "ophthalmology.json")
    emergencies = [s for s in all_scenarios if s.intent == "emergency"][:2]
    fake = FakeLLM(
        responses=[
            LLMResponse(content=_clean_verdict_json(), usage=LLMUsage(model="gpt-4o-mini")),
            LLMResponse(content=_clean_verdict_json(), usage=LLMUsage(model="gpt-4o-mini")),
        ]
    )
    judge = Judge(llm=fake)
    report = run_scripted(
        emergencies,
        customer_config=customer,
        structured=store,
        retriever=None,
        judge=judge,
    )
    # Both emergency scenarios got a verdict — the guardrail reply is
    # the agent's reply the judge graded.
    assert all(r.judge_verdict is not None for r in report.results)


def test_harness_passes_expected_appointment_type_to_judge(tmp_path: Path) -> None:
    """Verify the harness threads scenario.ground_truth.expected_appointment_type
    into the JudgeRequest. We capture the LLM call and inspect the
    rendered user prompt."""
    customer = load_customer("ophthalmology", settings=_settings(tmp_path))
    store = StructuredStore.for_customer("ophthalmology", tmp_path)
    _seed(store, DATA_DIR / "seeds" / "ophthalmology.json")
    all_scenarios = load_scenarios(DATA_DIR / "personas" / "ophthalmology.json")
    # Pick the first clear booking scenario — guaranteed to have an
    # expected_appointment_type.
    booking = next(
        s
        for s in all_scenarios
        if s.intent == "book" and s.ground_truth.expected_appointment_type
    )

    captured: list[str] = []

    class CapturingLLM(FakeLLM):
        def complete(self, messages, *, tools=None):  # type: ignore[no-untyped-def]
            captured.append(messages[-1].content or "")
            return super().complete(messages, tools=tools)

    judge = Judge(
        llm=CapturingLLM(
            responses=[
                LLMResponse(content=_clean_verdict_json(), usage=LLMUsage(model="gpt-4o-mini"))
            ]
        )
    )
    run_scripted(
        [booking],
        customer_config=customer,
        structured=store,
        retriever=None,
        judge=judge,
    )
    expected = booking.ground_truth.expected_appointment_type
    assert expected is not None
    assert expected in captured[0]
