"""Tests for the scenario generator."""

from __future__ import annotations

from collections import Counter

from clarion.schemas import Scenario
from clarion.simulator import OPHTHALMOLOGY, ORTHOPEDICS, generate
from clarion.simulator.templates import DISTRIBUTION


def test_generates_one_hundred_per_customer() -> None:
    assert len(generate(OPHTHALMOLOGY)) == 100
    assert len(generate(ORTHOPEDICS)) == 100


def test_generator_is_deterministic() -> None:
    a = generate(OPHTHALMOLOGY, seed=42)
    b = generate(OPHTHALMOLOGY, seed=42)
    assert [s.scenario_id for s in a] == [s.scenario_id for s in b]
    assert [s.messages for s in a] == [s.messages for s in b]


def test_different_seeds_produce_different_phrasings() -> None:
    a = generate(OPHTHALMOLOGY, seed=42)
    b = generate(OPHTHALMOLOGY, seed=123)
    # Distinct enough that at least one message differs.
    assert any(s1.messages != s2.messages for s1, s2 in zip(a, b, strict=True))


def test_distribution_matches_spec() -> None:
    scenarios = generate(OPHTHALMOLOGY)
    counts = Counter((s.difficulty, s.intent) for s in scenarios)
    for difficulty, intent, expected_count in DISTRIBUTION:
        assert counts[(difficulty, intent)] == expected_count, (
            f"{difficulty}/{intent}: got {counts[(difficulty, intent)]}, "
            f"expected {expected_count}"
        )


def test_scenario_ids_are_unique() -> None:
    scenarios = generate(OPHTHALMOLOGY)
    ids = [s.scenario_id for s in scenarios]
    assert len(set(ids)) == len(ids)


def test_emergency_scenarios_have_empty_script() -> None:
    scenarios = generate(OPHTHALMOLOGY)
    for s in scenarios:
        if s.intent == "emergency":
            assert s.llm_script == []
            assert s.ground_truth.expected_outcome == "escalated_emergency"
            assert s.ground_truth.should_escalate is True


def test_clinical_advice_scenarios_have_empty_script() -> None:
    scenarios = generate(OPHTHALMOLOGY)
    for s in scenarios:
        if s.intent == "clinical_advice":
            assert s.llm_script == []
            assert s.ground_truth.expected_outcome == "refused_clinical"


def test_orthopedics_cancel_routes_to_task() -> None:
    """Orthopedics doesn't have cancel_appointment — ground truth must
    reflect the task-creation fallback."""
    scenarios = generate(ORTHOPEDICS)
    cancels = [s for s in scenarios if s.intent == "cancel"]
    assert cancels
    for s in cancels:
        assert s.ground_truth.expected_tools == ["create_pms_task"]
        assert s.ground_truth.expected_outcome == "task_created"


def test_ophthalmology_cancel_uses_cancel_tool() -> None:
    scenarios = generate(OPHTHALMOLOGY)
    cancels = [s for s in scenarios if s.intent == "cancel"]
    assert cancels
    for s in cancels:
        assert "cancel_appointment" in s.ground_truth.expected_tools
        assert s.ground_truth.expected_outcome == "cancelled"


def test_clear_booking_scenarios_carry_search_then_book_script() -> None:
    scenarios = generate(OPHTHALMOLOGY)
    clear_books = [s for s in scenarios if s.intent == "book" and s.difficulty == "clear"]
    assert clear_books
    for s in clear_books:
        tool_names_in_script = [tc["name"] for step in s.llm_script for tc in step.tool_calls]
        assert "search_slots" in tool_names_in_script
        assert "book_appointment" in tool_names_in_script


def test_every_scenario_validates_as_schema(  # type: ignore[no-untyped-def]
) -> None:
    """Every generated scenario must be a valid Scenario instance —
    catches a Pydantic constraint regression early."""
    scenarios = generate(OPHTHALMOLOGY)
    for s in scenarios:
        assert isinstance(s, Scenario)
        # Round-trip through model_dump / construct again to catch any
        # subtle invariant the generator might have skipped.
        Scenario(**s.model_dump())
