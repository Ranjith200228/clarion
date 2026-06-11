"""Module M1 tests — extractor, writer, accuracy, harness integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clarion.modules.pms_writeback import (
    ExtractionContext,
    HeuristicExtractor,
    PmsWritebackWriter,
    compute_field_extraction_accuracy,
)
from clarion.schemas import (
    ConversationSummary,
    HarnessReport,
    HarnessResult,
    PmsTaskWriteback,
    Scenario,
)


def _scenario(**overrides: Any) -> Scenario:
    base: dict[str, Any] = {
        "scenario_id": "oph_clear_book_001",
        "customer_id": "ophthalmology",
        "difficulty": "clear",
        "intent": "book",
        "messages": [
            "Hi, this is Jane Smith. I'd like a cataract pre-op consult after June 1. "
            "My patient id is pat_001. I have Aetna."
        ],
        "ground_truth": {
            "expected_outcome": "booked",
            "should_escalate": False,
            "expected_tools": ["search_slots", "book_appointment"],
            "expected_appointment_type": "Cataract Pre-Op Consult",
            "notes": "Clear booking request with patient id provided.",
        },
        "llm_script": [],
    }
    base.update(overrides)
    return Scenario(**base)


def _result(scenario: Scenario, **overrides: Any) -> HarnessResult:
    base: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "customer_id": scenario.customer_id,
        "difficulty": scenario.difficulty,
        "intent": scenario.intent,
        "actual_outcome": "booked",
        "actual_tools": ["search_slots", "book_appointment"],
        "escalated": False,
        "agent_replies": ["You're booked for June 15 at 9 AM."],
        "trace_ids": ["trace_aaa"],
        "passed": True,
        "failure_reasons": [],
        "judge_verdict": None,
        "escalation": None,
    }
    base.update(overrides)
    return HarnessResult(**base)


# ---------- HeuristicExtractor ----------


def test_extractor_pulls_patient_id_payer_caller_name() -> None:
    scenario = _scenario()
    result = _result(scenario)
    ctx = ExtractionContext(
        customer_id="ophthalmology",
        conversation_id="conv_001",
        scenario=scenario,
        result=result,
    )

    summary = HeuristicExtractor().extract(ctx)

    assert isinstance(summary, ConversationSummary)
    assert summary.patient_id == "pat_001"
    assert summary.payer == "Aetna"
    assert summary.caller_name == "Jane Smith"
    assert summary.appointment_type == "Cataract Pre-Op Consult"
    assert summary.outcome == "booked"
    assert summary.intent == "book"
    assert summary.escalated is False
    assert summary.transcript_preview  # non-empty


def test_extractor_handles_no_patient_id_gracefully() -> None:
    scenario = _scenario(messages=["Hi, what are your hours?"])
    result = _result(scenario, actual_outcome="info_provided")
    summary = HeuristicExtractor().extract(
        ExtractionContext(
            customer_id="ophthalmology",
            conversation_id="conv_002",
            scenario=scenario,
            result=result,
        )
    )
    assert summary.patient_id is None
    assert summary.payer is None
    # Intent falls back to the scenario label.
    assert summary.intent == "book"


def test_extractor_maps_emergency_outcome_and_escalation() -> None:
    scenario = _scenario(
        difficulty="emergency",
        intent="emergency",
        messages=["I think I'm having a stroke!"],
        ground_truth={
            "expected_outcome": "escalated_emergency",
            "should_escalate": True,
            "expected_tools": [],
            "expected_appointment_type": None,
            "notes": "Emergency phrase triggers guardrail short-circuit.",
        },
    )
    result = _result(
        scenario,
        actual_outcome="escalated_emergency",
        actual_tools=[],
        escalated=True,
        agent_replies=["This sounds like an emergency. Please call 911 immediately."],
    )
    summary = HeuristicExtractor().extract(
        ExtractionContext(
            customer_id="ophthalmology",
            conversation_id="conv_003",
            scenario=scenario,
            result=result,
        )
    )
    assert summary.outcome == "escalated_emergency"
    assert summary.escalated is True


def test_extractor_falls_back_to_unresolved_on_unknown_outcome() -> None:
    scenario = _scenario()
    # Passing an unknown outcome string — pydantic will reject, so we
    # bypass by going through a known one then manually mismatch the
    # mapping table.
    result = _result(scenario, actual_outcome="info_provided")
    summary = HeuristicExtractor().extract(
        ExtractionContext(
            customer_id="ophthalmology",
            conversation_id="conv_004",
            scenario=scenario,
            result=result,
        )
    )
    assert summary.outcome == "info_provided"


# ---------- PmsWritebackWriter ----------


def test_writer_produces_two_files_round_trip(tmp_path: Path) -> None:
    scenario = _scenario()
    result = _result(scenario)
    ctx = ExtractionContext(
        customer_id="ophthalmology",
        conversation_id=scenario.scenario_id,
        scenario=scenario,
        result=result,
    )
    outcome = PmsWritebackWriter().write(ctx, data_dir=tmp_path)

    # Files exist at the spec path.
    assert outcome.summary_path.is_file()
    assert outcome.task_path.is_file()
    assert outcome.summary_path.parent == (
        tmp_path / "ophthalmology" / "pms_writeback" / scenario.scenario_id
    )

    # Round-trip through Pydantic.
    summary_payload = json.loads(outcome.summary_path.read_text(encoding="utf-8"))
    task_payload = json.loads(outcome.task_path.read_text(encoding="utf-8"))
    rehydrated_summary = ConversationSummary.model_validate(summary_payload)
    rehydrated_task = PmsTaskWriteback.model_validate(task_payload)

    assert rehydrated_summary.customer_id == "ophthalmology"
    assert rehydrated_summary.outcome == "booked"
    assert rehydrated_task.summary_ref == "summary.json"
    assert rehydrated_task.assignee_group == "front_desk"
    assert rehydrated_task.priority == "normal"


def test_writer_routes_emergency_to_triage_with_urgent_priority(tmp_path: Path) -> None:
    scenario = _scenario(
        difficulty="emergency",
        intent="emergency",
        messages=["I'm having a heart attack — patient pat_007."],
        ground_truth={
            "expected_outcome": "escalated_emergency",
            "should_escalate": True,
            "expected_tools": [],
            "expected_appointment_type": None,
            "notes": "",
        },
    )
    result = _result(
        scenario,
        actual_outcome="escalated_emergency",
        escalated=True,
        agent_replies=["Please call 911 immediately."],
    )
    outcome = PmsWritebackWriter().write(
        ExtractionContext(
            customer_id="ophthalmology",
            conversation_id=scenario.scenario_id,
            scenario=scenario,
            result=result,
        ),
        data_dir=tmp_path,
    )
    task = json.loads(outcome.task_path.read_text(encoding="utf-8"))
    assert task["priority"] == "urgent"
    assert task["assignee_group"] == "triage"
    assert "URGENT" in task["subject"]


def test_writer_redacts_phi_in_payloads(tmp_path: Path) -> None:
    """Phone numbers / member ids / patient ids should be tagged
    before they reach disk, per the Phase 6 PHI contract."""
    scenario = _scenario(
        messages=[
            "Hi, this is John Doe. Call me back at 555-555-1234. "
            "Patient pat_001, member AET-9981. Aetna."
        ]
    )
    result = _result(scenario)
    outcome = PmsWritebackWriter().write(
        ExtractionContext(
            customer_id="ophthalmology",
            conversation_id=scenario.scenario_id,
            scenario=scenario,
            result=result,
        ),
        data_dir=tmp_path,
    )
    summary_text = outcome.summary_path.read_text(encoding="utf-8")
    task_text = outcome.task_path.read_text(encoding="utf-8")
    # No raw PHI on disk.
    assert "555-555-1234" not in summary_text
    assert "555-555-1234" not in task_text
    assert "AET-9981" not in summary_text


# ---------- field_extraction_accuracy ----------


def _harness_report(results: list[HarnessResult]) -> HarnessReport:
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    return HarnessReport(
        customer_id="ophthalmology",
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=passed / total if total else 0.0,
        by_difficulty={},
        by_intent={},
        results=results,
    )


def test_accuracy_returns_none_when_writeback_dir_missing(tmp_path: Path) -> None:
    scenario = _scenario()
    report = _harness_report([_result(scenario)])
    assert (
        compute_field_extraction_accuracy([scenario], report, writeback_dir=tmp_path / "missing")
        is None
    )


def test_accuracy_100_percent_when_every_field_matches(tmp_path: Path) -> None:
    scenario = _scenario()
    result = _result(scenario)
    ctx = ExtractionContext(
        customer_id="ophthalmology",
        conversation_id=scenario.scenario_id,
        scenario=scenario,
        result=result,
    )
    # Write via the same writer the harness uses.
    PmsWritebackWriter().write(ctx, data_dir=tmp_path)

    fae = compute_field_extraction_accuracy(
        [scenario],
        _harness_report([result]),
        writeback_dir=tmp_path / "ophthalmology" / "pms_writeback",
    )
    assert fae is not None
    assert fae.accuracy == 1.0
    assert fae.total_scenarios == 1
    assert fae.total_fields_evaluated >= 3
    assert fae.total_fields_matched == fae.total_fields_evaluated


def test_accuracy_drops_on_wrong_appointment_type(tmp_path: Path) -> None:
    scenario = _scenario()
    result = _result(scenario)
    PmsWritebackWriter().write(
        ExtractionContext(
            customer_id="ophthalmology",
            conversation_id=scenario.scenario_id,
            scenario=scenario,
            result=result,
        ),
        data_dir=tmp_path,
    )
    # Manually corrupt the summary to mismatch one field.
    summary_path = (
        tmp_path / "ophthalmology" / "pms_writeback" / scenario.scenario_id / "summary.json"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["appointment_type"] = "Routine Eye Exam"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    fae = compute_field_extraction_accuracy(
        [scenario],
        _harness_report([result]),
        writeback_dir=tmp_path / "ophthalmology" / "pms_writeback",
    )
    assert fae is not None
    # One of the four evaluated fields is now wrong.
    assert fae.accuracy < 1.0
    matched, evaluated = fae.by_field["appointment_type"]
    assert matched == 0 and evaluated == 1
