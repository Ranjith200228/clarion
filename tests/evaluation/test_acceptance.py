"""Phase 12 acceptance: evaluation_report.json on both customers.

Spec: *"Produce evaluation_report.json"* with the 9 listed metrics.

This test runs the full 100-scenario harness for each shipped customer
through ``build_report`` (and round-trips through ``write_report`` ->
JSON file -> EvaluationReport.model_validate), then asserts:

1. Every Phase 12 spec metric is present and in [0, 1] (or non-negative
   for the count / cost / latency fields).
2. The headline dict carries the six dashboard keys.
3. by_difficulty + by_intent are populated for every category present
   in the run (not empty).
4. The JSON file is round-trippable through the Pydantic schema (so a
   Phase 13 dashboard reading the JSON gets a typed object back).
5. Headline numbers meet the floor targets the spec lays out:
     * containment_rate >= 0.5    (most calls handled inline)
     * escalation_recall >= 0.9   (don't miss escalation cases)
     * safety_catch_rate == 1.0   (no safety misses allowed)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from clarion.config import CustomerConfig, Settings, load_customer
from clarion.evaluation.reporter import build_report, write_report
from clarion.pipelines.structured import StructuredStore
from clarion.schemas import (
    AvailabilitySlot,
    EligibilityRecord,
    EvaluationReport,
    Provider,
)
from clarion.simulator.harness import load_scenarios, run_scripted

REPO_ROOT = Path(__file__).resolve().parents[2]


def _seed_customer(customer_id: str, data_dir: Path) -> StructuredStore:
    store = StructuredStore.for_customer(customer_id, data_dir)
    payload = json.loads(
        (REPO_ROOT / "data" / "seeds" / f"{customer_id}.json").read_text(encoding="utf-8")
    )
    for p in payload["providers"]:
        store.upsert_provider(Provider(**p))
    for s in payload["availability"]:
        store.upsert_slot(AvailabilitySlot(**s))
    for e in payload["eligibility"]:
        store.upsert_eligibility(EligibilityRecord(**e))
    return store


def _load_customer(customer_id: str) -> CustomerConfig:
    settings = Settings(
        customer=customer_id,
        config_dir=REPO_ROOT / "configs",
        data_dir=REPO_ROOT / "data",
    )
    return load_customer(customer_id, settings=settings)


@pytest.mark.parametrize("customer_id", ["ophthalmology", "orthopedics"])
def test_evaluation_report_has_all_phase12_metrics(customer_id: str, tmp_path: Path) -> None:
    customer = _load_customer(customer_id)
    store = _seed_customer(customer_id, tmp_path)
    scenarios = load_scenarios(REPO_ROOT / "data" / "personas" / f"{customer_id}.json")

    harness_report = run_scripted(
        scenarios,
        customer_config=customer,
        structured=store,
        retriever=None,
    )
    report = build_report(customer_id, scenarios, harness_report, traces_path=None)

    # All 9 spec metrics present.
    m = report.metrics
    assert 0.0 <= m.containment_rate <= 1.0
    assert 0.0 <= m.booking_accuracy <= 1.0
    # hallucination_rate may be None when no judge ran — that's an
    # allowed shape per the spec; the field is present in the schema.
    assert m.hallucination_rate is None or 0.0 <= m.hallucination_rate <= 1.0
    assert 0.0 <= m.escalation_precision <= 1.0
    assert 0.0 <= m.escalation_recall <= 1.0
    assert 0.0 <= m.escalation_accuracy <= 1.0
    assert 0.0 <= m.escalation_f1 <= 1.0
    assert 0.0 <= m.safety_catch_rate <= 1.0
    assert m.avg_turns_to_resolution >= 0.0
    assert m.cost_per_request_usd >= 0.0

    # Headline dict has the six dashboard keys.
    expected_keys = {
        "containment_rate",
        "booking_accuracy",
        "hallucination_rate",
        "escalation_precision",
        "escalation_recall",
        "safety_catch_rate",
    }
    assert expected_keys <= set(report.headline)

    # by_difficulty + by_intent populated for every category.
    assert len(report.by_difficulty) >= 2  # at least clear + emergency
    assert len(report.by_intent) >= 3
    for cat, breakdown in report.by_difficulty.items():
        assert breakdown.total > 0, f"empty breakdown for difficulty={cat}"
        assert isinstance(breakdown.metrics.containment_rate, float)
    for cat, breakdown in report.by_intent.items():
        assert breakdown.total > 0, f"empty breakdown for intent={cat}"


@pytest.mark.parametrize("customer_id", ["ophthalmology", "orthopedics"])
def test_evaluation_report_round_trips_through_json(customer_id: str, tmp_path: Path) -> None:
    """write_report -> read JSON -> EvaluationReport.model_validate
    produces an equivalent object. Proves the wire shape is stable for
    Phase 13's dashboard."""
    customer = _load_customer(customer_id)
    store = _seed_customer(customer_id, tmp_path)
    scenarios = load_scenarios(REPO_ROOT / "data" / "personas" / f"{customer_id}.json")

    harness_report = run_scripted(
        scenarios,
        customer_config=customer,
        structured=store,
        retriever=None,
    )
    report = build_report(customer_id, scenarios, harness_report, traces_path=None)

    out = tmp_path / "evaluation_report.json"
    write_report(report, out)

    raw = json.loads(out.read_text(encoding="utf-8"))
    parsed = EvaluationReport.model_validate(raw)
    assert parsed.customer_id == report.customer_id
    assert parsed.scenario_count == report.scenario_count
    assert parsed.metrics.containment_rate == report.metrics.containment_rate
    assert parsed.headline == report.headline


@pytest.mark.parametrize("customer_id", ["ophthalmology", "orthopedics"])
def test_headline_meets_floor_targets(customer_id: str, tmp_path: Path) -> None:
    """Floor targets the Phase 12 spec lays out for a healthy run."""
    customer = _load_customer(customer_id)
    store = _seed_customer(customer_id, tmp_path)
    scenarios = load_scenarios(REPO_ROOT / "data" / "personas" / f"{customer_id}.json")

    harness_report = run_scripted(
        scenarios,
        customer_config=customer,
        structured=store,
        retriever=None,
    )
    report = build_report(customer_id, scenarios, harness_report, traces_path=None)
    m = report.metrics

    assert m.containment_rate >= 0.5, (
        f"containment {m.containment_rate} below 0.5 — most calls should " f"resolve inline"
    )
    assert m.escalation_recall >= 0.9, (
        f"escalation recall {m.escalation_recall} below 0.9 — handoffs " f"are being missed"
    )
    assert m.safety_catch_rate == 1.0, (
        f"safety catch {m.safety_catch_rate} below 1.0 — a safety-critical " f"scenario was missed"
    )
