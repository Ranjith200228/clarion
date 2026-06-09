"""Phase 11 acceptance: precision/recall metrics generated.

Spec: *"Output: 0-1 escalation score. Acceptance: Precision/recall
metrics generated."*

This test runs the full 100-scenario harness for each shipped customer
in scripted mode, feeds the resulting predictions through
``stats_from_run``, and asserts:

1. The function returns a valid ``EscalationStats`` (precision, recall,
   F1, accuracy, plus the confusion matrix).
2. Predictions cover every scenario (no None escalation fields).
3. Recall on emergency scenarios is 100% — the agent must catch every
   one, by construction of the guardrail-driven outcome path.
4. Overall precision is non-trivial (>= 0.5) so the test would catch a
   regression that turned the scorer into a "predict True always"
   degenerate.

This file replaces the manual "are P/R numbers being produced?" check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from clarion.config import CustomerConfig, Settings, load_customer
from clarion.pipelines.structured import StructuredStore
from clarion.schemas import (
    AvailabilitySlot,
    EligibilityRecord,
    EscalationStats,
    Provider,
)
from clarion.sentinel.escalation import stats_from_run
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
def test_stats_from_run_produces_full_metric_suite(customer_id: str, tmp_path: Path) -> None:
    """Phase 11 acceptance: stats_from_run returns a populated
    EscalationStats over the full 100-scenario set for each customer."""
    customer = _load_customer(customer_id)
    store = _seed_customer(customer_id, tmp_path)
    scenarios = load_scenarios(REPO_ROOT / "data" / "personas" / f"{customer_id}.json")

    report = run_scripted(
        scenarios,
        customer_config=customer,
        structured=store,
        retriever=None,
    )

    stats = stats_from_run(scenarios, report)

    # The Phase 11 acceptance line: P/R metrics are produced for the
    # escalation predictions.
    assert isinstance(stats, EscalationStats)
    assert stats.total == 100
    assert (
        stats.true_positives + stats.false_positives + stats.true_negatives + stats.false_negatives
        == 100
    )
    assert 0.0 <= stats.precision <= 1.0
    assert 0.0 <= stats.recall <= 1.0
    assert 0.0 <= stats.f1 <= 1.0
    assert 0.0 <= stats.accuracy <= 1.0


@pytest.mark.parametrize("customer_id", ["ophthalmology", "orthopedics"])
def test_emergency_scenarios_are_always_flagged_for_escalation(
    customer_id: str, tmp_path: Path
) -> None:
    """Emergency scenarios are the highest-stakes category — recall must
    be 100% on them. Any miss is a P11 blocker."""
    customer = _load_customer(customer_id)
    store = _seed_customer(customer_id, tmp_path)
    scenarios = load_scenarios(REPO_ROOT / "data" / "personas" / f"{customer_id}.json")
    emergencies = [s for s in scenarios if s.intent == "emergency"]
    assert emergencies, f"no emergency scenarios for {customer_id}?"

    report = run_scripted(
        scenarios,
        customer_config=customer,
        structured=store,
        retriever=None,
    )
    by_id = {r.scenario_id: r for r in report.results}

    misses = []
    for s in emergencies:
        result = by_id[s.scenario_id]
        assert result.escalation is not None
        if not result.escalation["should_escalate"]:
            misses.append(s.scenario_id)

    assert not misses, f"emergency scenarios not flagged: {misses}"


@pytest.mark.parametrize("customer_id", ["ophthalmology", "orthopedics"])
def test_overall_precision_is_non_trivial(customer_id: str, tmp_path: Path) -> None:
    """Guard against a degenerate scorer that flags everything True.
    On scripted-mode runs every scenario plays its expected behaviour,
    so precision should be solidly above 0.5 — a "predict True always"
    scorer would crash through this assertion because the scripted
    happy-path scenarios (clear booking, FAQ) are NOT expected to
    escalate."""
    customer = _load_customer(customer_id)
    store = _seed_customer(customer_id, tmp_path)
    scenarios = load_scenarios(REPO_ROOT / "data" / "personas" / f"{customer_id}.json")

    report = run_scripted(
        scenarios,
        customer_config=customer,
        structured=store,
        retriever=None,
    )
    stats = stats_from_run(scenarios, report)

    assert stats.precision >= 0.5, (
        f"precision {stats.precision} is too low — scorer may be flagging "
        f"clean traffic as escalation. confusion matrix: "
        f"TP={stats.true_positives} FP={stats.false_positives} "
        f"TN={stats.true_negatives} FN={stats.false_negatives}"
    )
