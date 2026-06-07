"""End-to-end harness tests on both customer personas."""

from __future__ import annotations

import json
from pathlib import Path

from clarion.config import Settings, load_customer
from clarion.pipelines.structured import StructuredStore
from clarion.schemas import AvailabilitySlot, EligibilityRecord, Provider
from clarion.simulator.harness import (
    load_scenarios,
    run_scripted,
)

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


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        customer="ophthalmology",
        config_dir=CONFIGS_DIR,
        data_dir=tmp_path,
    )


def test_scripted_harness_passes_all_ophthalmology(tmp_path: Path) -> None:
    customer = load_customer("ophthalmology", settings=_settings(tmp_path))
    store = StructuredStore.for_customer("ophthalmology", tmp_path)
    _seed(store, DATA_DIR / "seeds" / "ophthalmology.json")
    scenarios = load_scenarios(DATA_DIR / "personas" / "ophthalmology.json")

    report = run_scripted(
        scenarios,
        customer_config=customer,
        structured=store,
        retriever=None,
    )

    assert report.total == 100
    assert report.failed == 0, [
        (r.scenario_id, r.failure_reasons) for r in report.results if not r.passed
    ][:5]
    assert report.pass_rate == 1.0


def test_scripted_harness_passes_all_orthopedics(tmp_path: Path) -> None:
    customer = load_customer(
        "orthopedics",
        settings=Settings(
            customer="orthopedics",
            config_dir=CONFIGS_DIR,
            data_dir=tmp_path,
        ),
    )
    store = StructuredStore.for_customer("orthopedics", tmp_path)
    _seed(store, DATA_DIR / "seeds" / "orthopedics.json")
    scenarios = load_scenarios(DATA_DIR / "personas" / "orthopedics.json")

    report = run_scripted(
        scenarios,
        customer_config=customer,
        structured=store,
        retriever=None,
    )

    assert report.total == 100
    assert report.failed == 0, [
        (r.scenario_id, r.failure_reasons) for r in report.results if not r.passed
    ][:5]


def test_report_breakdowns_are_populated(tmp_path: Path) -> None:
    customer = load_customer("ophthalmology", settings=_settings(tmp_path))
    store = StructuredStore.for_customer("ophthalmology", tmp_path)
    _seed(store, DATA_DIR / "seeds" / "ophthalmology.json")
    # Just the first 10 scenarios for speed.
    scenarios = load_scenarios(DATA_DIR / "personas" / "ophthalmology.json")[:10]
    report = run_scripted(
        scenarios,
        customer_config=customer,
        structured=store,
        retriever=None,
    )
    assert sum(c["total"] for c in report.by_difficulty.values()) == 10
    assert sum(c["total"] for c in report.by_intent.values()) == 10


def test_harness_short_circuits_emergency_without_llm_calls(
    tmp_path: Path,
) -> None:
    """Emergency scenarios ship with llm_script=[] — the FakeLLM should
    never be consulted. We verify by running only the emergency subset
    and asserting they all pass + escalation flag is set."""
    customer = load_customer("ophthalmology", settings=_settings(tmp_path))
    store = StructuredStore.for_customer("ophthalmology", tmp_path)
    _seed(store, DATA_DIR / "seeds" / "ophthalmology.json")
    all_scenarios = load_scenarios(DATA_DIR / "personas" / "ophthalmology.json")
    emergencies = [s for s in all_scenarios if s.intent == "emergency"]
    assert len(emergencies) == 10

    report = run_scripted(
        emergencies,
        customer_config=customer,
        structured=store,
        retriever=None,
    )
    assert report.failed == 0
    for r in report.results:
        assert r.actual_outcome == "escalated_emergency"
        assert r.escalated is True
