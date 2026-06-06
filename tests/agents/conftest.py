"""Shared fixtures for agent tests."""

from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path

import pytest
from clarion.config import CustomerConfig, EscalationThresholds, Settings, load_customer
from clarion.pipelines.structured import StructuredStore
from clarion.schemas import AvailabilitySlot, EligibilityRecord, Provider
from clarion.tools.base import ToolContext

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"


def _real_settings(customer: str) -> Settings:
    return Settings(customer=customer, config_dir=CONFIGS_DIR, data_dir=DATA_DIR)


@pytest.fixture
def ophthalmology_config() -> CustomerConfig:
    return load_customer("ophthalmology", settings=_real_settings("ophthalmology"))


@pytest.fixture
def orthopedics_config() -> CustomerConfig:
    return load_customer("orthopedics", settings=_real_settings("orthopedics"))


@pytest.fixture
def minimal_config(tmp_path: Path) -> CustomerConfig:
    return CustomerConfig(
        customer_id="demo",
        display_name="Demo Clinic",
        specialties=["Consult"],
        enabled_tools=[
            "search_slots",
            "book_appointment",
            "cancel_appointment",
            "check_eligibility",
            "create_pms_task",
        ],
        escalation=EscalationThresholds(),
        languages=["en"],
        rules_path=tmp_path / "rules",
        agent_persona="You are Clarion, a friendly demo assistant.",
    )


@pytest.fixture
def store_with_ophthalmology_seed(
    tmp_path: Path, ophthalmology_config: CustomerConfig
) -> StructuredStore:
    """A fresh sqlite seeded from the real ophthalmology seed JSON."""
    s = StructuredStore(tmp_path / "oph.sqlite")
    payload = json.loads((DATA_DIR / "seeds" / "ophthalmology.json").read_text(encoding="utf-8"))
    for p in payload["providers"]:
        s.upsert_provider(Provider(**p))
    for slot in payload["availability"]:
        s.upsert_slot(AvailabilitySlot(**slot))
    for e in payload["eligibility"]:
        s.upsert_eligibility(EligibilityRecord(**e))
    return s


@pytest.fixture
def store_with_orthopedics_seed(
    tmp_path: Path, orthopedics_config: CustomerConfig
) -> StructuredStore:
    s = StructuredStore(tmp_path / "ortho.sqlite")
    payload = json.loads((DATA_DIR / "seeds" / "orthopedics.json").read_text(encoding="utf-8"))
    for p in payload["providers"]:
        s.upsert_provider(Provider(**p))
    for slot in payload["availability"]:
        s.upsert_slot(AvailabilitySlot(**slot))
    for e in payload["eligibility"]:
        s.upsert_eligibility(EligibilityRecord(**e))
    return s


@pytest.fixture
def minimal_store(tmp_path: Path) -> StructuredStore:
    """Hand-rolled tiny store for unit tests that don't want the full seed."""
    s = StructuredStore(tmp_path / "min.sqlite")
    s.upsert_provider(
        Provider(
            provider_id="prov_demo",
            full_name="Dr. Demo",
            specialties=["Consult"],
            location="Main",
        )
    )
    s.upsert_slot(
        AvailabilitySlot(
            slot_id="slot_demo_1",
            provider_id="prov_demo",
            appointment_type="Consult",
            slot_date=date(2026, 6, 15),
            start_time=time(9, 0),
            duration_minutes=30,
        )
    )
    s.upsert_eligibility(
        EligibilityRecord(
            patient_id="pat_demo",
            payer="Aetna",
            member_id="A1",
            status="active",
            plan_name="PPO",
            effective_date=date(2026, 1, 1),
        )
    )
    return s


@pytest.fixture
def minimal_ctx(minimal_config: CustomerConfig, minimal_store: StructuredStore) -> ToolContext:
    return ToolContext(customer=minimal_config, structured=minimal_store)
