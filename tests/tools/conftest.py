"""Shared fixtures for tool tests."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import pytest
from clarion.config import CustomerConfig, EscalationThresholds
from clarion.pipelines.structured import StructuredStore
from clarion.schemas import (
    AvailabilitySlot,
    EligibilityRecord,
    Provider,
)
from clarion.tools.base import ToolContext


@pytest.fixture
def store(tmp_path: Path) -> StructuredStore:
    s = StructuredStore(tmp_path / "test.sqlite")
    # One provider + two slots + one eligibility record is enough for every
    # tool test to have something to bind to.
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
    s.upsert_slot(
        AvailabilitySlot(
            slot_id="slot_demo_2",
            provider_id="prov_demo",
            appointment_type="Consult",
            slot_date=date(2026, 6, 16),
            start_time=time(10, 0),
            duration_minutes=30,
        )
    )
    s.upsert_eligibility(
        EligibilityRecord(
            patient_id="pat_demo",
            payer="Aetna",
            member_id="AET-1",
            status="active",
            plan_name="PPO",
            effective_date=date(2026, 1, 1),
        )
    )
    return s


@pytest.fixture
def customer_config(tmp_path: Path) -> CustomerConfig:
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
        agent_persona="You are Clarion, a demo assistant.",
    )


@pytest.fixture
def ctx(customer_config: CustomerConfig, store: StructuredStore) -> ToolContext:
    return ToolContext(customer=customer_config, structured=store)
