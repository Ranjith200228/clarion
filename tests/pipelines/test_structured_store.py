"""Unit tests for the SQLite structured store.

Every test gets a fresh sqlite file under tmp_path. No fixtures leak.
"""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import pytest
from clarion.pipelines.structured.store import (
    SlotAlreadyBookedError,
    StructuredStore,
)
from clarion.schemas import (
    AvailabilitySlot,
    EligibilityRecord,
    Provider,
)


@pytest.fixture
def store(tmp_path: Path) -> StructuredStore:
    return StructuredStore(tmp_path / "db.sqlite")


def _make_provider(provider_id: str = "prov_demo") -> Provider:
    return Provider(
        provider_id=provider_id,
        full_name="Dr. Demo",
        specialties=["Consult"],
        location="Main",
    )


def _make_slot(
    slot_id: str = "slot_demo",
    provider_id: str = "prov_demo",
    appointment_type: str = "Consult",
    slot_date: date = date(2026, 6, 15),
    start_time: time = time(9, 0),
) -> AvailabilitySlot:
    return AvailabilitySlot(
        slot_id=slot_id,
        provider_id=provider_id,
        appointment_type=appointment_type,
        slot_date=slot_date,
        start_time=start_time,
        duration_minutes=30,
    )


# ---------- schema bootstrap ----------


def test_store_creates_schema_on_init(tmp_path: Path) -> None:
    db = tmp_path / "new.sqlite"
    assert not db.exists()
    StructuredStore(db)
    assert db.exists()
    # Re-init is idempotent.
    StructuredStore(db)


def test_for_customer_path_convention(tmp_path: Path) -> None:
    store = StructuredStore.for_customer("demo", tmp_path)
    assert store.db_path == tmp_path / "demo" / "structured.sqlite"
    assert store.db_path.exists()


# ---------- providers ----------


def test_upsert_and_list_providers(store: StructuredStore) -> None:
    store.upsert_provider(_make_provider("p1"))
    store.upsert_provider(_make_provider("p2"))
    out = store.list_providers()
    assert [p.provider_id for p in out] == ["p1", "p2"]


def test_upsert_provider_idempotent(store: StructuredStore) -> None:
    store.upsert_provider(_make_provider("p1"))
    store.upsert_provider(
        Provider(
            provider_id="p1",
            full_name="Updated",
            specialties=["NewSpec"],
            location="Other",
            accepts_new_patients=False,
        )
    )
    out = store.list_providers()
    assert len(out) == 1
    assert out[0].full_name == "Updated"
    assert out[0].specialties == ["NewSpec"]
    assert out[0].accepts_new_patients is False


# ---------- slot search ----------


def test_search_slots_returns_only_open_and_in_range(store: StructuredStore) -> None:
    store.upsert_provider(_make_provider())
    store.upsert_slot(_make_slot("s_open", slot_date=date(2026, 6, 15)))
    store.upsert_slot(_make_slot("s_past", slot_date=date(2026, 5, 1)))
    booked = _make_slot("s_booked", slot_date=date(2026, 6, 15), start_time=time(10, 0))
    booked = booked.model_copy(update={"is_booked": True})
    store.upsert_slot(booked)

    out = store.search_slots("Consult", on_or_after=date(2026, 6, 1))
    assert [s.slot_id for s in out] == ["s_open"]


def test_search_slots_filters_by_provider(store: StructuredStore) -> None:
    store.upsert_provider(_make_provider("p1"))
    store.upsert_provider(_make_provider("p2"))
    store.upsert_slot(_make_slot("s1", provider_id="p1"))
    store.upsert_slot(
        _make_slot("s2", provider_id="p2", start_time=time(10, 0)),
    )

    out = store.search_slots("Consult", on_or_after=date(2026, 6, 1), provider_id="p2")
    assert [s.slot_id for s in out] == ["s2"]


def test_search_slots_orders_by_date_then_time(store: StructuredStore) -> None:
    store.upsert_provider(_make_provider())
    store.upsert_slot(_make_slot("late", slot_date=date(2026, 6, 16)))
    store.upsert_slot(_make_slot("early", slot_date=date(2026, 6, 15), start_time=time(8, 0)))
    store.upsert_slot(_make_slot("mid", slot_date=date(2026, 6, 15), start_time=time(11, 0)))

    out = store.search_slots("Consult", on_or_after=date(2026, 6, 1))
    assert [s.slot_id for s in out] == ["early", "mid", "late"]


# ---------- booking ----------


def test_book_slot_creates_appointment_and_flips_flag(store: StructuredStore) -> None:
    store.upsert_provider(_make_provider())
    store.upsert_slot(_make_slot("s1"))

    appt = store.book_slot("s1", patient_id="pat_42", notes="callback ok")
    assert appt.slot_id == "s1"
    assert appt.patient_id == "pat_42"
    assert appt.status == "booked"
    assert appt.notes == "callback ok"

    # That slot is no longer searchable.
    assert store.search_slots("Consult", on_or_after=date(2026, 6, 1)) == []

    # And we can read the appointment back.
    again = store.get_appointment(appt.appointment_id)
    assert again is not None
    assert again.appointment_id == appt.appointment_id


def test_book_slot_raises_on_double_book(store: StructuredStore) -> None:
    store.upsert_provider(_make_provider())
    store.upsert_slot(_make_slot("s1"))
    store.book_slot("s1", patient_id="pat_a")

    with pytest.raises(SlotAlreadyBookedError):
        store.book_slot("s1", patient_id="pat_b")


def test_book_slot_raises_on_unknown_slot(store: StructuredStore) -> None:
    with pytest.raises(SlotAlreadyBookedError):
        store.book_slot("ghost", patient_id="pat_a")


def test_cancel_appointment_frees_slot(store: StructuredStore) -> None:
    store.upsert_provider(_make_provider())
    store.upsert_slot(_make_slot("s1"))
    appt = store.book_slot("s1", patient_id="pat_a")

    changed = store.cancel_appointment(appt.appointment_id)
    assert changed is True

    # Slot is bookable again.
    out = store.search_slots("Consult", on_or_after=date(2026, 6, 1))
    assert [s.slot_id for s in out] == ["s1"]

    # Status reads back as cancelled.
    again = store.get_appointment(appt.appointment_id)
    assert again is not None
    assert again.status == "cancelled"


def test_cancel_appointment_idempotent(store: StructuredStore) -> None:
    store.upsert_provider(_make_provider())
    store.upsert_slot(_make_slot("s1"))
    appt = store.book_slot("s1", patient_id="pat_a")

    assert store.cancel_appointment(appt.appointment_id) is True
    assert store.cancel_appointment(appt.appointment_id) is False


def test_cancel_appointment_unknown_returns_false(store: StructuredStore) -> None:
    assert store.cancel_appointment("ghost") is False


# ---------- eligibility ----------


def test_upsert_and_get_eligibility(store: StructuredStore) -> None:
    rec = EligibilityRecord(
        patient_id="pat_1",
        payer="Aetna",
        member_id="AET-1",
        status="active",
        plan_name="PPO",
        effective_date=date(2026, 1, 1),
    )
    store.upsert_eligibility(rec)

    out = store.get_eligibility("pat_1")
    assert out == rec


def test_get_eligibility_missing_returns_none(store: StructuredStore) -> None:
    assert store.get_eligibility("ghost") is None


# ---------- per-customer isolation ----------


def test_two_customers_dont_share_state(tmp_path: Path) -> None:
    a = StructuredStore.for_customer("alpha", tmp_path)
    b = StructuredStore.for_customer("beta", tmp_path)

    a.upsert_provider(_make_provider("only_in_alpha"))
    assert [p.provider_id for p in a.list_providers()] == ["only_in_alpha"]
    assert b.list_providers() == []


# ---------- pms tasks ----------


def test_create_task_round_trips(store: StructuredStore) -> None:
    task = store.create_task(
        subject="Verify payer",
        body="Aetna eligibility unclear — call patient back.",
        patient_id="pat_demo",
        priority="normal",
    )
    assert task.task_id.startswith("task_")
    assert task.status == "open"
    again = store.get_task(task.task_id)
    assert again is not None
    assert again == task


def test_create_task_without_patient_id(store: StructuredStore) -> None:
    task = store.create_task(subject="General", body="Front-desk follow-up.")
    assert task.patient_id is None
    assert store.get_task(task.task_id) == task


def test_list_open_tasks_orders_by_created_at(store: StructuredStore) -> None:
    a = store.create_task(subject="A", body="first")
    b = store.create_task(subject="B", body="second", priority="urgent")
    open_tasks = store.list_open_tasks()
    assert [t.task_id for t in open_tasks] == [a.task_id, b.task_id]


def test_list_open_tasks_filters_by_priority(store: StructuredStore) -> None:
    store.create_task(subject="N", body="x", priority="normal")
    u = store.create_task(subject="U", body="y", priority="urgent")
    urgent = store.list_open_tasks(priority="urgent")
    assert [t.task_id for t in urgent] == [u.task_id]


def test_get_unknown_task_returns_none(store: StructuredStore) -> None:
    assert store.get_task("ghost") is None
