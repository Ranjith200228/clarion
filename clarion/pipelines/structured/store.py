"""Thin SQLite wrapper for the structured pipeline.

``StructuredStore`` is the only thing tools (Phase 4) call into. It owns the
SQLite connection, applies the schema, and returns validated Pydantic
models — never raw rows.

One SQLite file per customer. Path resolution:

    <data_dir>/<customer_id>/structured.sqlite

so two customers can run side by side with no interference.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, time
from pathlib import Path

from clarion.schemas import (
    Appointment,
    AvailabilitySlot,
    EligibilityRecord,
    Provider,
)

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class SlotAlreadyBookedError(RuntimeError):
    """Raised when book_slot is called on a slot that's already booked."""


class StructuredStore:
    """SQLite-backed structured store, one instance per customer."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ---------- lifecycle ----------

    @classmethod
    def for_customer(cls, customer_id: str, data_dir: Path) -> StructuredStore:
        """Conventional path: ``<data_dir>/<customer_id>/structured.sqlite``."""
        return cls(data_dir / customer_id / "structured.sqlite")

    def _ensure_schema(self) -> None:
        ddl = _SCHEMA_PATH.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(ddl)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------- providers ----------

    def upsert_provider(self, p: Provider) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO providers
                    (provider_id, full_name, specialties, location, accepts_new_patients)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(provider_id) DO UPDATE SET
                    full_name = excluded.full_name,
                    specialties = excluded.specialties,
                    location = excluded.location,
                    accepts_new_patients = excluded.accepts_new_patients
                """,
                (
                    p.provider_id,
                    p.full_name,
                    json.dumps(p.specialties),
                    p.location,
                    int(p.accepts_new_patients),
                ),
            )

    def list_providers(self) -> list[Provider]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM providers ORDER BY provider_id").fetchall()
        return [_row_to_provider(r) for r in rows]

    # ---------- availability / slots ----------

    def upsert_slot(self, s: AvailabilitySlot) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO availability
                    (slot_id, provider_id, appointment_type, slot_date,
                     start_time, duration_minutes, is_booked)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slot_id) DO UPDATE SET
                    provider_id = excluded.provider_id,
                    appointment_type = excluded.appointment_type,
                    slot_date = excluded.slot_date,
                    start_time = excluded.start_time,
                    duration_minutes = excluded.duration_minutes,
                    is_booked = excluded.is_booked
                """,
                (
                    s.slot_id,
                    s.provider_id,
                    s.appointment_type,
                    s.slot_date.isoformat(),
                    s.start_time.isoformat(),
                    s.duration_minutes,
                    int(s.is_booked),
                ),
            )

    def search_slots(
        self,
        appointment_type: str,
        on_or_after: date,
        *,
        provider_id: str | None = None,
        limit: int = 5,
    ) -> list[AvailabilitySlot]:
        """Return upcoming open slots matching the filter."""
        params: list[object] = [appointment_type, on_or_after.isoformat()]
        sql = (
            "SELECT * FROM availability "
            "WHERE appointment_type = ? AND slot_date >= ? AND is_booked = 0"
        )
        if provider_id is not None:
            sql += " AND provider_id = ?"
            params.append(provider_id)
        sql += " ORDER BY slot_date, start_time LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_slot(r) for r in rows]

    # ---------- appointments ----------

    def book_slot(
        self,
        slot_id: str,
        patient_id: str,
        *,
        notes: str | None = None,
    ) -> Appointment:
        """Atomically mark a slot booked and create the appointment row.

        Raises:
            SlotAlreadyBookedError: slot doesn't exist or is already taken.
        """
        with self._connect() as conn:
            slot_row = conn.execute(
                "SELECT * FROM availability WHERE slot_id = ?", (slot_id,)
            ).fetchone()
            if slot_row is None or slot_row["is_booked"]:
                raise SlotAlreadyBookedError(f"Slot {slot_id} is unavailable")

            appointment_id = f"appt_{uuid.uuid4().hex[:12]}"
            starts_at = datetime.fromisoformat(f"{slot_row['slot_date']}T{slot_row['start_time']}")
            conn.execute(
                """
                INSERT INTO appointments
                    (appointment_id, slot_id, patient_id, provider_id,
                     appointment_type, starts_at, duration_minutes, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'booked', ?)
                """,
                (
                    appointment_id,
                    slot_id,
                    patient_id,
                    slot_row["provider_id"],
                    slot_row["appointment_type"],
                    starts_at.isoformat(),
                    slot_row["duration_minutes"],
                    notes,
                ),
            )
            conn.execute("UPDATE availability SET is_booked = 1 WHERE slot_id = ?", (slot_id,))

        return Appointment(
            appointment_id=appointment_id,
            slot_id=slot_id,
            patient_id=patient_id,
            provider_id=slot_row["provider_id"],
            appointment_type=slot_row["appointment_type"],
            starts_at=starts_at,
            duration_minutes=slot_row["duration_minutes"],
            status="booked",
            notes=notes,
        )

    def cancel_appointment(self, appointment_id: str) -> bool:
        """Cancel an appointment and free its slot. Returns True if anything changed."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT slot_id, status FROM appointments WHERE appointment_id = ?",
                (appointment_id,),
            ).fetchone()
            if row is None or row["status"] == "cancelled":
                return False
            conn.execute(
                "UPDATE appointments SET status = 'cancelled' WHERE appointment_id = ?",
                (appointment_id,),
            )
            conn.execute(
                "UPDATE availability SET is_booked = 0 WHERE slot_id = ?",
                (row["slot_id"],),
            )
        return True

    def get_appointment(self, appointment_id: str) -> Appointment | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
            ).fetchone()
        return _row_to_appointment(row) if row else None

    # ---------- eligibility ----------

    def upsert_eligibility(self, e: EligibilityRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO eligibility
                    (patient_id, payer, member_id, status, plan_name,
                     effective_date, termination_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(patient_id) DO UPDATE SET
                    payer = excluded.payer,
                    member_id = excluded.member_id,
                    status = excluded.status,
                    plan_name = excluded.plan_name,
                    effective_date = excluded.effective_date,
                    termination_date = excluded.termination_date
                """,
                (
                    e.patient_id,
                    e.payer,
                    e.member_id,
                    e.status,
                    e.plan_name,
                    e.effective_date.isoformat() if e.effective_date else None,
                    e.termination_date.isoformat() if e.termination_date else None,
                ),
            )

    def get_eligibility(self, patient_id: str) -> EligibilityRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM eligibility WHERE patient_id = ?", (patient_id,)
            ).fetchone()
        return _row_to_eligibility(row) if row else None


# ---------- row → model adapters ----------


def _row_to_provider(r: sqlite3.Row) -> Provider:
    return Provider(
        provider_id=r["provider_id"],
        full_name=r["full_name"],
        specialties=json.loads(r["specialties"]),
        location=r["location"],
        accepts_new_patients=bool(r["accepts_new_patients"]),
    )


def _row_to_slot(r: sqlite3.Row) -> AvailabilitySlot:
    return AvailabilitySlot(
        slot_id=r["slot_id"],
        provider_id=r["provider_id"],
        appointment_type=r["appointment_type"],
        slot_date=date.fromisoformat(r["slot_date"]),
        start_time=time.fromisoformat(r["start_time"]),
        duration_minutes=r["duration_minutes"],
        is_booked=bool(r["is_booked"]),
    )


def _row_to_appointment(r: sqlite3.Row) -> Appointment:
    return Appointment(
        appointment_id=r["appointment_id"],
        slot_id=r["slot_id"],
        patient_id=r["patient_id"],
        provider_id=r["provider_id"],
        appointment_type=r["appointment_type"],
        starts_at=datetime.fromisoformat(r["starts_at"]),
        duration_minutes=r["duration_minutes"],
        status=r["status"],
        notes=r["notes"],
    )


def _row_to_eligibility(r: sqlite3.Row) -> EligibilityRecord:
    return EligibilityRecord(
        patient_id=r["patient_id"],
        payer=r["payer"],
        member_id=r["member_id"],
        status=r["status"],
        plan_name=r["plan_name"],
        effective_date=date.fromisoformat(r["effective_date"]) if r["effective_date"] else None,
        termination_date=(
            date.fromisoformat(r["termination_date"]) if r["termination_date"] else None
        ),
    )
