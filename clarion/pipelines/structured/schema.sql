-- Clarion structured pipeline — SQLite schema.
--
-- One SQLite file per customer (data/<customer>/structured.sqlite). The
-- agent never sees raw SQL; it calls StructuredStore methods, which return
-- validated Pydantic models from clarion.schemas.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS providers (
    provider_id          TEXT PRIMARY KEY,
    full_name            TEXT NOT NULL,
    specialties          TEXT NOT NULL,           -- JSON array
    location             TEXT NOT NULL,
    accepts_new_patients INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS availability (
    slot_id          TEXT PRIMARY KEY,
    provider_id      TEXT NOT NULL REFERENCES providers(provider_id),
    appointment_type TEXT NOT NULL,
    slot_date        TEXT NOT NULL,               -- ISO YYYY-MM-DD
    start_time       TEXT NOT NULL,               -- ISO HH:MM:SS
    duration_minutes INTEGER NOT NULL,
    is_booked        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_availability_search
    ON availability (appointment_type, slot_date, is_booked);

CREATE TABLE IF NOT EXISTS appointments (
    appointment_id   TEXT PRIMARY KEY,
    slot_id          TEXT NOT NULL UNIQUE REFERENCES availability(slot_id),
    patient_id       TEXT NOT NULL,
    provider_id      TEXT NOT NULL REFERENCES providers(provider_id),
    appointment_type TEXT NOT NULL,
    starts_at        TEXT NOT NULL,               -- ISO datetime
    duration_minutes INTEGER NOT NULL,
    status           TEXT NOT NULL DEFAULT 'booked',
    notes            TEXT
);

CREATE INDEX IF NOT EXISTS idx_appointments_patient
    ON appointments (patient_id, status);

CREATE TABLE IF NOT EXISTS eligibility (
    patient_id       TEXT PRIMARY KEY,
    payer            TEXT NOT NULL,
    member_id        TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'unknown',
    plan_name        TEXT,
    effective_date   TEXT,                        -- ISO YYYY-MM-DD
    termination_date TEXT                         -- ISO YYYY-MM-DD
);
