"""Tests for the PHI redactor."""

from __future__ import annotations

import pytest
from clarion.sentinel.phi import redact, redact_with_counts

# ---------- single-pattern coverage ----------


@pytest.mark.parametrize(
    "raw, expected_tag",
    [
        ("call me at 555-0100", "<PHONE>"),
        ("phone (555) 555-1234 please", "<PHONE>"),
        ("+1 555.555.1234 works", "<PHONE>"),
        ("ssn is 123-45-6789", "<SSN>"),
        ("email john@example.com", "<EMAIL>"),
        ("member AET-9981 active", "<MEMBER_ID>"),
        ("member BCB-3320 active", "<MEMBER_ID>"),
        ("patient pat_001 calling", "<PATIENT_ID>"),
    ],
)
def test_each_pattern_is_redacted(raw: str, expected_tag: str) -> None:
    out = redact(raw)
    assert expected_tag in out
    # Something was changed.
    assert out != raw


# ---------- non-PHI is left alone ----------


def test_appointment_dates_pass_through_untouched() -> None:
    raw = "Booked you for 2026-06-15 at 9 AM with Dr. Smith."
    out = redact(raw)
    assert "2026-06-15" in out
    assert "Dr. Smith" in out


def test_short_numbers_are_not_phone_numbers() -> None:
    raw = "Your appointment lasts 60 minutes."
    out = redact(raw)
    assert "60 minutes" in out


def test_appointment_type_strings_dont_match_member_id() -> None:
    raw = "Glaucoma Follow-Up needs 20 minutes."
    out = redact(raw)
    assert "Glaucoma Follow-Up" in out
    assert "<MEMBER_ID>" not in out


# ---------- combined / counts ----------


def test_redacts_multiple_patterns_in_one_pass() -> None:
    raw = (
        "Hi, this is John for pat_001, member AET-9981, "
        "callback at 555-555-1234 or john@example.com."
    )
    out = redact(raw)
    assert "pat_001" not in out
    assert "AET-9981" not in out
    assert "555-555-1234" not in out
    assert "john@example.com" not in out
    assert "<PATIENT_ID>" in out
    assert "<MEMBER_ID>" in out
    assert "<PHONE>" in out
    assert "<EMAIL>" in out


def test_redact_with_counts_returns_hit_breakdown() -> None:
    raw = "calls: 555-0100 and 555-0200; members AET-9981 and BCB-3320"
    result = redact_with_counts(raw)
    assert result.tag_counts == {"<PHONE>": 2, "<MEMBER_ID>": 2}
    assert "555-0100" not in result.text
    assert "AET-9981" not in result.text


def test_redact_with_counts_empty_on_clean_text() -> None:
    result = redact_with_counts("Hi, I'd like to book an appointment.")
    assert result.tag_counts == {}
    assert result.text.startswith("Hi")
