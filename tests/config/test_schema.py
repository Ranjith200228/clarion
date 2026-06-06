"""Schema-level tests for CustomerConfig.

These tests don't touch the filesystem — they build the model directly so
failures pinpoint a schema constraint, not a YAML issue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from clarion.config.schema import (
    CustomerConfig,
    EscalationThresholds,
)
from pydantic import ValidationError


def _valid_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "customer_id": "demo_clinic",
        "display_name": "Demo Clinic",
        "specialties": ["General Consult"],
        "enabled_tools": ["search_slots", "book_appointment"],
        "rules_path": Path("/tmp/rules/demo"),
        "agent_persona": "You are Clarion, a friendly demo assistant.",
    }
    base.update(overrides)
    return base


# ---------- happy path ----------


def test_minimal_valid_config() -> None:
    c = CustomerConfig(**_valid_payload())
    assert c.customer_id == "demo_clinic"
    assert c.vertical == "healthcare-scheduling"  # default
    assert c.languages == ["en"]  # default
    assert isinstance(c.escalation, EscalationThresholds)
    assert c.escalation.low_confidence == 0.6  # default


def test_languages_lowercased_and_deduped_when_unique() -> None:
    c = CustomerConfig(**_valid_payload(languages=["EN", "Es"]))
    assert c.languages == ["en", "es"]


# ---------- customer_id ----------


@pytest.mark.parametrize("bad_id", ["Has Space", "UPPER", "with.dot", ""])
def test_invalid_customer_id_rejected(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        CustomerConfig(**_valid_payload(customer_id=bad_id))


# ---------- enabled_tools ----------


def test_unknown_tool_rejected() -> None:
    with pytest.raises(ValidationError):
        CustomerConfig(**_valid_payload(enabled_tools=["search_slots", "delete_universe"]))


def test_duplicate_tools_rejected() -> None:
    with pytest.raises(ValidationError):
        CustomerConfig(**_valid_payload(enabled_tools=["search_slots", "search_slots"]))


def test_empty_tools_rejected() -> None:
    with pytest.raises(ValidationError):
        CustomerConfig(**_valid_payload(enabled_tools=[]))


# ---------- specialties ----------


def test_empty_specialties_rejected() -> None:
    with pytest.raises(ValidationError):
        CustomerConfig(**_valid_payload(specialties=[]))


# ---------- languages ----------


def test_duplicate_languages_rejected_case_insensitive() -> None:
    with pytest.raises(ValidationError):
        CustomerConfig(**_valid_payload(languages=["en", "EN"]))


def test_empty_languages_rejected() -> None:
    with pytest.raises(ValidationError):
        CustomerConfig(**_valid_payload(languages=[]))


# ---------- escalation thresholds ----------


@pytest.mark.parametrize(
    "field, value",
    [
        ("low_confidence", -0.1),
        ("low_confidence", 1.1),
        ("frustration", 2.0),
        ("max_clarifications", 0),
        ("max_clarifications", 21),
    ],
)
def test_escalation_out_of_range_rejected(field: str, value: float | int) -> None:
    with pytest.raises(ValidationError):
        EscalationThresholds(**{field: value})


def test_escalation_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        EscalationThresholds(low_confidence=0.6, unknown_field=42)  # type: ignore[call-arg]


# ---------- extra=forbid on the whole CustomerConfig ----------


def test_unknown_top_level_field_rejected() -> None:
    with pytest.raises(ValidationError):
        CustomerConfig(**_valid_payload(secret_backdoor="oops"))
