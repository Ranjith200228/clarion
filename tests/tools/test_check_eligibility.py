"""Tests for the check_eligibility tool."""

from __future__ import annotations

import pytest
from clarion.schemas.tools import CheckEligibilityInput
from clarion.tools.base import ToolContext
from clarion.tools.check_eligibility import CheckEligibilityTool
from pydantic import ValidationError


def test_returns_record_when_on_file(ctx: ToolContext) -> None:
    out = CheckEligibilityTool().run(CheckEligibilityInput(patient_id="pat_demo"), ctx)
    assert out.ok is True
    assert out.on_file is True
    assert out.record is not None
    assert out.record.patient_id == "pat_demo"
    assert out.record.payer == "Aetna"
    assert out.record.status == "active"


def test_returns_on_file_false_for_unknown_patient(ctx: ToolContext) -> None:
    out = CheckEligibilityTool().run(CheckEligibilityInput(patient_id="ghost"), ctx)
    assert out.ok is True
    assert out.on_file is False
    assert out.record is None


def test_input_rejects_empty_patient_id() -> None:
    with pytest.raises(ValidationError):
        CheckEligibilityInput(patient_id="")


def test_input_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        CheckEligibilityInput(
            patient_id="pat_demo",
            secret="oops",  # type: ignore[call-arg]
        )
