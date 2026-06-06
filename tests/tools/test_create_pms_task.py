"""Tests for the create_pms_task tool."""

from __future__ import annotations

import pytest
from clarion.schemas.tools import CreatePmsTaskInput
from clarion.tools.base import ToolContext
from clarion.tools.create_pms_task import CreatePmsTaskTool
from pydantic import ValidationError


def test_files_a_normal_task(ctx: ToolContext) -> None:
    out = CreatePmsTaskTool().run(
        CreatePmsTaskInput(
            subject="Verify payer",
            body="Eligibility unclear — call patient back.",
            patient_id="pat_demo",
        ),
        ctx,
    )
    assert out.ok is True
    assert out.task_id is not None
    assert out.task_id.startswith("task_")

    # Persisted: store can read it back.
    record = ctx.structured.get_task(out.task_id)
    assert record is not None
    assert record.subject == "Verify payer"
    assert record.priority == "normal"
    assert record.status == "open"


def test_files_an_urgent_task_without_patient(ctx: ToolContext) -> None:
    out = CreatePmsTaskTool().run(
        CreatePmsTaskInput(
            subject="Emergency caller — possible stroke",
            body="Caller advised to call 911. Follow up immediately.",
            priority="urgent",
        ),
        ctx,
    )
    assert out.ok is True
    assert out.task_id is not None

    record = ctx.structured.get_task(out.task_id)
    assert record is not None
    assert record.priority == "urgent"
    assert record.patient_id is None


def test_input_rejects_empty_subject_or_body() -> None:
    with pytest.raises(ValidationError):
        CreatePmsTaskInput(subject="", body="text")
    with pytest.raises(ValidationError):
        CreatePmsTaskInput(subject="ok", body="")


def test_input_rejects_overlong_fields() -> None:
    with pytest.raises(ValidationError):
        CreatePmsTaskInput(subject="x" * 201, body="text")
    with pytest.raises(ValidationError):
        CreatePmsTaskInput(subject="ok", body="x" * 4001)


def test_input_rejects_unknown_priority() -> None:
    with pytest.raises(ValidationError):
        CreatePmsTaskInput(
            subject="ok",
            body="text",
            priority="critical",  # type: ignore[arg-type]
        )


def test_input_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        CreatePmsTaskInput(
            subject="ok",
            body="text",
            secret="oops",  # type: ignore[call-arg]
        )
