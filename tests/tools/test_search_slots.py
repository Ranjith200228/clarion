"""Tests for the search_slots tool."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest
from clarion.schemas.tools import SearchSlotsInput
from clarion.tools.base import ToolContext
from clarion.tools.search_slots import SearchSlotsTool
from pydantic import ValidationError


def test_returns_ok_with_matching_slots(ctx: ToolContext) -> None:
    out = SearchSlotsTool().run(
        SearchSlotsInput(appointment_type="Consult", on_or_after=date(2026, 6, 1)),
        ctx,
    )
    assert out.ok is True
    assert out.error is None
    assert [s.slot_id for s in out.slots] == ["slot_demo_1", "slot_demo_2"]


def test_respects_provider_filter(ctx: ToolContext) -> None:
    out = SearchSlotsTool().run(
        SearchSlotsInput(
            appointment_type="Consult",
            on_or_after=date(2026, 6, 1),
            provider_id="ghost",
        ),
        ctx,
    )
    assert out.ok is True
    assert out.slots == []


def test_respects_limit(ctx: ToolContext) -> None:
    out = SearchSlotsTool().run(
        SearchSlotsInput(
            appointment_type="Consult",
            on_or_after=date(2026, 6, 1),
            limit=1,
        ),
        ctx,
    )
    assert out.ok is True
    assert len(out.slots) == 1


def test_input_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SearchSlotsInput(
            appointment_type="Consult",
            on_or_after=date(2026, 6, 1),
            secret="oops",  # type: ignore[call-arg]
        )


def test_input_rejects_limit_out_of_range() -> None:
    with pytest.raises(ValidationError):
        SearchSlotsInput(appointment_type="Consult", on_or_after=date(2026, 6, 1), limit=0)
    with pytest.raises(ValidationError):
        SearchSlotsInput(appointment_type="Consult", on_or_after=date(2026, 6, 1), limit=999)


def test_db_failure_returns_structured_error(
    customer_config: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the store throws, the tool must NOT raise — it returns ok=False."""
    bad_store = MagicMock()
    bad_store.search_slots.side_effect = RuntimeError("simulated db outage")
    ctx = ToolContext(customer=customer_config, structured=bad_store)  # type: ignore[arg-type]

    out = SearchSlotsTool().run(
        SearchSlotsInput(appointment_type="Consult", on_or_after=date(2026, 6, 1)),
        ctx,
    )
    assert out.ok is False
    assert out.error is not None
    assert "search_slots failed" in out.error
    assert out.slots == []
