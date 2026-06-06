"""``search_slots`` — find open appointment slots matching a filter.

Thin wrapper over ``StructuredStore.search_slots`` that adapts the I/O to
the tool protocol and applies retry + structured error handling.
"""

from __future__ import annotations

from typing import ClassVar

from clarion.schemas.tools import (
    SearchSlotsInput,
    SearchSlotsOutput,
)
from clarion.tools.base import ToolContext, run_with_retry


class SearchSlotsTool:
    """Return upcoming open slots, sorted by date then time."""

    name: ClassVar[str] = "search_slots"
    input_model: ClassVar[type[SearchSlotsInput]] = SearchSlotsInput
    output_model: ClassVar[type[SearchSlotsOutput]] = SearchSlotsOutput

    def run(self, input: SearchSlotsInput, ctx: ToolContext) -> SearchSlotsOutput:
        try:
            slots = run_with_retry(
                lambda: ctx.structured.search_slots(
                    appointment_type=input.appointment_type,
                    on_or_after=input.on_or_after,
                    provider_id=input.provider_id,
                    limit=input.limit,
                )
            )
        except Exception as e:
            return SearchSlotsOutput(ok=False, error=f"search_slots failed: {e}", slots=[])
        return SearchSlotsOutput(ok=True, slots=slots)
