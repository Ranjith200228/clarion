"""``check_eligibility`` — look up the patient's insurance eligibility record.

Wraps ``StructuredStore.get_eligibility``. Returns ``on_file=False`` when
no record exists (so the agent can collect payer details and create a
follow-up task), and the full record when it does.
"""

from __future__ import annotations

from typing import ClassVar

from clarion.schemas.tools import (
    CheckEligibilityInput,
    CheckEligibilityOutput,
)
from clarion.tools.base import ToolContext, run_with_retry


class CheckEligibilityTool:
    """Return the patient's eligibility record, or a "not on file" signal."""

    name: ClassVar[str] = "check_eligibility"
    input_model: ClassVar[type[CheckEligibilityInput]] = CheckEligibilityInput
    output_model: ClassVar[type[CheckEligibilityOutput]] = CheckEligibilityOutput

    def run(self, input: CheckEligibilityInput, ctx: ToolContext) -> CheckEligibilityOutput:
        try:
            record = run_with_retry(lambda: ctx.structured.get_eligibility(input.patient_id))
        except Exception as e:
            return CheckEligibilityOutput(
                ok=False,
                error=f"check_eligibility failed: {e}",
                record=None,
                on_file=False,
            )
        return CheckEligibilityOutput(
            ok=True,
            record=record,
            on_file=record is not None,
        )
