"""``create_pms_task`` — file a task for a human to action.

This is the agent's escape valve: any time the workflow needs human
follow-up (unverified payer, workers' comp pending, emergency, refill
request), the agent calls this tool. The actual handling happens out
of band.
"""

from __future__ import annotations

from typing import ClassVar

from clarion.schemas.tools import (
    CreatePmsTaskInput,
    CreatePmsTaskOutput,
)
from clarion.tools.base import ToolContext, run_with_retry


class CreatePmsTaskTool:
    """File a follow-up task for the front desk."""

    name: ClassVar[str] = "create_pms_task"
    input_model: ClassVar[type[CreatePmsTaskInput]] = CreatePmsTaskInput
    output_model: ClassVar[type[CreatePmsTaskOutput]] = CreatePmsTaskOutput

    def run(self, input: CreatePmsTaskInput, ctx: ToolContext) -> CreatePmsTaskOutput:
        try:
            task = run_with_retry(
                lambda: ctx.structured.create_task(
                    subject=input.subject,
                    body=input.body,
                    patient_id=input.patient_id,
                    priority=input.priority,
                )
            )
        except Exception as e:
            return CreatePmsTaskOutput(
                ok=False,
                error=f"create_pms_task failed: {e}",
                task_id=None,
            )
        return CreatePmsTaskOutput(ok=True, task_id=task.task_id)
