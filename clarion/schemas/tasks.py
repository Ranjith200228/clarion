"""Domain model for PMS tasks (escalations / follow-ups).

A PMS task is whatever the agent files for a human to handle later — an
unverified-payer follow-up, a workers' comp claim check, an emergency
that needs immediate human attention, etc. Stored in the same SQLite
file as the rest of the structured pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from clarion.schemas.tools import TaskPriority

TaskStatus = Literal["open", "closed"]


class PmsTask(BaseModel):
    """One task filed by the agent for the front desk to action."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=4000)
    patient_id: str | None = None
    priority: TaskPriority = "normal"
    status: TaskStatus = "open"
    created_at: datetime
