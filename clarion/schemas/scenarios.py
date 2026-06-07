"""Schemas for the simulation harness.

A ``Scenario`` is one synthetic patient interaction the agent will be
asked to handle. ``GroundTruth`` is what should happen. ``HarnessResult``
captures what actually happened so the Phase 12 metrics framework can
score it.

These live in ``clarion.schemas`` so the API layer (Phase 8) and a
future dashboard can read scenarios without dragging in simulator code.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Difficulty = Literal[
    "clear",
    "ambiguous",
    "rule_violating",
    "emergency",
    "non_english",
]

Intent = Literal[
    "book",
    "cancel",
    "reschedule",
    "eligibility",
    "faq",
    "emergency",
    "clinical_advice",
]

Outcome = Literal[
    "booked",
    "cancelled",
    "task_created",
    "escalated_emergency",
    "refused_clinical",
    "info_provided",
]


class GroundTruth(BaseModel):
    """What the agent is supposed to do for this scenario."""

    model_config = ConfigDict(extra="forbid")

    expected_outcome: Outcome
    should_escalate: bool  # any human handoff (urgent task or callback task)
    expected_tools: list[str] = Field(default_factory=list)
    expected_appointment_type: str | None = None
    notes: str = ""


class LLMScriptStep(BaseModel):
    """One ``LLMResponse`` worth of canned data.

    Optional ``content`` + zero or more ``tool_calls``. Mirrors the
    ``LLMResponse`` shape so the scripted FakeLLM in the harness can
    replay this verbatim. Token counts default to small synthetic values
    so cost / latency aggregates remain non-zero in CI.
    """

    model_config = ConfigDict(extra="forbid")

    content: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    input_tokens: int = 50
    output_tokens: int = 20
    model: str = "gpt-4o-mini"


class Scenario(BaseModel):
    """One synthetic patient interaction."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    customer_id: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    difficulty: Difficulty
    intent: Intent
    messages: list[str] = Field(min_length=1, max_length=10)
    ground_truth: GroundTruth
    # Optional canned LLM script for scripted-mode runs (no real LLM call).
    # When absent, the harness needs a live LLM client.
    llm_script: list[LLMScriptStep] = Field(default_factory=list)


class HarnessResult(BaseModel):
    """What actually happened when a scenario was run."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    customer_id: str
    difficulty: Difficulty
    intent: Intent
    actual_outcome: Outcome
    actual_tools: list[str]
    escalated: bool  # any urgent / non-urgent task was filed
    agent_replies: list[str]
    trace_ids: list[str]
    passed: bool
    failure_reasons: list[str] = Field(default_factory=list)


class HarnessReport(BaseModel):
    """Aggregate results for one harness run."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    by_difficulty: dict[str, dict[str, int]]
    by_intent: dict[str, dict[str, int]]
    results: list[HarnessResult]
