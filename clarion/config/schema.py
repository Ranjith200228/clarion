"""Pydantic schema for a single Clarion customer (tenant).

A ``CustomerConfig`` is the only thing that differs between customers — agent
code reads this object and behaves accordingly. There is no per-customer
branching in agent code.

The schema captures everything mentioned in the spec's Phase 2 acceptance:

* specialties (e.g., "Cataract Pre-Op Consult")
* enabled_tools (subset of the registry)
* escalation thresholds (low confidence, clarification count, frustration)
* languages
* rules_path (where the unstructured rules corpus lives for RAG)
* agent_persona (system-prompt fragment describing how the agent presents)

Adding a new customer means dropping a new YAML in ``configs/``; adding a
new field here is the only place agent code learns about it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Closed enum: tools the platform ships. A customer enables a subset.
ToolName = Literal[
    "search_slots",
    "book_appointment",
    "cancel_appointment",
    "check_eligibility",
    "create_pms_task",
]


class EscalationThresholds(BaseModel):
    """When the supervisor should hand the call to a human.

    Each field is one signal; the escalation engine fuses them (Phase 11).
    Defaults are conservative — practices can tighten per customer.
    """

    model_config = ConfigDict(extra="forbid")

    # Agent's self-reported confidence below this triggers escalation.
    low_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    # Hard cap on clarification turns before we hand off.
    max_clarifications: int = Field(default=3, ge=1, le=20)

    # Frustration score (from voice-emotion + dialogue signals) above this
    # triggers escalation. Phase 18 wires the voice side in.
    frustration: float = Field(default=0.7, ge=0.0, le=1.0)

    # Always escalate if RAG returns conflicting rules.
    on_rule_conflict: bool = Field(default=True)


class CustomerConfig(BaseModel):
    """Everything Clarion needs to behave correctly for one customer."""

    model_config = ConfigDict(extra="forbid")

    # Stable identifier (also the YAML filename stem, e.g. "ophthalmology").
    customer_id: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")

    # Human-readable practice name.
    display_name: str = Field(min_length=1)

    # Vertical tag — keeps the door open for non-healthcare verticals later.
    vertical: str = Field(default="healthcare-scheduling", min_length=1)

    # Free-form specialty list shown to the agent.
    specialties: list[str] = Field(min_length=1)

    # Subset of the platform tool registry this customer wants enabled.
    enabled_tools: list[ToolName] = Field(min_length=1)

    # Escalation policy.
    escalation: EscalationThresholds = Field(default_factory=EscalationThresholds)

    # ISO 639-1 language codes (e.g. "en", "es").
    languages: list[str] = Field(default_factory=lambda: ["en"], min_length=1)

    # Directory holding rules markdown chunks for RAG (Phase 3 indexes it).
    rules_path: Path

    # System-prompt fragment describing the agent's persona for this customer.
    agent_persona: str = Field(min_length=1)

    @field_validator("enabled_tools")
    @classmethod
    def _unique_tools(cls, v: list[ToolName]) -> list[ToolName]:
        if len(set(v)) != len(v):
            raise ValueError("enabled_tools must not contain duplicates")
        return v

    @field_validator("languages")
    @classmethod
    def _lower_languages(cls, v: list[str]) -> list[str]:
        out = [lang.lower() for lang in v]
        if len(set(out)) != len(out):
            raise ValueError("languages must not contain duplicates")
        return out
