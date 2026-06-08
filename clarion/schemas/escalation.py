"""Schemas for the Phase 11 escalation engine.

Five signals are scored in [0, 1] (higher = more reason to escalate) and
combined via a weighted sum into a single composite ``score`` field.
``should_escalate`` is derived from ``score >= threshold``; the threshold
defaults to 0.5 but the customer config can move it.

Phase 12's evaluation framework will read ``EscalationStats`` (the
precision/recall rollup) directly into ``evaluation_report.json``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EscalationSignals(BaseModel):
    """Per-signal escalation scores, all in [0, 1]."""

    model_config = ConfigDict(extra="forbid")

    low_confidence: float = Field(ge=0.0, le=1.0)
    repeated_clarification: float = Field(ge=0.0, le=1.0)
    rule_conflict: float = Field(ge=0.0, le=1.0)
    frustration: float = Field(ge=0.0, le=1.0)
    unsupported_request: float = Field(ge=0.0, le=1.0)


class EscalationWeights(BaseModel):
    """Composite-score weights. Should sum to 1.0 in practice; the scorer
    normalizes defensively if not.

    Defaults are tuned so any single severe signal can push past a 0.5
    threshold, but two mild signals also can — matches the "fuse multiple
    weak signals" intent in the spec.
    """

    model_config = ConfigDict(extra="forbid")

    low_confidence: float = Field(default=0.30, ge=0.0, le=1.0)
    repeated_clarification: float = Field(default=0.15, ge=0.0, le=1.0)
    rule_conflict: float = Field(default=0.20, ge=0.0, le=1.0)
    frustration: float = Field(default=0.20, ge=0.0, le=1.0)
    unsupported_request: float = Field(default=0.15, ge=0.0, le=1.0)


class EscalationScore(BaseModel):
    """One turn / conversation's escalation verdict."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    signals: EscalationSignals
    threshold: float = Field(ge=0.0, le=1.0)
    should_escalate: bool
    reasons: list[str] = Field(default_factory=list)


class EscalationStats(BaseModel):
    """Aggregate precision / recall / F1 across a scenario batch.

    ``true_positive`` = predicted_escalate AND ground_truth_escalate.
    Predictions come from the scorer; ground truth from
    ``Scenario.ground_truth.should_escalate``.
    """

    model_config = ConfigDict(extra="forbid")

    total: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
