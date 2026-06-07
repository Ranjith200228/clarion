"""Sentinel trust engine: guardrails + PHI redaction + audit (Phase 6),
LLM-as-judge (Phase 10), escalation scorer (Phase 11)."""

from clarion.sentinel.phi import RedactionResult, redact, redact_with_counts

__all__ = [
    "RedactionResult",
    "redact",
    "redact_with_counts",
]
