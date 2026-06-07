"""Sentinel trust engine: guardrails + PHI redaction + audit (Phase 6),
LLM-as-judge (Phase 10), escalation scorer (Phase 11)."""

from clarion.sentinel.guardrails import (
    GuardrailHit,
    GuardrailKind,
    detect_clinical_advice_request,
    detect_emergency,
)
from clarion.sentinel.phi import RedactionResult, redact, redact_with_counts

__all__ = [
    "GuardrailHit",
    "GuardrailKind",
    "RedactionResult",
    "detect_clinical_advice_request",
    "detect_emergency",
    "redact",
    "redact_with_counts",
]
