"""Sentinel trust engine: guardrails + PHI redaction + audit (Phase 6),
LLM-as-judge (Phase 10), escalation scorer (Phase 11)."""

from clarion.sentinel.audit import AuditLog, AuditTurn, new_conversation_id
from clarion.sentinel.escalation import (
    ConversationFacts,
    EscalationScorer,
    compute_stats,
)
from clarion.sentinel.frustration import (
    FrustrationResult,
    detect_frustration,
    detect_frustration_over_turns,
)
from clarion.sentinel.guardrails import (
    GuardrailHit,
    GuardrailKind,
    detect_clinical_advice_request,
    detect_emergency,
)
from clarion.sentinel.judge import Judge
from clarion.sentinel.phi import RedactionResult, redact, redact_with_counts

__all__ = [
    "AuditLog",
    "AuditTurn",
    "ConversationFacts",
    "EscalationScorer",
    "FrustrationResult",
    "GuardrailHit",
    "GuardrailKind",
    "Judge",
    "RedactionResult",
    "compute_stats",
    "detect_clinical_advice_request",
    "detect_emergency",
    "detect_frustration",
    "detect_frustration_over_turns",
    "new_conversation_id",
    "redact",
    "redact_with_counts",
]
