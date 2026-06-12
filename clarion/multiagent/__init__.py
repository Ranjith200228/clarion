"""LangGraph-backed multi-agent runtime.

Hierarchical: ``router`` (intent classification) → one of N
``specialists`` (focused tool subset) → ``supervisor`` (completion
check + escalation). All nodes share the same ``CustomerConfig`` +
``LLMClient`` + ``ToolDispatcher`` that the single-agent backend
uses; the trust engine (guardrails, judge, PHI redaction) sits at
the runtime boundary so it applies to both backends identically.

Public surface (built up across commits):
  MultiAgentState, SpecialistIntent, SupervisorDecision  (commit 1)
  IntentRouter                                           (commit 2)
  specialists/                                           (commit 3)
  Supervisor                                             (commit 4)
  MultiAgentRunner                                       (commit 5)
"""

from clarion.multiagent.router import (
    HeuristicIntentRouter,
    LLMIntentRouter,
    Router,
)
from clarion.multiagent.specialists import (
    EMERGENCY_REPLY,
    BookingSpecialist,
    CancelSpecialist,
    EligibilitySpecialist,
    EmergencySpecialist,
    InfoSpecialist,
    Specialist,
)
from clarion.multiagent.state import (
    MultiAgentState,
    SpecialistIntent,
    SupervisorDecision,
    initial_state,
)
from clarion.multiagent.supervisor import (
    DEFAULT_MAX_VISITS,
    ESCALATION_HANDOFF_TEXT,
    Supervisor,
    route_after_supervisor,
)

__all__ = [
    "BookingSpecialist",
    "CancelSpecialist",
    "DEFAULT_MAX_VISITS",
    "EMERGENCY_REPLY",
    "ESCALATION_HANDOFF_TEXT",
    "EligibilitySpecialist",
    "EmergencySpecialist",
    "HeuristicIntentRouter",
    "InfoSpecialist",
    "LLMIntentRouter",
    "MultiAgentState",
    "Router",
    "Specialist",
    "SpecialistIntent",
    "Supervisor",
    "SupervisorDecision",
    "initial_state",
    "route_after_supervisor",
]
