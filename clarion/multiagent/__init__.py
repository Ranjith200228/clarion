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

from clarion.multiagent.state import (
    MultiAgentState,
    SpecialistIntent,
    SupervisorDecision,
    initial_state,
)

__all__ = [
    "MultiAgentState",
    "SpecialistIntent",
    "SupervisorDecision",
    "initial_state",
]
