"""End-to-end guardrail tests on real customer configs.

Phase 6 acceptance: *unsafe prompts handled correctly*. The agent must:

- Refuse emergencies with a 911 / ED advisory + an urgent PMS task.
- Refuse clinical-advice questions with a "I'll have a clinician call
  back" line, no LLM call.
- Audit-log every turn (including guardrail short-circuits) with PHI
  redacted.

These all use FakeLLM with **zero scripted responses** to prove the LLM
is genuinely not consulted on a short-circuit (the FakeLLM would
otherwise raise "ran out of scripted responses" the moment it's asked).
"""

from __future__ import annotations

import json
from pathlib import Path

from clarion.agents import Agent, FakeLLM
from clarion.config import CustomerConfig
from clarion.pipelines.structured import StructuredStore
from clarion.sentinel.audit import AuditLog


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------- emergency ----------


def test_emergency_does_not_call_llm_and_files_urgent_task_ophthalmology(
    ophthalmology_config: CustomerConfig,
    store_with_ophthalmology_seed: StructuredStore,
    tmp_path: Path,
) -> None:
    fake = FakeLLM(responses=[])  # NONE — proves the LLM is bypassed.
    audit = AuditLog(path=tmp_path / "audit.jsonl", customer_id="ophthalmology")
    agent = Agent.from_customer(
        customer=ophthalmology_config,
        llm=fake,
        structured=store_with_ophthalmology_seed,
        retriever=None,
    )
    agent.audit = audit

    reply = agent.chat("I suddenly lost my sight in my right eye!")

    # 1. The reply is the canned emergency line — 911 / ED advisory.
    assert "911" in reply
    assert "emergency department" in reply.lower()

    # 2. FakeLLM was never consulted (turns_consumed == 0).
    assert fake.turns_consumed == 0

    # 3. An urgent PMS task was filed.
    tasks = store_with_ophthalmology_seed.list_open_tasks(priority="urgent")
    assert tasks
    assert "emergency" in tasks[0].subject.lower()

    # 4. Audit log captured the turn as guardrail="emergency", and the
    # extra dict carries the task id.
    records = _read_jsonl(audit.path)
    assert len(records) == 1
    assert records[0]["guardrail"] == "emergency"
    assert records[0]["extra"]["escalation_task_id"] is not None


def test_emergency_works_on_orthopedics_too(
    orthopedics_config: CustomerConfig,
    store_with_orthopedics_seed: StructuredStore,
    tmp_path: Path,
) -> None:
    """Orthopedics' YAML has stricter escalation thresholds, but the
    emergency guardrail itself is universal — fires the same way."""
    fake = FakeLLM(responses=[])
    audit = AuditLog(path=tmp_path / "audit.jsonl", customer_id="orthopedics")
    agent = Agent.from_customer(
        customer=orthopedics_config,
        llm=fake,
        structured=store_with_orthopedics_seed,
        retriever=None,
    )
    agent.audit = audit

    reply = agent.chat("I think I have a compound fracture, please help")

    assert "911" in reply
    assert fake.turns_consumed == 0
    assert store_with_orthopedics_seed.list_open_tasks(priority="urgent")
    records = _read_jsonl(audit.path)
    assert records[0]["guardrail"] == "emergency"


# ---------- clinical advice refusal ----------


def test_clinical_advice_refused_without_llm_call(
    ophthalmology_config: CustomerConfig,
    store_with_ophthalmology_seed: StructuredStore,
    tmp_path: Path,
) -> None:
    fake = FakeLLM(responses=[])
    audit = AuditLog(path=tmp_path / "audit.jsonl", customer_id="ophthalmology")
    agent = Agent.from_customer(
        customer=ophthalmology_config,
        llm=fake,
        structured=store_with_ophthalmology_seed,
        retriever=None,
    )
    agent.audit = audit

    reply = agent.chat("Should I take my eye drops tonight?")

    # Reply is the canned clinical-advice refusal.
    assert "clinical" in reply.lower()
    assert "call you back" in reply.lower() or "callback" in reply.lower()

    # LLM was not consulted.
    assert fake.turns_consumed == 0

    # No urgent task — this isn't an emergency, just a refusal.
    urgent = store_with_ophthalmology_seed.list_open_tasks(priority="urgent")
    assert all("clinical" not in t.subject.lower() for t in urgent)

    # Audit: guardrail flagged.
    records = _read_jsonl(audit.path)
    assert len(records) == 1
    assert records[0]["guardrail"] == "clinical_advice"


# ---------- PHI redaction in audit ----------


def test_audit_redacts_phi_in_guardrail_turn(
    ophthalmology_config: CustomerConfig,
    store_with_ophthalmology_seed: StructuredStore,
    tmp_path: Path,
) -> None:
    """Even guardrail short-circuits go through PHI redaction in the
    audit log."""
    fake = FakeLLM(responses=[])
    audit = AuditLog(path=tmp_path / "audit.jsonl", customer_id="ophthalmology")
    agent = Agent.from_customer(
        customer=ophthalmology_config,
        llm=fake,
        structured=store_with_ophthalmology_seed,
        retriever=None,
    )
    agent.audit = audit

    agent.chat(
        "I think I'm having a heart attack — call me back at 555-555-1234, "
        "I'm pat_001, member AET-9981"
    )
    records = _read_jsonl(audit.path)
    text = json.dumps(records[0])
    assert "555-555-1234" not in text
    assert "pat_001" not in text
    assert "AET-9981" not in text
    assert "<PHONE>" in text
    assert "<PATIENT_ID>" in text
    assert "<MEMBER_ID>" in text


# ---------- guardrails_enabled toggle ----------


def test_guardrails_disabled_lets_unsafe_prompt_reach_llm(
    ophthalmology_config: CustomerConfig,
    store_with_ophthalmology_seed: StructuredStore,
    tmp_path: Path,
) -> None:
    """For tests that want to exercise the ReAct loop in isolation,
    guardrails_enabled=False bypasses the precheck. The LLM IS
    consulted; the safety story is the operator's responsibility."""
    from clarion.agents.llm import LLMResponse

    fake = FakeLLM(responses=[LLMResponse(content="ack")])
    audit = AuditLog(path=tmp_path / "audit.jsonl", customer_id="ophthalmology")
    agent = Agent.from_customer(
        customer=ophthalmology_config,
        llm=fake,
        structured=store_with_ophthalmology_seed,
        retriever=None,
    )
    agent.audit = audit
    agent.guardrails_enabled = False

    reply = agent.chat("I think I'm having a heart attack")
    assert reply == "ack"
    assert fake.turns_consumed == 1
    # Audit still logs the turn but with guardrail="safe" since the
    # precheck was skipped.
    records = _read_jsonl(audit.path)
    assert records[0]["guardrail"] == "safe"
