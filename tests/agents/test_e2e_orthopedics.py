"""End-to-end multi-tenant test on the orthopedics config.

This is the Phase 5 acceptance test for *multi-tenancy*: the same Agent
class, the same react_loop, the same tools — only the YAML changes — and
the agent's behavior shifts accordingly.

The clearest single check: orthopedics' YAML drops cancel_appointment.
If the LLM tries to call it anyway, the registry refuses, the tool
result reads "not enabled", and a competent LLM falls back to filing a
PMS task per the practice's "all cancellations go to a human" rule.
"""

from __future__ import annotations

from datetime import date

from clarion.agents import Agent, FakeLLM, LLMResponse, ToolCall
from clarion.config import CustomerConfig
from clarion.pipelines.structured import StructuredStore
from clarion.pipelines.unstructured import chunk_rules_dir
from clarion.rag.embeddings import TfidfEmbedder
from clarion.rag.retriever import Retriever


def _build_retriever(customer: CustomerConfig, out_dir) -> Retriever:  # type: ignore[no-untyped-def]
    chunks = chunk_rules_dir(customer.rules_path)
    return Retriever.build_and_save(chunks, embedder=TfidfEmbedder(), out_dir=out_dir)


def test_cancellation_falls_back_to_pms_task_when_tool_disabled(
    orthopedics_config: CustomerConfig,
    store_with_orthopedics_seed: StructuredStore,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """The agent attempts cancel_appointment, the registry refuses (it's
    not in orthopedics' enabled_tools), and the next LLM turn files a
    PMS task — exactly what 05_cancellation_and_reschedule.md prescribes.
    """
    retriever = _build_retriever(orthopedics_config, tmp_path / "rag")
    fake = FakeLLM(
        responses=[
            # 1. LLM tries cancel_appointment (it was greedy / didn't notice
            # cancel_appointment isn't advertised in the tools list).
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="cancel_appointment",
                        arguments={"appointment_id": "appt_doesnt_matter"},
                    ),
                )
            ),
            # 2. Tool result was ok=False ("not enabled"). LLM reads the
            # rules and switches to filing a task.
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        id="c2",
                        name="create_pms_task",
                        arguments={
                            "subject": "Cancellation request — call back",
                            "body": "Patient pat_101 wants to cancel their upcoming "
                            "appointment. Per policy, cancellations route to a human. "
                            "Callback: 555-0101.",
                            "patient_id": "pat_101",
                            "priority": "normal",
                        },
                    ),
                )
            ),
            # 3. Final text to the patient.
            LLMResponse(
                content=(
                    "I've passed your cancellation request to our team — someone "
                    "will call you back within one business day to confirm."
                )
            ),
        ]
    )

    agent = Agent.from_customer(
        customer=orthopedics_config,
        llm=fake,
        structured=store_with_orthopedics_seed,
        retriever=retriever,
    )
    reply = agent.chat("Hi, this is pat_101. I need to cancel my appointment.")

    # 1. The agent told the patient a human will handle it.
    assert "call you back" in reply.lower() or "callback" in reply.lower()

    # 2. The first tool reply explicitly was "not enabled".
    tool_msgs = [m for m in agent.transcript if m.role == "tool"]
    assert tool_msgs
    assert tool_msgs[0].name == "cancel_appointment"
    assert tool_msgs[0].content is not None
    assert "not enabled" in tool_msgs[0].content
    # And the second was the create_pms_task success.
    assert tool_msgs[1].name == "create_pms_task"
    assert '"ok": true' in tool_msgs[1].content  # type: ignore[arg-type]

    # 3. A real task was filed in the store with the right subject.
    tasks = store_with_orthopedics_seed.list_open_tasks()
    assert any("cancellation" in t.subject.lower() for t in tasks)


def test_tools_advertised_to_llm_match_orthopedics_yaml(
    orthopedics_config: CustomerConfig,
    store_with_orthopedics_seed: StructuredStore,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """Belt-and-suspenders: even if the LLM never tries to call
    cancel_appointment, it must never *see* it in the advertised tool
    list."""
    retriever = _build_retriever(orthopedics_config, tmp_path / "rag2")
    fake = FakeLLM(responses=[LLMResponse(content="ok")])
    agent = Agent.from_customer(
        customer=orthopedics_config,
        llm=fake,
        structured=store_with_orthopedics_seed,
        retriever=retriever,
    )
    agent.chat("hi")

    _, tools = fake.calls[-1]
    advertised = {t.name for t in tools}
    assert "cancel_appointment" not in advertised
    assert advertised == {
        "search_slots",
        "book_appointment",
        "check_eligibility",
        "create_pms_task",
    }


def test_reschedule_emerges_from_search_then_book(
    orthopedics_config: CustomerConfig,
    store_with_orthopedics_seed: StructuredStore,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """Rescheduling is not a separate tool — the agent composes it from
    search_slots + book_appointment (and, since orthopedics doesn't have
    cancel_appointment, a PMS task for the old slot)."""
    retriever = _build_retriever(orthopedics_config, tmp_path / "rag3")
    fake = FakeLLM(
        responses=[
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="search_slots",
                        arguments={
                            "appointment_type": "New Patient Joint Consult",
                            "on_or_after": date(2026, 6, 1).isoformat(),
                            "limit": 3,
                        },
                    ),
                )
            ),
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        id="c2",
                        name="book_appointment",
                        arguments={
                            "slot_id": "slot_ortho_001",
                            "patient_id": "pat_101",
                            "patient_name": "Casey Lopez",
                            "patient_phone": "555-0142",
                            "patient_email": "casey.lopez@example.invalid",
                        },
                    ),
                )
            ),
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        id="c3",
                        name="create_pms_task",
                        arguments={
                            "subject": "Cancel old appointment",
                            "body": "Patient pat_101 rescheduled to slot_ortho_001 "
                            "on 2026-06-15. Cancel their prior 2026-06-12 booking.",
                            "patient_id": "pat_101",
                            "priority": "normal",
                        },
                    ),
                )
            ),
            LLMResponse(content="Done — you're now booked for June 15."),
        ]
    )
    agent = Agent.from_customer(
        customer=orthopedics_config,
        llm=fake,
        structured=store_with_orthopedics_seed,
        retriever=retriever,
    )
    reply = agent.chat("Hi, I'm pat_101. I'd like to reschedule my joint consult to next week.")

    assert "June 15" in reply
    # New slot reserved.
    appts = store_with_orthopedics_seed.get_appointment.__self__  # type: ignore[attr-defined]
    # Easier: query directly.
    open_after = store_with_orthopedics_seed.search_slots(
        appointment_type="New Patient Joint Consult",
        on_or_after=date(2026, 6, 1),
    )
    assert "slot_ortho_001" not in {s.slot_id for s in open_after}
    # And the cancellation-side task is filed.
    tasks = store_with_orthopedics_seed.list_open_tasks()
    assert any("cancel" in t.subject.lower() for t in tasks)
    # Keep the type-checker happy about the unused intermediate.
    _ = appts
