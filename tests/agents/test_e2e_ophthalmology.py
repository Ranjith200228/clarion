"""End-to-end booking flow on the real ophthalmology config.

This is the headline Phase 5 acceptance test: a patient calls, asks to
book a cataract pre-op consult, the agent searches, checks eligibility,
books, and confirms. The whole flow runs through the real Agent class
with the real ophthalmology YAML + seeded SQLite, scripted by FakeLLM
so it's deterministic.

The FakeLLM here plays the role the OpenAI LLM will play in production:
it decides which tool to call next given the conversation state. We
script exactly the calls we expect, and assert:

1. each call returned ok=True with sensible data
2. the SQLite store reflects the change (slot booked, appointment row
   present)
3. the agent's final text matches what the patient should hear
"""

from __future__ import annotations

from datetime import date

from clarion.agents import Agent, FakeLLM, LLMResponse, ToolCall
from clarion.config import CustomerConfig
from clarion.pipelines.structured import StructuredStore
from clarion.rag.embeddings import TfidfEmbedder
from clarion.rag.retriever import Retriever


def _build_retriever(customer: CustomerConfig, tmp_path) -> Retriever:  # type: ignore[no-untyped-def]
    # Use the real rules dir, build a fresh tmp index so test artifacts
    # don't depend on whether the user ran the ingest CLI.
    from clarion.pipelines.unstructured import chunk_rules_dir

    chunks = chunk_rules_dir(customer.rules_path)
    return Retriever.build_and_save(chunks, embedder=TfidfEmbedder(), out_dir=tmp_path)


def test_full_booking_flow_for_cataract_pre_op(
    ophthalmology_config: CustomerConfig,
    store_with_ophthalmology_seed: StructuredStore,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    retriever = _build_retriever(ophthalmology_config, tmp_path / "rag")

    # The script the LLM "would have generated" — five steps:
    #   1. check eligibility for the patient
    #   2. search slots for the asked-for appointment type
    #   3. book the first slot
    #   4. file a follow-up task summarizing the booking
    #   5. tell the patient (free text)
    fake = FakeLLM(
        responses=[
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="check_eligibility",
                        arguments={"patient_id": "pat_001"},
                    ),
                )
            ),
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        id="c2",
                        name="search_slots",
                        arguments={
                            "appointment_type": "Cataract Pre-Op Consult",
                            "on_or_after": date(2026, 6, 1).isoformat(),
                            "limit": 3,
                        },
                    ),
                )
            ),
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        id="c3",
                        name="book_appointment",
                        arguments={
                            "slot_id": "slot_oph_001",
                            "patient_id": "pat_001",
                            "notes": "Callback ok at 555-0100.",
                        },
                    ),
                )
            ),
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        id="c4",
                        name="create_pms_task",
                        arguments={
                            "subject": "Confirm cataract pre-op transport",
                            "body": "Patient pat_001 booked for 2026-06-15 09:00. "
                            "Confirm they have a ride home (dilated visit).",
                            "patient_id": "pat_001",
                            "priority": "normal",
                        },
                    ),
                )
            ),
            LLMResponse(
                content=(
                    "You're all set — Cataract Pre-Op Consult on June 15 at 9 AM "
                    "with Dr. Smith at the Main Campus. Plan for a ride home "
                    "since you'll be dilated. We'll see you then."
                )
            ),
        ]
    )

    agent = Agent.from_customer(
        customer=ophthalmology_config,
        llm=fake,
        structured=store_with_ophthalmology_seed,
        retriever=retriever,
    )
    reply = agent.chat(
        "Hi, this is John Doe (patient id pat_001). I'd like a cataract pre-op "
        "consult some time after June 1 — what's available?"
    )

    # 1. The agent's final text reaches the patient.
    assert "Cataract" in reply
    assert "June 15" in reply

    # 2. The slot is no longer searchable; the appointment row exists.
    open_after = store_with_ophthalmology_seed.search_slots(
        appointment_type="Cataract Pre-Op Consult",
        on_or_after=date(2026, 6, 1),
    )
    assert "slot_oph_001" not in {s.slot_id for s in open_after}

    # 3. A follow-up task was filed.
    open_tasks = store_with_ophthalmology_seed.list_open_tasks()
    assert any("transport" in t.subject.lower() for t in open_tasks)

    # 4. We consumed exactly five LLM turns (no surprise extra round trips).
    assert fake.turns_consumed == 5

    # 5. Each tool call returned ok=True (no silent failures).
    tool_msgs = [m for m in agent.transcript if m.role == "tool"]
    for tm in tool_msgs:
        assert tm.content is not None and '"ok": true' in tm.content


def test_faq_question_grounds_on_rag_without_tool_calls(
    ophthalmology_config: CustomerConfig,
    store_with_ophthalmology_seed: StructuredStore,
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """A pure FAQ ("do you accept Kaiser?") doesn't need a tool — the
    agent reads the practice rules block from the system prompt and
    answers from there. We verify by inspecting *what* was sent to the
    LLM (the rules block must mention Kaiser)."""
    retriever = _build_retriever(ophthalmology_config, tmp_path / "rag2")
    fake = FakeLLM(
        responses=[
            LLMResponse(
                content=(
                    "We're sorry — Kaiser Permanente is out-of-network for us. "
                    "We can offer self-pay pricing or refer you to an in-network "
                    "practice if you'd prefer."
                )
            )
        ]
    )
    agent = Agent.from_customer(
        customer=ophthalmology_config,
        llm=fake,
        structured=store_with_ophthalmology_seed,
        retriever=retriever,
    )
    reply = agent.chat("Hi, do you accept Kaiser insurance?")

    # The reply quotes the correct answer.
    assert "kaiser" in reply.lower()
    assert "out-of-network" in reply.lower()

    # The LLM was given the right rule chunk in the system prompt.
    messages, _ = fake.calls[-1]
    system_msg = next(m for m in messages if m.role == "system")
    assert system_msg.content is not None
    assert "Kaiser" in system_msg.content
    assert "04_insurance_and_payers.md" in system_msg.content
