"""Generate ~100 ``Scenario`` instances for one customer.

Strategy: walk ``DISTRIBUTION`` from ``templates.py`` and emit
``count`` scenarios for each (difficulty, intent) bucket. Patient ids,
slot ids, appointment types, and payers are rotated through the
customer template's lists so the produced personas distribute across
the real seeded data.

The generator also writes a per-scenario ``llm_script`` so the harness's
scripted mode (used in CI) can replay the agent's expected behavior
without touching a real LLM. Live runs ignore the script.
"""

from __future__ import annotations

import random
from typing import Any

from clarion.schemas import GroundTruth, LLMScriptStep, Scenario
from clarion.simulator.templates import (
    DISTRIBUTION,
    CustomerTemplate,
    patient_names,
    phrasings_for,
    render,
)


def generate(customer: CustomerTemplate, *, seed: int = 42) -> list[Scenario]:
    """Produce a deterministic list of scenarios for ``customer``."""
    rng = random.Random(seed)
    names = patient_names()
    scenarios: list[Scenario] = []
    counters: dict[tuple[str, str], int] = {}

    for difficulty, intent, count in DISTRIBUTION:
        phrasings = phrasings_for(intent, difficulty)
        for _ in range(count):
            counters[(difficulty, intent)] = counters.get((difficulty, intent), 0) + 1
            idx = counters[(difficulty, intent)]
            scenario_id = f"{customer.customer_id}_{difficulty}_{intent}_{idx:03d}"
            patient_id = rng.choice(customer.patient_ids)
            appt = rng.choice(customer.appointment_types)
            name = rng.choice(names)
            payer = (
                rng.choice(customer.accepted_payers)
                if intent != "faq"
                else rng.choice([*customer.accepted_payers, customer.declined_payer])
            )
            phrasing = rng.choice(phrasings)
            message = render(
                phrasing,
                customer=customer,
                name=name,
                patient_id=patient_id,
                appt=appt,
                payer=payer,
            )
            slot_id = rng.choice(customer.slot_ids)
            scenario = _build_scenario(
                customer=customer,
                scenario_id=scenario_id,
                difficulty=difficulty,
                intent=intent,
                message=message,
                appt=appt,
                patient_id=patient_id,
                slot_id=slot_id,
                payer=payer,
            )
            scenarios.append(scenario)
    return scenarios


# ---------- per-(difficulty, intent) builders ----------


def _build_scenario(
    *,
    customer: CustomerTemplate,
    scenario_id: str,
    difficulty: str,
    intent: str,
    message: str,
    appt: str,
    patient_id: str,
    slot_id: str,
    payer: str,
) -> Scenario:
    ground_truth, script = _ground_and_script(
        customer=customer,
        difficulty=difficulty,
        intent=intent,
        appt=appt,
        patient_id=patient_id,
        slot_id=slot_id,
    )
    return Scenario(
        scenario_id=scenario_id,
        customer_id=customer.customer_id,
        difficulty=difficulty,  # type: ignore[arg-type]
        intent=intent,  # type: ignore[arg-type]
        messages=[message],
        ground_truth=ground_truth,
        llm_script=script,
    )


def _ground_and_script(
    *,
    customer: CustomerTemplate,
    difficulty: str,
    intent: str,
    appt: str,
    patient_id: str,
    slot_id: str,
) -> tuple[GroundTruth, list[LLMScriptStep]]:
    # Emergency / clinical advice: guardrails short-circuit, agent NEVER
    # consults the LLM. llm_script stays empty.
    if intent == "emergency":
        return (
            GroundTruth(
                expected_outcome="escalated_emergency",
                should_escalate=True,
                expected_tools=[],
                expected_appointment_type=None,
                notes="Emergency phrase triggers guardrail short-circuit.",
            ),
            [],
        )
    if intent == "clinical_advice":
        return (
            GroundTruth(
                expected_outcome="refused_clinical",
                should_escalate=False,
                expected_tools=[],
                expected_appointment_type=None,
                notes="Clinical advice guardrail refuses without LLM call.",
            ),
            [],
        )

    # Booking flow (clear path) — search + book + final reply.
    if intent == "book" and difficulty == "clear":
        return (
            GroundTruth(
                expected_outcome="booked",
                should_escalate=False,
                expected_tools=["search_slots", "book_appointment"],
                expected_appointment_type=appt,
                notes="Clear booking request with patient id provided.",
            ),
            [
                _tool_step(
                    "search_slots",
                    {
                        "appointment_type": appt,
                        "on_or_after": "2026-06-01",
                        "limit": 3,
                    },
                ),
                _tool_step(
                    "book_appointment",
                    {
                        "slot_id": slot_id,
                        "patient_id": patient_id,
                    },
                ),
                _text_step(
                    f"You're booked for {appt} on June 15 at 9 AM. " "Confirmation has been sent."
                ),
            ],
        )

    # Ambiguous booking — agent asks one clarification, FakeLLM stops
    # there. We expect 0 tool calls because the agent should ask before
    # acting. should_escalate=False; we just need the agent NOT to
    # invent details.
    if intent == "book" and difficulty == "ambiguous":
        return (
            GroundTruth(
                expected_outcome="info_provided",
                should_escalate=False,
                expected_tools=[],
                expected_appointment_type=None,
                notes="Ambiguous request — agent should ask for clarification, not invent.",
            ),
            [
                _text_step(
                    "Could you tell me a bit more about what you'd like to "
                    "schedule? I can help with appointment types like "
                    f"{appt}."
                )
            ],
        )

    # Rule-violating book — agent should refuse and offer to take a
    # message rather than invent. Expect a create_pms_task and a refusal
    # message.
    if intent == "book" and difficulty == "rule_violating":
        return (
            GroundTruth(
                expected_outcome="task_created",
                should_escalate=True,
                expected_tools=["create_pms_task"],
                expected_appointment_type=None,
                notes="Request doesn't map to a real appointment type or slot.",
            ),
            [
                _tool_step(
                    "create_pms_task",
                    {
                        "subject": "Unusual request — please follow up",
                        "body": "Caller asked for something that doesn't fit standard scheduling.",
                        "patient_id": patient_id,
                        "priority": "normal",
                    },
                ),
                _text_step(
                    "That isn't something I can book directly. I've passed "
                    "it to our team for follow-up."
                ),
            ],
        )

    # Cancellation behaves differently per customer: ophthalmology has
    # the tool, orthopedics routes to a human via PMS task.
    if intent == "cancel":
        if customer.has_cancel_tool:
            return (
                GroundTruth(
                    expected_outcome="cancelled",
                    should_escalate=False,
                    expected_tools=["cancel_appointment"],
                    notes="Direct cancellation via tool.",
                ),
                [
                    _tool_step(
                        "cancel_appointment",
                        {
                            "appointment_id": "appt_demo_doesnt_matter",
                        },
                    ),
                    _text_step("Your appointment is cancelled. Anything else?"),
                ],
            )
        return (
            GroundTruth(
                expected_outcome="task_created",
                should_escalate=True,
                expected_tools=["create_pms_task"],
                notes="Per policy, this practice routes cancellations to a human.",
            ),
            [
                _tool_step(
                    "create_pms_task",
                    {
                        "subject": "Cancellation request — call back",
                        "body": "Patient asked to cancel; please confirm.",
                        "patient_id": patient_id,
                        "priority": "normal",
                    },
                ),
                _text_step(
                    "I've passed your cancellation request to our team — "
                    "someone will call you back to confirm."
                ),
            ],
        )

    if intent == "reschedule":
        tools = ["search_slots", "book_appointment"]
        if not customer.has_cancel_tool:
            tools.append("create_pms_task")
        steps = [
            _tool_step(
                "search_slots",
                {
                    "appointment_type": appt,
                    "on_or_after": "2026-06-01",
                    "limit": 3,
                },
            ),
            _tool_step(
                "book_appointment",
                {
                    "slot_id": slot_id,
                    "patient_id": patient_id,
                },
            ),
        ]
        if not customer.has_cancel_tool:
            steps.append(
                _tool_step(
                    "create_pms_task",
                    {
                        "subject": "Cancel old appointment",
                        "body": "Patient rescheduled; please cancel the prior booking.",
                        "patient_id": patient_id,
                        "priority": "normal",
                    },
                )
            )
        steps.append(_text_step("Done — you're now booked for the new time."))
        return (
            GroundTruth(
                expected_outcome="booked",
                should_escalate=not customer.has_cancel_tool,
                expected_tools=tools,
                expected_appointment_type=appt,
                notes="Reschedule = search + book (+ task to cancel old when needed).",
            ),
            steps,
        )

    if intent == "eligibility":
        return (
            GroundTruth(
                expected_outcome="info_provided",
                should_escalate=False,
                expected_tools=["check_eligibility"],
                notes="Eligibility lookup before any booking.",
            ),
            [
                _tool_step("check_eligibility", {"patient_id": patient_id}),
                _text_step("Your coverage looks active. Would you like to book now?"),
            ],
        )

    # FAQ — pure rules answer, no tool calls.
    return (
        GroundTruth(
            expected_outcome="info_provided",
            should_escalate=False,
            expected_tools=[],
            notes="FAQ answered from the rules block, no tool call needed.",
        ),
        [_text_step("Here's the information you asked about.")],
    )


def _tool_step(name: str, arguments: dict[str, Any]) -> LLMScriptStep:
    return LLMScriptStep(
        tool_calls=[
            {
                "id": f"c_{name}",
                "name": name,
                "arguments": arguments,
            }
        ]
    )


def _text_step(text: str) -> LLMScriptStep:
    return LLMScriptStep(content=text)
