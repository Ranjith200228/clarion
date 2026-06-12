"""End-to-end tests for the MultiAgentRunner / StateGraph wiring.

These exercise the full router -> specialist -> supervisor traversal
with FakeLLM for the LLM-backed specialists (no real OpenAI calls)
and the HeuristicIntentRouter so routing stays deterministic. The
goal is to prove the graph topology is correct: control reaches the
right specialist, the supervisor sees the result, and the runner
returns the expected assistant text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from clarion.agents.llm import FakeLLM, LLMResponse
from clarion.config import Settings, load_customer
from clarion.multiagent import MultiAgentRunner
from clarion.pipelines.structured import StructuredStore
from clarion.schemas import AvailabilitySlot, EligibilityRecord, Provider
from clarion.tools.base import ToolContext

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
SEEDS_DIR = REPO_ROOT / "data" / "seeds"


@pytest.fixture
def ophthalmology_runner(tmp_path: Path) -> tuple[MultiAgentRunner, FakeLLM]:
    """Build a MultiAgentRunner for the ophthalmology customer with a
    seeded StructuredStore and HeuristicIntentRouter (so routing is
    deterministic; LLM calls only happen inside specialists)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    store = StructuredStore.for_customer("ophthalmology", data_dir)
    payload = json.loads((SEEDS_DIR / "ophthalmology.json").read_text(encoding="utf-8"))
    for p in payload["providers"]:
        store.upsert_provider(Provider(**p))
    for s in payload["availability"]:
        store.upsert_slot(AvailabilitySlot(**s))
    for e in payload["eligibility"]:
        store.upsert_eligibility(EligibilityRecord(**e))

    settings = Settings(customer="ophthalmology", config_dir=CONFIGS_DIR, data_dir=data_dir)
    customer = load_customer("ophthalmology", settings=settings)
    fake = FakeLLM(responses=[])

    runner = MultiAgentRunner(
        customer=customer,
        llm=fake,
        ctx=ToolContext(customer=customer, structured=store),
        use_heuristic_router=True,
    )
    return runner, fake


def test_emergency_intent_bypasses_llm_and_emits_911(
    ophthalmology_runner: tuple[MultiAgentRunner, FakeLLM],
) -> None:
    runner, fake = ophthalmology_runner
    # No FakeLLM responses scripted — emergency path must not call the LLM.
    reply = runner.chat("I'm having chest pain — what should I do?")
    assert "911" in reply
    # Emergency specialist short-circuits AND the heuristic router doesn't
    # touch the LLM. Total LLM calls must be zero.
    assert fake.turns_consumed == 0
    # Transcript was advanced — user turn + assistant turn appended.
    assert len(runner.transcript) == 2
    assert runner.transcript[0].role == "user"
    assert runner.transcript[1].role == "assistant"
    assert "911" in (runner.transcript[1].content or "")


def test_info_intent_runs_specialist_loop(
    ophthalmology_runner: tuple[MultiAgentRunner, FakeLLM],
) -> None:
    runner, fake = ophthalmology_runner
    # Info specialist's ReAct loop runs once; one assistant turn with no
    # tool calls -> graph exits via supervisor FINISH.
    fake.responses.append(
        LLMResponse(content="We're open 8am to 5pm Monday through Friday.", tool_calls=())
    )
    reply = runner.chat("What are your hours?")
    assert "8am" in reply
    assert fake.turns_consumed == 1


def test_multi_turn_conversation_preserves_transcript(
    ophthalmology_runner: tuple[MultiAgentRunner, FakeLLM],
) -> None:
    runner, fake = ophthalmology_runner
    fake.responses.extend(
        [
            LLMResponse(content="We're open 8am to 5pm.", tool_calls=()),
            LLMResponse(content="Yes, we accept Aetna.", tool_calls=()),
        ]
    )
    runner.chat("What are your hours?")
    runner.chat("Do you take Aetna?")
    # 4 turns: 2 user + 2 assistant.
    assert len(runner.transcript) == 4
    assert runner.transcript[0].content == "What are your hours?"
    assert runner.transcript[2].content == "Do you take Aetna?"
    # Each specialist got the running transcript appended; verify by checking
    # the second LLM call saw the first turn's content.
    second_call_messages, _ = fake.calls[1]
    transcripts_seen = [m.content for m in second_call_messages if m.content]
    assert any("hours" in (c or "").lower() for c in transcripts_seen)
