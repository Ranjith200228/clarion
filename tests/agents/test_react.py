"""Unit tests for the ReAct loop and ToolDispatcher."""

from __future__ import annotations

from datetime import date

from clarion.agents.llm import (
    FakeLLM,
    LLMResponse,
    Message,
    ToolCall,
)
from clarion.agents.react import (
    DEFAULT_MAX_STEPS,
    ToolDispatcher,
    react_loop,
)
from clarion.config import CustomerConfig
from clarion.tools.base import ToolContext


def _user_seed() -> list[Message]:
    return [Message.system("system"), Message.user("hi")]


# ---------- ToolDispatcher ----------


def test_dispatch_runs_valid_search_slots(minimal_ctx: ToolContext) -> None:
    dispatcher = ToolDispatcher(minimal_ctx.customer, minimal_ctx)
    out = dispatcher.dispatch(
        ToolCall(
            id="c1",
            name="search_slots",
            arguments={
                "appointment_type": "Consult",
                "on_or_after": date(2026, 6, 1).isoformat(),
            },
        )
    )
    assert out["ok"] is True
    assert isinstance(out["slots"], list)
    assert out["slots"][0]["slot_id"] == "slot_demo_1"


def test_dispatch_returns_validation_error_on_bad_args(
    minimal_ctx: ToolContext,
) -> None:
    dispatcher = ToolDispatcher(minimal_ctx.customer, minimal_ctx)
    out = dispatcher.dispatch(
        ToolCall(id="c1", name="search_slots", arguments={"appointment_type": ""}),
    )
    assert out["ok"] is False
    assert "invalid arguments" in out["error"]


def test_dispatch_returns_not_enabled_error_when_tool_disabled(
    minimal_config: CustomerConfig,
    minimal_ctx: ToolContext,
) -> None:
    # Build a customer that DOES NOT enable cancel_appointment.
    restricted = minimal_config.model_copy(update={"enabled_tools": ["search_slots"]})
    dispatcher = ToolDispatcher(restricted, minimal_ctx)
    out = dispatcher.dispatch(
        ToolCall(id="c1", name="cancel_appointment", arguments={"appointment_id": "x"}),
    )
    assert out["ok"] is False
    assert "not enabled" in out["error"]


# ---------- react_loop happy path ----------


def test_loop_returns_immediately_when_llm_emits_text(
    minimal_config: CustomerConfig, minimal_ctx: ToolContext
) -> None:
    llm = FakeLLM(responses=[LLMResponse(content="hello!")])
    messages = _user_seed()
    result = react_loop(
        llm=llm,
        messages=messages,
        customer=minimal_config,
        ctx=minimal_ctx,
    )
    assert result.final_text == "hello!"
    assert llm.turns_consumed == 1
    assert len(result.steps) == 1
    assert result.steps[0].tool_calls == []


def test_loop_runs_tool_then_returns_text(
    minimal_config: CustomerConfig, minimal_ctx: ToolContext
) -> None:
    llm = FakeLLM(
        responses=[
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="search_slots",
                        arguments={
                            "appointment_type": "Consult",
                            "on_or_after": date(2026, 6, 1).isoformat(),
                        },
                    ),
                )
            ),
            LLMResponse(content="I have one slot for you."),
        ]
    )
    messages = _user_seed()
    result = react_loop(
        llm=llm,
        messages=messages,
        customer=minimal_config,
        ctx=minimal_ctx,
    )
    assert result.final_text == "I have one slot for you."
    assert llm.turns_consumed == 2
    # Step 1: one tool call with a result. Step 2: text reply.
    assert len(result.steps) == 2
    assert result.steps[0].tool_calls[0].name == "search_slots"
    assert result.steps[0].tool_results[0]["ok"] is True


def test_loop_passes_only_enabled_tools_to_llm(
    minimal_config: CustomerConfig, minimal_ctx: ToolContext
) -> None:
    restricted = minimal_config.model_copy(update={"enabled_tools": ["search_slots"]})
    llm = FakeLLM(responses=[LLMResponse(content="ok")])
    react_loop(
        llm=llm,
        messages=_user_seed(),
        customer=restricted,
        ctx=minimal_ctx,
    )
    _, tools = llm.calls[-1]
    assert {t.name for t in tools} == {"search_slots"}


def test_loop_normalizes_blank_tool_call_ids(
    minimal_config: CustomerConfig, minimal_ctx: ToolContext
) -> None:
    """Some LLMs occasionally emit '' as the call id — we must invent one
    so the role=tool message has a stable id the next assistant turn can
    reference."""
    llm = FakeLLM(
        responses=[
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        id="",
                        name="search_slots",
                        arguments={
                            "appointment_type": "Consult",
                            "on_or_after": date(2026, 6, 1).isoformat(),
                        },
                    ),
                )
            ),
            LLMResponse(content="done"),
        ]
    )
    messages = _user_seed()
    react_loop(
        llm=llm,
        messages=messages,
        customer=minimal_config,
        ctx=minimal_ctx,
    )
    tool_msgs = [m for m in messages if m.role == "tool"]
    assert tool_msgs
    assert tool_msgs[0].tool_call_id  # non-empty


# ---------- safety: max_steps ----------


def test_loop_returns_fallback_when_max_steps_hit(
    minimal_config: CustomerConfig, minimal_ctx: ToolContext
) -> None:
    # LLM never emits text — only tool calls — so we exhaust steps.
    looping = LLMResponse(
        tool_calls=(
            ToolCall(
                id="c1",
                name="search_slots",
                arguments={
                    "appointment_type": "Consult",
                    "on_or_after": date(2026, 6, 1).isoformat(),
                },
            ),
        )
    )
    llm = FakeLLM(responses=[looping] * 3)
    result = react_loop(
        llm=llm,
        messages=_user_seed(),
        customer=minimal_config,
        ctx=minimal_ctx,
        max_steps=3,
    )
    assert result.stopped_for_max_steps is True
    assert "teammate" in result.final_text.lower()


def test_default_max_steps_is_sane() -> None:
    assert 1 <= DEFAULT_MAX_STEPS <= 16
