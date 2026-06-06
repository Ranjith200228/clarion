"""Tests for the system prompt builder."""

from __future__ import annotations

from clarion.agents.prompt import PromptContext, build_system_prompt
from clarion.config import CustomerConfig


def test_prompt_includes_display_name_and_persona(minimal_config: CustomerConfig) -> None:
    ctx = PromptContext(customer=minimal_config, retriever=None)
    text = build_system_prompt(ctx)
    assert minimal_config.display_name in text
    assert minimal_config.agent_persona.strip()[:30] in text


def test_prompt_includes_hard_operating_contract(minimal_config: CustomerConfig) -> None:
    text = build_system_prompt(PromptContext(customer=minimal_config, retriever=None))
    # Spot-check the must-have rules.
    assert "never give clinical advice" in text
    assert "call 911" in text
    assert "create_pms_task" in text
    assert "confirm patient identity" in text.lower() or "Confirm patient identity" in text


def test_prompt_includes_escalation_thresholds_from_config(
    minimal_config: CustomerConfig,
) -> None:
    text = build_system_prompt(PromptContext(customer=minimal_config, retriever=None))
    e = minimal_config.escalation
    assert f"{e.low_confidence:.2f}" in text
    assert str(e.max_clarifications) in text
    assert f"{e.frustration:.2f}" in text


def test_prompt_omits_rules_block_when_no_retriever(
    minimal_config: CustomerConfig,
) -> None:
    text = build_system_prompt(
        PromptContext(customer=minimal_config, retriever=None),
        user_message="anything",
    )
    assert "Practice rules relevant to this turn" not in text


def test_prompt_omits_rules_block_when_no_user_message_yet(
    minimal_config: CustomerConfig,
) -> None:
    # No user message yet (greeting turn) — even if we had a retriever, the
    # rules block should be skipped.
    text = build_system_prompt(
        PromptContext(customer=minimal_config, retriever=None),
        user_message=None,
    )
    assert "Practice rules relevant to this turn" not in text
