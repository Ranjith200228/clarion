"""Tests for the demo-mode LLM fallback in api.sessions.

When ``OPENAI_API_KEY`` is unset (an HF Space that hasn't been
configured with a secret yet), ``default_llm_factory`` must return a
``DemoModeLLM`` so visitors see a clear "set the key to enable"
bubble instead of a 500. Production sets the key and never sees the
fallback.
"""

from __future__ import annotations

import pytest

from api.sessions import DEMO_MODE_REPLY, DemoModeLLM, default_llm_factory


def test_demo_mode_llm_emits_canned_reply() -> None:
    llm = DemoModeLLM()
    resp = llm.complete(messages=[], tools=None)
    assert resp.content == DEMO_MODE_REPLY
    assert resp.tool_calls == ()
    # No tokens, no model — accounting layer treats it as free.
    assert resp.usage.model == "demo-mode"
    assert resp.usage.input_tokens == 0
    assert resp.usage.output_tokens == 0


def test_factory_returns_demo_mode_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = default_llm_factory()
    assert isinstance(llm, DemoModeLLM)


def test_factory_returns_openai_client_when_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a key IS present, the factory must build an OpenAIClient.

    OpenAIClient's ``__init__`` instantiates the openai SDK — we
    don't want to touch the network, but we DO want to confirm the
    factory branched correctly. The cheapest way is to assert the
    return type is NOT DemoModeLLM (we don't import OpenAIClient
    here because that would itself trigger the openai import).
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    llm = default_llm_factory()
    assert not isinstance(llm, DemoModeLLM)
