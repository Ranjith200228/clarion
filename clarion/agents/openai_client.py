"""OpenAI implementation of the LLMClient protocol.

Used in production when ``OPENAI_API_KEY`` is set. Tests use ``FakeLLM``
instead so CI never needs a key. Both implement the same ``complete``
contract; the agent code can't tell which it's talking to.

Model defaults to ``gpt-4o-mini`` (the spec's choice — cheap, supports
tool calling). Override with ``CLARION_MODEL`` env var or the constructor
arg if you want to A/B with another model later.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from clarion.agents.llm import (
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
)

log = logging.getLogger(__name__)


DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIClient:
    """Wraps the OpenAI SDK in the LLMClient Protocol shape."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        # Import lazily so unit tests don't pay the openai import cost.
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OpenAIClient requires OPENAI_API_KEY (env or constructor arg).")
        self._client = OpenAI(api_key=key)
        self._model = model or os.environ.get("CLARION_MODEL") or DEFAULT_MODEL
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        payload_messages = [_to_openai_message(m) for m in messages]
        payload_tools = [_to_openai_tool(t) for t in (tools or [])]

        log.debug(
            "openai.complete model=%s msgs=%d tools=%d",
            self._model,
            len(payload_messages),
            len(payload_tools),
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=payload_messages,  # type: ignore[arg-type]
            tools=payload_tools or None,  # type: ignore[arg-type]
            temperature=self._temperature,
        )
        choice = resp.choices[0]
        msg = choice.message
        return LLMResponse(
            content=msg.content,
            tool_calls=tuple(_from_openai_tool_call(tc) for tc in (msg.tool_calls or [])),
        )


# ---------- adapters ----------


def _to_openai_message(m: Message) -> dict[str, Any]:
    """Convert Clarion Message → OpenAI chat message dict."""
    base: dict[str, Any] = {"role": m.role}
    if m.role == "assistant":
        if m.content is not None:
            base["content"] = m.content
        if m.tool_calls:
            base["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in m.tool_calls
            ]
        return base
    if m.role == "tool":
        # OpenAI requires both tool_call_id and content for role=tool.
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id,
            "content": m.content or "",
        }
    # system / user
    return {"role": m.role, "content": m.content or ""}


def _to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _from_openai_tool_call(tc: Any) -> ToolCall:
    """OpenAI returns ``arguments`` as a JSON string — parse it."""
    try:
        arguments = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        log.warning("openai tool_call had non-JSON arguments: %r", tc.function.arguments)
        arguments = {}
    return ToolCall(id=tc.id, name=tc.function.name, arguments=arguments)
