"""LLM abstraction the ReAct loop sits on top of.

Why an abstraction? Two reasons:

1. Tests must run without an OpenAI key. ``FakeLLM`` plays back canned
   responses so the loop is exercised deterministically in CI.
2. The agent code shouldn't care which model it's talking to — Phase 16's
   LangGraph refactor will reuse the same LLM Protocol.

The Protocol mirrors OpenAI's tool-calling shape (assistant messages can
have ``content`` and/or ``tool_calls``; tool results are dicts that look
like OpenAI tool messages). That keeps the OpenAI adapter trivial.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

# Roles match OpenAI's chat-completions format exactly.
Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """One function call the LLM asked us to make.

    ``arguments`` is the parsed dict. Callers convert to/from JSON at the
    LLM boundary; inside the agent we keep it structured.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    """One entry in the conversation transcript.

    Mirrors OpenAI's chat-message shape but the rest of the codebase
    doesn't need to know that.
    """

    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None  # set on role="tool" replies
    name: str | None = None  # tool name on role="tool" replies

    @staticmethod
    def system(text: str) -> Message:
        return Message(role="system", content=text)

    @staticmethod
    def user(text: str) -> Message:
        return Message(role="user", content=text)

    @staticmethod
    def assistant(
        text: str | None = None,
        *,
        tool_calls: tuple[ToolCall, ...] = (),
    ) -> Message:
        return Message(role="assistant", content=text, tool_calls=tool_calls)

    @staticmethod
    def tool(*, call_id: str, name: str, content: str) -> Message:
        return Message(role="tool", content=content, tool_call_id=call_id, name=name)


@dataclass(frozen=True)
class LLMUsage:
    """Token + model accounting for one LLM call.

    The model field is what the observability layer (Phase 7) reads to
    look up pricing. Tokens come straight from the provider's response
    when available; FakeLLM defaults to zero so unit tests don't have to
    estimate counts.
    """

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    """One assistant turn — either free-text content, tool calls, or both.

    Mirrors what OpenAI's ``chat.completions.create`` returns (we ignore
    streaming for now; Phase 17 voice can add it). ``usage`` carries the
    token counts + model that Phase 7's observability layer feeds into
    the cost calculator.
    """

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: LLMUsage = field(default_factory=LLMUsage)


@dataclass(frozen=True)
class ToolSpec:
    """A tool advertised to the LLM. ``parameters`` is a JSON Schema dict."""

    name: str
    description: str
    parameters: dict[str, Any]


class LLMClient(Protocol):
    """Anything that can take a transcript + tool list and return one turn."""

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse: ...


# ---------- FakeLLM for tests ----------


@dataclass
class FakeLLM:
    """Deterministic stand-in for tests.

    Initialize with a list of canned ``LLMResponse``s. Each call to
    ``complete`` pops the next one. The transcript and tool list passed in
    are recorded on ``calls`` so tests can assert *what the agent told
    the LLM*, not just what the agent did with the reply.
    """

    responses: list[LLMResponse]
    calls: list[tuple[list[Message], list[ToolSpec]]] = field(default_factory=list)
    _next: int = 0

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        self.calls.append((list(messages), list(tools or [])))
        if self._next >= len(self.responses):
            raise RuntimeError(
                f"FakeLLM ran out of scripted responses after {self._next} turns. "
                f"Add more or check the agent isn't looping."
            )
        resp = self.responses[self._next]
        self._next += 1
        return resp

    @property
    def turns_consumed(self) -> int:
        return self._next


# ---------- helpers ----------


def json_dumps(obj: Any) -> str:
    """Stable JSON dump used in tool replies. Pretty-printing keeps logs
    legible without burning many tokens at MVP scale."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
