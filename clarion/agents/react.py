"""ReAct loop and tool dispatch.

The loop is intentionally small:

    for _ in range(max_steps):
        response = llm.complete(messages, tools=specs)
        messages.append(assistant_msg_from(response))
        if not response.tool_calls:
            return response.content or ""
        for call in response.tool_calls:
            result = dispatcher.dispatch(call)
            messages.append(tool_msg(call, result))

Everything below is plumbing — validating the LLM's tool args against
the tool's Pydantic input, catching exceptions and surfacing them as
ok=False replies, capping the steps so a broken LLM script can't burn
through a budget.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from clarion.agents.llm import (
    LLMClient,
    LLMResponse,
    Message,
    ToolCall,
    json_dumps,
)
from clarion.agents.openai_schema import tools_to_specs
from clarion.config import CustomerConfig
from clarion.observability import Tracer, cost_usd
from clarion.schemas.tools import ToolOutput
from clarion.tools.base import Tool, ToolContext, ToolError
from clarion.tools.registry import (
    ToolNotEnabledError,
    available_tools,
    get_tool,
)

log = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 8


@dataclass
class StepRecord:
    """One LLM call + its tool-call fanout. Used for tracing + tests."""

    assistant_content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ReactResult:
    final_text: str
    steps: list[StepRecord]
    stopped_for_max_steps: bool = False


class ToolDispatcher:
    """Run validated tool calls and surface failures as ok=False outputs.

    Holds the per-conversation ``ToolContext`` and an optional ``Tracer``
    so individual call sites don't have to thread either through.
    """

    def __init__(
        self,
        customer: CustomerConfig,
        ctx: ToolContext,
        *,
        tracer: Tracer | None = None,
    ) -> None:
        self._customer = customer
        self._ctx = ctx
        self._tracer = tracer

    def dispatch(self, call: ToolCall) -> dict[str, Any]:
        """Execute one tool call. Always returns a dict for the LLM."""
        if self._tracer is None:
            return self._dispatch_inner(call)
        with self._tracer.span(
            f"tool.{call.name}",
            tool=call.name,
            arguments_keys=list(call.arguments.keys()),
        ) as span:
            result = self._dispatch_inner(call)
            span.set("ok", bool(result.get("ok")))
            if not result.get("ok") and result.get("error"):
                span.set("error", str(result["error"])[:200])
            return result

    def _dispatch_inner(self, call: ToolCall) -> dict[str, Any]:
        try:
            tool = get_tool(call.name, self._customer)
        except ToolNotEnabledError as e:
            return {"ok": False, "error": str(e)}

        try:
            input_obj = _validate_input(tool, call.arguments)
        except ValidationError as e:
            return {"ok": False, "error": f"invalid arguments: {e.errors()}"}

        try:
            output = tool.run(input_obj, self._ctx)
        except ToolError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            log.exception("Unexpected error in tool %s", call.name)
            return {"ok": False, "error": f"unexpected error: {e}"}

        return _output_to_dict(output)


def react_loop(
    *,
    llm: LLMClient,
    messages: list[Message],
    customer: CustomerConfig,
    ctx: ToolContext,
    max_steps: int = DEFAULT_MAX_STEPS,
    tracer: Tracer | None = None,
) -> ReactResult:
    """Drive the loop until the LLM stops asking for tool calls.

    ``messages`` is mutated in place — callers (e.g. ``Agent``) keep the
    full transcript on their side. If a ``tracer`` is provided, each
    iteration emits a ``react.step`` span containing nested
    ``llm.complete`` and ``tool.<name>`` spans annotated with tokens,
    cost, and per-tool ok flags.
    """
    specs = tools_to_specs(available_tools(customer))
    dispatcher = ToolDispatcher(customer, ctx, tracer=tracer)
    steps: list[StepRecord] = []

    for step_i in range(max_steps):
        step_cm = (
            tracer.span("react.step", step_index=step_i) if tracer is not None else _null_span()
        )
        with step_cm:
            response = _complete_with_span(llm, messages, specs, tracer)
            step = StepRecord(
                assistant_content=response.content,
                tool_calls=list(response.tool_calls),
            )
            messages.append(_assistant_message(response))

            if not response.tool_calls:
                steps.append(step)
                return ReactResult(final_text=response.content or "", steps=steps)

            for call in response.tool_calls:
                # Ensure every call has a stable id (some LLMs emit empty ones).
                call_id = call.id or f"call_{uuid.uuid4().hex[:8]}"
                normalized = ToolCall(id=call_id, name=call.name, arguments=call.arguments)
                result = dispatcher.dispatch(normalized)
                step.tool_results.append(result)
                messages.append(
                    Message.tool(
                        call_id=call_id,
                        name=call.name,
                        content=json_dumps(result),
                    )
                )
            steps.append(step)
            log.debug("react step %d done (%d tool calls)", step_i, len(response.tool_calls))

    # Ran out of steps without the LLM producing a final text turn.
    return ReactResult(
        final_text=(
            "I'm having trouble completing that — let me have a teammate " "call you back."
        ),
        steps=steps,
        stopped_for_max_steps=True,
    )


# ---------- helpers ----------


def _complete_with_span(
    llm: LLMClient,
    messages: list[Message],
    specs: list[Any],
    tracer: Tracer | None,
) -> LLMResponse:
    """Call ``llm.complete`` inside an ``llm.complete`` span when tracing is on."""
    if tracer is None:
        return llm.complete(messages, tools=specs)
    with tracer.span("llm.complete", advertised_tools=len(specs)) as span:
        response = llm.complete(messages, tools=specs)
        usage = response.usage
        cost = cost_usd(
            usage.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        span.update(
            {
                "model": usage.model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": cost,
                "tool_calls_count": len(response.tool_calls),
            }
        )
        return response


from collections.abc import Iterator  # noqa: E402
from contextlib import contextmanager  # noqa: E402  (helper-only import)


@contextmanager
def _null_span() -> Iterator[None]:
    """No-op context manager used when no tracer is provided."""
    yield None


def _assistant_message(response: LLMResponse) -> Message:
    return Message.assistant(text=response.content, tool_calls=response.tool_calls)


def _validate_input(tool: Tool[Any, Any], arguments: dict[str, Any]) -> Any:
    """Re-validate tool args through the tool's Pydantic input model.

    Defensive — the LLM may emit JSON that *looks* right but is missing
    a required field. We surface that as an LLM-visible error rather
    than letting it crash the tool.
    """
    input_model = tool.input_model  # type: ignore[attr-defined]
    return input_model(**arguments)


def _output_to_dict(output: Any) -> dict[str, Any]:
    if isinstance(output, ToolOutput):
        # ``mode="json"`` makes dates / Enums serialize as strings the LLM
        # can read directly.
        return output.model_dump(mode="json")
    if isinstance(output, dict):
        return output
    return {"ok": True, "data": output}


__all__ = [
    "DEFAULT_MAX_STEPS",
    "ReactResult",
    "StepRecord",
    "ToolDispatcher",
    "react_loop",
]
