"""Convert a Clarion tool into the OpenAI function-calling schema.

OpenAI's chat-completions tool format wants::

    {
      "type": "function",
      "function": {
        "name": "<tool name>",
        "description": "<one-liner>",
        "parameters": <JSON Schema for the input>
      }
    }

Pydantic v2 gives us the parameters schema via ``Model.model_json_schema``.
We do small post-processing:

* strip Pydantic's ``title`` keys (noise in the LLM's prompt)
* set ``additionalProperties=false`` (matches ``extra="forbid"`` on the
  input models, and prevents the LLM from inventing fields)
* inline ``$ref`` definitions so the schema is self-contained
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from clarion.agents.llm import ToolSpec
from clarion.tools.base import Tool


def tool_to_spec(tool: Tool[Any, Any]) -> ToolSpec:
    """Build a ``ToolSpec`` from a Tool instance.

    The description is the tool's input-model docstring if any, falling
    back to the tool class docstring, falling back to the tool's name.
    Description text is what the LLM uses to decide whether to call the
    tool, so we want it short and intent-focused.
    """
    # input_model is runtime metadata each tool class sets (see Phase 4
    # commit 9 for why it's not on the Protocol contract).
    input_model: type[BaseModel] = tool.input_model  # type: ignore[attr-defined]
    if input_model is None or not issubclass(input_model, BaseModel):
        raise TypeError(
            f"Tool {tool.name!r} is missing a Pydantic ``input_model`` attribute"
        )
    description = (
        (tool.__class__.__doc__ or "").strip()
        or (getattr(input_model, "__doc__", "") or "").strip()
        or tool.name
    )
    # Keep just the first paragraph — long docstrings burn tokens.
    description = description.split("\n\n", 1)[0].strip()

    raw_schema = input_model.model_json_schema()
    cleaned = _clean_schema(raw_schema)
    return ToolSpec(name=tool.name, description=description, parameters=cleaned)


def tools_to_specs(tools: list[Tool[Any, Any]]) -> list[ToolSpec]:
    return [tool_to_spec(t) for t in tools]


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline $refs, strip titles, enforce additionalProperties=false."""
    definitions = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            # Inline $ref.
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = definitions[ref.removeprefix("#/$defs/")]
                return walk(target)
            out: dict[str, Any] = {}
            for k, v in node.items():
                if k == "title":
                    continue  # noise
                out[k] = walk(v)
            # Tighten object schemas so the LLM cannot smuggle keys.
            if out.get("type") == "object" and "additionalProperties" not in out:
                out["additionalProperties"] = False
            return out
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    cleaned: dict[str, Any] = walk(schema)
    return cleaned
