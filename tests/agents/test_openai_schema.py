"""Tests for Pydantic → OpenAI tool schema conversion."""

from __future__ import annotations

from clarion.agents.openai_schema import tool_to_spec, tools_to_specs
from clarion.tools import (
    BookAppointmentTool,
    CreatePmsTaskTool,
    SearchSlotsTool,
)


def test_spec_has_name_description_parameters() -> None:
    spec = tool_to_spec(SearchSlotsTool())
    assert spec.name == "search_slots"
    assert spec.description
    assert spec.parameters["type"] == "object"


def test_parameters_include_required_fields() -> None:
    spec = tool_to_spec(SearchSlotsTool())
    required = set(spec.parameters["required"])
    assert {"appointment_type", "on_or_after"} <= required


def test_optional_fields_have_defaults_not_in_required() -> None:
    spec = tool_to_spec(SearchSlotsTool())
    required = set(spec.parameters["required"])
    assert "limit" not in required
    assert spec.parameters["properties"]["limit"]["default"] == 5


def test_additional_properties_locked_down_on_every_object() -> None:
    spec = tool_to_spec(BookAppointmentTool())
    assert spec.parameters["additionalProperties"] is False


def test_title_keys_stripped() -> None:
    spec = tool_to_spec(CreatePmsTaskTool())
    # Walk every dict — there must be no remaining "title" keys anywhere.
    stack: list[object] = [spec.parameters]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            assert "title" not in node, f"unexpected title key in {node}"
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def test_tools_to_specs_preserves_order() -> None:
    tools = [SearchSlotsTool(), CreatePmsTaskTool()]
    specs = tools_to_specs(tools)
    assert [s.name for s in specs] == ["search_slots", "create_pms_task"]


def test_priority_enum_present_for_create_pms_task() -> None:
    spec = tool_to_spec(CreatePmsTaskTool())
    priority = spec.parameters["properties"]["priority"]
    assert priority["enum"] == ["normal", "urgent"]
