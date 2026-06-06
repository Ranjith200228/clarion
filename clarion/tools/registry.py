"""Tool registry — the only way the agent gets a tool.

The platform ships five tools. Each customer enables a subset via the
``enabled_tools`` field on their ``CustomerConfig``. The registry is how
this multi-tenancy is enforced in code:

* ``get_tool(name, customer)`` returns the tool instance, or raises
  ``ToolNotEnabledError`` if the customer hasn't enabled it.
* ``available_tools(customer)`` lists the tool instances the agent may call.

There is no global "list of tools the agent can see" — only the customer's
subset. The agent (Phase 5) calls ``available_tools`` once at startup; the
LLM never learns about a tool the customer disabled.
"""

from __future__ import annotations

from typing import Any

from clarion.config import CustomerConfig
from clarion.tools.base import Tool, ToolError
from clarion.tools.book_appointment import BookAppointmentTool
from clarion.tools.cancel_appointment import CancelAppointmentTool
from clarion.tools.check_eligibility import CheckEligibilityTool
from clarion.tools.create_pms_task import CreatePmsTaskTool
from clarion.tools.search_slots import SearchSlotsTool

# Every shipping tool, keyed by its name. Keys must match the ``ToolName``
# Literal in ``clarion.config.schema``; if a new tool is added, both places
# have to agree (the registry test enforces this).
_REGISTRY: dict[str, Tool[Any, Any]] = {
    "search_slots": SearchSlotsTool(),
    "book_appointment": BookAppointmentTool(),
    "cancel_appointment": CancelAppointmentTool(),
    "check_eligibility": CheckEligibilityTool(),
    "create_pms_task": CreatePmsTaskTool(),
}


class ToolNotEnabledError(ToolError):
    """Raised when the agent tries to use a tool the customer hasn't enabled."""


def get_tool(name: str, customer: CustomerConfig) -> Tool[Any, Any]:
    """Return a tool by name, enforcing the customer's enabled_tools list.

    Raises:
        ToolNotEnabledError: ``name`` isn't in customer.enabled_tools, or
            no tool by that name is shipped at all.
    """
    if name not in customer.enabled_tools:
        raise ToolNotEnabledError(
            f"Tool '{name}' is not enabled for customer '{customer.customer_id}'. "
            f"Enabled: {sorted(customer.enabled_tools)}"
        )
    return _REGISTRY[name]


def available_tools(customer: CustomerConfig) -> list[Tool[Any, Any]]:
    """Tools this customer has enabled, in the configured order."""
    return [_REGISTRY[name] for name in customer.enabled_tools]


def all_shipped_tool_names() -> list[str]:
    """Every tool the platform ships, regardless of customer."""
    return list(_REGISTRY.keys())
