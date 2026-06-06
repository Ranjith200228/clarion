"""Mocked PMS tools the agent calls.

The agent never imports tool classes directly — it goes through the
registry so per-customer enabled_tools is always enforced::

    from clarion.tools import available_tools, get_tool

    for tool in available_tools(customer_config):
        ...  # advertise to the LLM

    tool = get_tool("book_appointment", customer_config)
    output = tool.run(input_model, ctx)
"""

from clarion.tools.base import (
    Tool,
    ToolContext,
    ToolError,
    run_with_retry,
)
from clarion.tools.book_appointment import BookAppointmentTool
from clarion.tools.cancel_appointment import CancelAppointmentTool
from clarion.tools.check_eligibility import CheckEligibilityTool
from clarion.tools.create_pms_task import CreatePmsTaskTool
from clarion.tools.registry import (
    ToolNotEnabledError,
    all_shipped_tool_names,
    available_tools,
    get_tool,
)
from clarion.tools.search_slots import SearchSlotsTool

__all__ = [
    "BookAppointmentTool",
    "CancelAppointmentTool",
    "CheckEligibilityTool",
    "CreatePmsTaskTool",
    "SearchSlotsTool",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolNotEnabledError",
    "all_shipped_tool_names",
    "available_tools",
    "get_tool",
    "run_with_retry",
]
