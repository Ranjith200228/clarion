"""Tests for the tool registry's per-customer enabled_tools enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest
from clarion.config import CustomerConfig, EscalationThresholds
from clarion.tools import (
    BookAppointmentTool,
    CancelAppointmentTool,
    CheckEligibilityTool,
    CreatePmsTaskTool,
    SearchSlotsTool,
    ToolNotEnabledError,
    all_shipped_tool_names,
    available_tools,
    get_tool,
)


def _customer(enabled: list[str], tmp_path: Path) -> CustomerConfig:
    return CustomerConfig(
        customer_id="demo",
        display_name="Demo",
        specialties=["Consult"],
        enabled_tools=enabled,  # type: ignore[arg-type]
        escalation=EscalationThresholds(),
        languages=["en"],
        rules_path=tmp_path / "rules",
        agent_persona="p",
    )


def test_all_shipped_tool_names_matches_phase_4_spec() -> None:
    names = set(all_shipped_tool_names())
    assert names == {
        "search_slots",
        "book_appointment",
        "cancel_appointment",
        "check_eligibility",
        "create_pms_task",
    }


def test_available_tools_returns_only_enabled_ones(tmp_path: Path) -> None:
    customer = _customer(["search_slots", "book_appointment"], tmp_path)
    tools = available_tools(customer)
    classes = [type(t) for t in tools]
    assert classes == [SearchSlotsTool, BookAppointmentTool]


def test_available_tools_preserves_customer_order(tmp_path: Path) -> None:
    """Two customers enable the same tools in different orders — the
    registry must return each customer's configured order so an agent
    that uses the first tool by convention behaves differently per
    customer if the operator wants it to."""
    a = _customer(["search_slots", "check_eligibility"], tmp_path)
    b = _customer(["check_eligibility", "search_slots"], tmp_path)
    assert [type(t) for t in available_tools(a)] == [SearchSlotsTool, CheckEligibilityTool]
    assert [type(t) for t in available_tools(b)] == [CheckEligibilityTool, SearchSlotsTool]


def test_get_tool_returns_enabled_tool(tmp_path: Path) -> None:
    customer = _customer(["book_appointment"], tmp_path)
    tool = get_tool("book_appointment", customer)
    assert isinstance(tool, BookAppointmentTool)


def test_get_tool_disabled_raises_actionable_error(tmp_path: Path) -> None:
    customer = _customer(["search_slots"], tmp_path)
    with pytest.raises(ToolNotEnabledError) as exc:
        get_tool("cancel_appointment", customer)
    msg = str(exc.value)
    assert "cancel_appointment" in msg
    assert "demo" in msg
    assert "search_slots" in msg  # error tells the operator what IS enabled


def test_orthopedics_yaml_cannot_get_cancel_tool() -> None:
    """End-to-end: the real configs/orthopedics.yaml dropped cancel_appointment;
    the registry must honor that."""
    from clarion.config import Settings, load_customer

    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_customer(
        "orthopedics",
        settings=Settings(
            customer="orthopedics",
            config_dir=repo_root / "configs",
            data_dir=repo_root / "data",
        ),
    )
    # Sanity: yaml definitely lacks it.
    assert "cancel_appointment" not in cfg.enabled_tools

    with pytest.raises(ToolNotEnabledError):
        get_tool("cancel_appointment", cfg)

    enabled_classes = {type(t) for t in available_tools(cfg)}
    # The four orthopedics enables.
    assert enabled_classes == {
        SearchSlotsTool,
        BookAppointmentTool,
        CheckEligibilityTool,
        CreatePmsTaskTool,
    }


def test_ophthalmology_yaml_gets_all_five_tools() -> None:
    from clarion.config import Settings, load_customer

    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_customer(
        "ophthalmology",
        settings=Settings(
            customer="ophthalmology",
            config_dir=repo_root / "configs",
            data_dir=repo_root / "data",
        ),
    )
    classes = {type(t) for t in available_tools(cfg)}
    assert classes == {
        SearchSlotsTool,
        BookAppointmentTool,
        CancelAppointmentTool,
        CheckEligibilityTool,
        CreatePmsTaskTool,
    }
