"""Smoke tests for the surviving Phase G tabs.

Phase G retired the read-only v1 tabs (Quality Metrics, Escalations,
Trace Explorer) — their content now lives in the v2 views and the
source files have been removed. Live Agent + Voice Agent remain as
the only interactive surfaces; their state-helpers are exercised
here.

The v2 view renderers have their own dedicated test modules
(``test_mission_control.py``, ``test_sentinel_ops.py``, etc.).
"""

from __future__ import annotations

from gradio_app import tab_live_agent

# ---------- live agent state ----------


def test_live_agent_set_customer_resets_running_totals() -> None:
    st = tab_live_agent.LiveAgentState(
        customer_id="ophthalmology",
        conversation_id="conv_old",
        running_cost_usd=0.01,
        running_input_tokens=100,
        running_output_tokens=50,
    )
    new = tab_live_agent.set_customer(st, "orthopedics")
    assert new.customer_id == "orthopedics"
    assert new.conversation_id is None
    assert new.running_cost_usd == 0.0
    assert new.running_input_tokens == 0
    assert new.running_output_tokens == 0
