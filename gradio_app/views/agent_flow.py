"""Agent Flow — the multi-agent reasoning visualization.

Per the Phase 0 plan: *make the agent reasoning visible. Make the
orchestration visible. Make the intelligence visible.* This view
takes one TraceEntry and renders the full path through the
hierarchical agent graph as a connected diagram.

Layout, top to bottom:

1. **Turn header** — tenant + scenario id + final outcome badge,
   plus a one-line description of the turn driving the flow.
2. **Flow diagram** — six nodes left-to-right
   (PATIENT → ROUTER → SPECIALIST → TOOLS → SENTINEL → RESPONSE)
   connected by arrow-tipped CSS borders. Each node uses the
   ``components.agent_node`` primitive (which already knows about
   the data-state colouring).
3. **Specialist panel** — five mini-cards, one per specialist,
   active one lit in cyan, others greyed. Closes the
   "five specialists exist but only one ran" story.
4. **Tools panel** — the list of tools that fired this turn,
   monospaced.
5. **Trust posture** — a row of chips: escalation_score + reasons
   + judge violations + final outcome.

The view is per-tenant; the app callback re-renders on every
customer dropdown change. A future Phase E or G iteration can add
a scenario-id dropdown to switch turns; for now the default is
the first entry.
"""

from __future__ import annotations

from gradio_app import components as c
from gradio_app.data_sources import AgentFlowSnapshot, FlowNode

# ---------- public ----------


def build_html(flow: AgentFlowSnapshot) -> str:
    if not flow.has_data:
        return empty_html(tenant=flow.tenant)
    return (
        '<div class="clarion-stack" style="gap: 24px;">'
        + c.page_intro(
            title="Agent Flow",
            what=(
                "Live trace through the multi-agent graph that "
                "handled this conversation."
            ),
            quote="Many minds. One conversation.",
        )
        + _turn_header(flow)
        + _flow_diagram(flow)
        + '<div class="clarion-row" style="align-items: stretch; gap: 16px;">'
        + _specialist_panel(flow.chosen_specialist, flow.other_specialists)
        + _tools_panel(flow.tools_called)
        + "</div>"
        + _trust_posture(flow)
        + "</div>"
    )


def empty_html(*, tenant: str = "this customer") -> str:
    body = (
        '<div class="clarion-stack" style="align-items: center; gap: 12px;">'
        '<div style="font-size: var(--fs-lg); color: var(--c-text);">'
        f"No turns to visualize for {_esc(tenant)}."
        "</div>"
        '<div style="font-size: var(--fs-sm); color: var(--c-text-muted);">'
        "The agent-flow diagram needs at least one TraceEntry. Run the harness:"
        "</div>"
        '<div class="clarion-mono" style="padding: 12px 16px; '
        'background: var(--c-bg-subtle); border-radius: var(--r-md);">'
        f"python -m clarion.evaluation.cli run {_esc(tenant).lower()}"
        "</div>"
        "</div>"
    )
    return (
        '<div class="clarion-stack" style="gap: 24px;">'
        + c.page_intro(
            title="Agent Flow",
            what=(
                "Live trace through the multi-agent graph that "
                "handled this conversation."
            ),
            quote="Many minds. One conversation.",
        )
        + c.panel(title="Awaiting Data", body_html=body)
        + "</div>"
    )


# ---------- top header ----------


def _turn_header(flow: AgentFlowSnapshot) -> str:
    """Identifier strip — scenario id + intent + final outcome."""
    outcome_badge = c.status_badge(
        _outcome_severity(flow.final_outcome),
        label=flow.final_outcome.upper(),
    )
    return (
        '<div class="clarion-panel">'
        '<div class="clarion-row" style="justify-content: space-between; gap: 12px;">'
        '<div class="clarion-row" style="gap: 12px;">'
        f'<div style="font-size: var(--fs-md); font-weight: var(--fw-semibold); '
        f'color: var(--c-text-strong);">{_esc(flow.scenario_id)}</div>'
        + c.mono(flow.intent)
        + "</div>"
        f"{outcome_badge}"
        "</div>"
        f'<div style="font-size: var(--fs-sm); color: var(--c-text-muted); '
        f'margin-top: 8px;">{_esc(flow.user_message) or "(no user message)"}</div>'
        "</div>"
    )


# ---------- flow diagram ----------


def _flow_diagram(flow: AgentFlowSnapshot) -> str:
    """Six nodes in a horizontal row with arrow connectors between.

    Connector is a flex-grow pseudo line with the arrowhead drawn
    via CSS borders inline. The whole row scrolls horizontally on
    narrow viewports so nothing overflows.
    """
    order = ("patient", "router", "specialist", "tools", "sentinel", "response")
    nodes = flow.nodes
    pieces: list[str] = []
    for idx, pos in enumerate(order):
        node = nodes.get(pos)
        if node is None:
            continue
        pieces.append(_node_with_detail(node))
        if idx < len(order) - 1:
            pieces.append(_connector(nodes.get(order[idx + 1])))
    inner = (
        '<div class="clarion-row" style="gap: 0; align-items: center; '
        'overflow-x: auto; padding-bottom: 4px;">'
        + "".join(pieces)
        + "</div>"
    )
    return c.panel(title="Path Through the Graph", body_html=inner)


def _node_with_detail(node: FlowNode) -> str:
    """One agent_node + a one-line subtitle. The subtitle is the
    flow-specific data (e.g. "chose Booking", "search_slots") that
    the agent_node primitive doesn't render."""
    inner_node = c.agent_node(
        name=node.name,
        state=node.state,
        ms=node.ms,
        cost_usd=node.cost_usd,
    )
    return (
        '<div class="clarion-stack" style="gap: 6px; min-width: 140px;">'
        + inner_node
        + f'<div style="font-size: var(--fs-xs); color: var(--c-text-muted); '
        f'text-align: center; padding: 0 4px;">{_esc(node.detail)}</div>'
        "</div>"
    )


def _connector(next_node: FlowNode | None) -> str:
    """A horizontal arrow between two nodes. Colour matches the
    downstream node's state so the lit path is visually traceable."""
    state = next_node.state if next_node is not None else "idle"
    colour_map = {
        "active":    "var(--c-accent)",
        "done":      "var(--c-healthy)",
        "escalated": "var(--c-critical)",
        "idle":      "var(--c-border-strong)",
    }
    colour = colour_map.get(state, "var(--c-border-strong)")
    return (
        '<div style="flex: 1 1 24px; min-width: 24px; max-width: 40px; '
        'display: flex; align-items: center;">'
        f'<div style="flex: 1; height: 2px; background: {colour};"></div>'
        f'<div style="width: 0; height: 0; border-top: 6px solid transparent; '
        f'border-bottom: 6px solid transparent; border-left: 8px solid {colour};"></div>'
        "</div>"
    )


# ---------- specialist panel ----------


def _specialist_panel(chosen: str, others: list[str]) -> str:
    """5-card grid of specialists, active one lit."""
    cards = [_specialist_card(chosen, is_active=True)]
    cards.extend(_specialist_card(name, is_active=False) for name in others)
    body = (
        '<div style="display: grid; grid-template-columns: repeat(auto-fit, '
        'minmax(140px, 1fr)); gap: 8px;">'
        + "".join(cards)
        + "</div>"
    )
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title="Specialists", body_html=body)
        + "</div>"
    )


def _specialist_card(name: str, *, is_active: bool) -> str:
    """Mini-card that mirrors agent_node visually but is keyed on
    is_active rather than the trace-derived state."""
    state = "active" if is_active else "idle"
    badge_state: c.Status = "info" if is_active else "unknown"
    badge_label = "ACTIVE" if is_active else "IDLE"
    return (
        f'<div class="clarion-agent-node" data-state="{state}">'
        f'<div class="clarion-agent-node-name">{_esc(name)}</div>'
        '<div class="clarion-row" style="gap: 6px; margin-top: 4px;">'
        + c.status_badge(badge_state, label=badge_label)
        + "</div>"
        "</div>"
    )


# ---------- tools panel ----------


def _tools_panel(tools: list[str]) -> str:
    if not tools:
        body = _empty_message("No tool calls fired on this turn.")
    else:
        chips = "".join(_tool_chip(name) for name in tools)
        body = (
            '<div class="clarion-row" style="gap: 8px; flex-wrap: wrap;">'
            + chips
            + "</div>"
        )
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title="Tool Calls", body_html=body)
        + "</div>"
    )


def _tool_chip(name: str) -> str:
    return (
        '<span style="display: inline-flex; align-items: center; '
        'padding: 4px 10px; background: var(--c-bg-subtle); '
        'border-radius: var(--r-pill); font-family: var(--font-mono); '
        f'font-size: var(--fs-xs); color: var(--c-mono-fg);">{_esc(name)}</span>'
    )


# ---------- trust posture ----------


def _trust_posture(flow: AgentFlowSnapshot) -> str:
    """Row of chips summarising what Sentinel saw + decided."""
    score = flow.escalation_score
    score_status: c.Status
    if score >= 0.80:
        score_status = "critical"
    elif score >= 0.50:
        score_status = "warning"
    else:
        score_status = "healthy"

    reason_chips = "".join(
        c.status_badge("warning", label=reason)
        for reason in flow.escalation_reasons[:5]
    )
    violation_chips = "".join(
        c.status_badge("critical", label=violation)
        for violation in flow.judge_violations[:3]
    )

    body = (
        '<div class="clarion-stack" style="gap: 12px;">'
        '<div class="clarion-row" style="gap: 12px; flex-wrap: wrap;">'
        + c.status_badge(score_status, label=f"sentinel score {score:.2f}")
        + c.status_badge(
            _outcome_severity(flow.final_outcome),
            label=f"outcome: {flow.final_outcome}",
        )
        + "</div>"
        '<div class="clarion-stack" style="gap: 8px;">'
        '<div class="clarion-kpi-label">ESCALATION REASONS</div>'
        '<div class="clarion-row" style="gap: 6px; flex-wrap: wrap;">'
        + (reason_chips or _muted("none fired"))
        + "</div>"
        "</div>"
        '<div class="clarion-stack" style="gap: 8px;">'
        '<div class="clarion-kpi-label">JUDGE VIOLATIONS</div>'
        '<div class="clarion-row" style="gap: 6px; flex-wrap: wrap;">'
        + (violation_chips or _muted("none flagged"))
        + "</div>"
        "</div>"
        "</div>"
    )
    return c.panel(title="Trust Posture", body_html=body)


def _muted(text: str) -> str:
    return (
        f'<span style="font-size: var(--fs-sm); color: var(--c-text-muted);">'
        f"{_esc(text)}</span>"
    )


# ---------- helpers ----------


def _outcome_severity(outcome: str) -> c.Status:
    if outcome.startswith("escalated"):
        return "critical"
    if outcome in {"unresolved", "refused_clinical"}:
        return "warning"
    if outcome in {"booked", "rescheduled", "cancelled", "info_provided", "task_created"}:
        return "healthy"
    return "info"


def _section_title(*, title: str, subtitle: str) -> str:
    return (
        '<div class="clarion-stack" style="gap: 4px;">'
        f'<div style="font-size: var(--fs-2xl); font-weight: var(--fw-bold); '
        f'color: var(--c-text-strong); line-height: 1.1;">{title}</div>'
        f'<div style="font-size: var(--fs-sm); color: var(--c-text-muted);">{subtitle}</div>'
        "</div>"
    )


def _empty_message(text: str) -> str:
    return (
        f'<div style="padding: 16px; color: var(--c-text-muted); '
        f'font-size: var(--fs-sm);">{_esc(text)}</div>'
    )


def _esc(text: str) -> str:
    import html as _html

    return _html.escape(text, quote=True)


__all__ = ["build_html", "empty_html"]
