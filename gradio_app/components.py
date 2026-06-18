"""Pure render helpers for the Clarion visual system.

Every function here takes typed inputs, returns an HTML string, and
holds no state. Views use these as building blocks — wrap the
returned string in a ``gr.HTML`` and the styled component appears.

Why string-returning and not ``gr.HTML``-returning:

- Snapshot tests can assert exact strings (deterministic, easy
  diffs).
- Composition stays cheap — a panel that lists 10 incident rows
  builds one string instead of constructing 10 Gradio components.
- Reuse outside Gradio — these strings also work in static report
  exports (Phase F+ will render some of them server-side).

The class names referenced here are defined in
``gradio_app/style.css``. Test coverage for the contract between
the two lives in ``tests/gradio_app/test_components.py``.
"""

from __future__ import annotations

import html as _html
import math
from typing import Literal

from gradio_app.theme import PALETTE

# Public type aliases — every view imports these for callsite typing.
Status = Literal["healthy", "warning", "critical", "info", "unknown"]
Trend = Literal["up", "down", "flat"]
SignalWeight = Literal["heavy", "medium", "light"]
AgentState = Literal["idle", "active", "done", "escalated"]


def _esc(text: str) -> str:
    """HTML-escape — used everywhere user-supplied text crosses a tag."""
    return _html.escape(text, quote=True)


# ---------- KPI tile ----------


def kpi_tile(
    *,
    label: str,
    value: str,
    delta: str | None = None,
    trend: Trend = "flat",
    status: Status = "info",
) -> str:
    """Render one KPI tile for the Mission Control top strip.

    Args:
        label:   short uppercase label ("CONTAINMENT").
        value:   primary number ("74.0%"). Already formatted —
                 callers pick the precision + unit.
        delta:   optional change vs. previous period ("+2.1%").
                 None hides the delta row.
        trend:   "up" / "down" / "flat" — sets the delta color.
        status:  border-left color: healthy/warning/critical/info/unknown.

    Returns a self-contained ``<div class="clarion-kpi-tile">`` HTML
    string. No script, no inline style overrides.
    """
    delta_html = ""
    if delta is not None:
        delta_html = (
            f'<div class="clarion-kpi-delta" data-trend="{trend}">'
            f"{_esc(delta)}"
            "</div>"
        )
    return (
        f'<div class="clarion-kpi-tile" data-status="{status}">'
        f'<div class="clarion-kpi-label">{_esc(label)}</div>'
        f'<div class="clarion-kpi-value">{_esc(value)}</div>'
        f"{delta_html}"
        "</div>"
    )


# ---------- Status badge ----------


def status_badge(state: Status, *, label: str | None = None) -> str:
    """Small pill: dot + state name. Used everywhere a tenant or
    span needs a one-glance condition indicator."""
    text = label if label is not None else state.upper()
    return (
        f'<span class="clarion-badge" data-state="{state}">'
        '<span class="dot"></span>'
        f"{_esc(text)}"
        "</span>"
    )


# ---------- Trust gauge (SVG) ----------


def trust_gauge(
    *,
    score: float,
    threshold: float = 0.5,
    label: str = "TRUST",
) -> str:
    """Inline-SVG semicircle gauge — the Sentinel hero element.

    Score is clamped to [0, 1]. A vertical hairline marks the
    decision threshold so a viewer instantly sees "above" vs.
    "below". Fill color reflects the band:

      score < threshold/2          -> healthy (we're far below
                                      the escalation trigger)
      score < threshold            -> info    (approaching)
      score < threshold + 0.2      -> warning
      else                         -> critical

    The SVG is sized 180x110 (viewBox); the wrapper styles padding,
    label, and the displayed value below.
    """
    score = max(0.0, min(1.0, score))

    # Map score → arc end angle. We sweep from 180° (left) to 0° (right)
    # across the visible semicircle.
    deg = 180.0 - 180.0 * score
    rad = math.radians(deg)
    cx, cy, r = 90.0, 95.0, 70.0
    end_x = cx + r * math.cos(rad)
    end_y = cy - r * math.sin(rad)

    # Track end is always at (cx + r, cy) — the right side of the semicircle.
    track_end_x = cx + r
    track_end_y = cy
    track_start_x = cx - r
    track_start_y = cy

    # Fill color by band.
    if score < threshold / 2:
        fill = PALETTE["healthy"]
    elif score < threshold:
        fill = PALETTE["accent"]
    elif score < threshold + 0.2:
        fill = PALETTE["warning"]
    else:
        fill = PALETTE["critical"]

    # Threshold marker — vertical hairline at the threshold's x position.
    th_deg = 180.0 - 180.0 * threshold
    th_rad = math.radians(th_deg)
    th_x = cx + r * math.cos(th_rad)

    svg = (
        '<svg viewBox="0 0 180 110" width="180" height="110" '
        'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        # Track (full semicircle).
        f'<path d="M {track_start_x} {track_start_y} '
        f"A {r} {r} 0 0 1 {track_end_x} {track_end_y}\" "
        f'fill="none" stroke="{PALETTE["bg_subtle"]}" stroke-width="14" '
        'stroke-linecap="round"/>'
        # Fill (partial arc, score-driven).
        f'<path d="M {track_start_x} {track_start_y} '
        f"A {r} {r} 0 0 1 {end_x:.2f} {end_y:.2f}\" "
        f'fill="none" stroke="{fill}" stroke-width="14" '
        'stroke-linecap="round"/>'
        # Threshold hairline.
        f'<line x1="{th_x:.2f}" y1="{cy - r - 4}" x2="{th_x:.2f}" '
        f'y2="{cy - r + 18}" stroke="{PALETTE["text_muted"]}" '
        'stroke-width="2" stroke-dasharray="3 3"/>'
        "</svg>"
    )
    return (
        '<div class="clarion-gauge">'
        f'<div class="clarion-gauge-label">{_esc(label)}</div>'
        f"{svg}"
        f'<div class="clarion-gauge-value">{score:.2f}</div>'
        "</div>"
    )


# ---------- Signal bar ----------


def signal_bar(
    *,
    name: str,
    value: float,
    weight: SignalWeight = "medium",
) -> str:
    """One weighted-signal row in the Sentinel Ops Center.

    ``value`` is the raw signal in [0, 1]; ``weight`` colors the
    fill so a viewer sees "heavy contributor" vs. "light" at a
    glance.
    """
    pct = max(0.0, min(100.0, value * 100.0))
    return (
        '<div class="clarion-signal">'
        f'<div class="clarion-signal-name">{_esc(name)}</div>'
        '<div class="clarion-signal-track">'
        f'<div class="clarion-signal-fill" data-weight="{weight}" '
        f'style="width: {pct:.1f}%"></div>'
        "</div>"
        f'<div class="clarion-signal-value">{value:.2f}</div>'
        "</div>"
    )


# ---------- Latency ring (SVG) ----------


def latency_ring(
    *,
    stage: str,
    ms: int,
    target_ms: int,
) -> str:
    """Small SVG ring chart — used for STT / Agent / TTS latency.

    Ring fill proportion = ms / target_ms, capped at 1.0. Color:
    green when comfortably under target, amber within 25%, red
    when over budget.
    """
    ratio = ms / max(1, target_ms)
    pct = min(1.0, ratio)
    if ratio < 0.75:
        color = PALETTE["healthy"]
    elif ratio < 1.0:
        color = PALETTE["warning"]
    else:
        color = PALETTE["critical"]

    cx, cy, r = 30.0, 30.0, 22.0
    circumference = 2 * math.pi * r
    fill_len = pct * circumference

    svg = (
        '<svg viewBox="0 0 60 60" width="60" height="60" '
        'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        # Background track.
        f'<circle cx="{cx}" cy="{cy}" r="{r}" '
        f'fill="none" stroke="{PALETTE["bg_subtle"]}" stroke-width="6"/>'
        # Fill arc — rotated -90° via transform so 0 starts at top.
        f'<circle cx="{cx}" cy="{cy}" r="{r}" '
        f'fill="none" stroke="{color}" stroke-width="6" '
        f'stroke-dasharray="{fill_len:.2f} {circumference:.2f}" '
        'stroke-linecap="round" '
        f'transform="rotate(-90 {cx} {cy})"/>'
        "</svg>"
    )
    return (
        '<div class="clarion-latency-ring">'
        f'<div class="clarion-latency-ring-label">{_esc(stage)}</div>'
        f"{svg}"
        f'<div class="clarion-latency-ring-value">{ms} ms</div>'
        "</div>"
    )


# ---------- Tenant card ----------


def tenant_card(
    *,
    customer_id: str,
    display_name: str,
    health: Status,
    last_run_at: str,
    is_active: bool = False,
) -> str:
    """One left-rail nav item (Phase G). The ``is_active`` flag
    swaps the border to the cyan accent so the current selection
    pops."""
    return (
        f'<div class="clarion-tenant-card" data-active="{str(is_active).lower()}">'
        f'<div class="clarion-tenant-card-name">{_esc(display_name)}</div>'
        '<div class="clarion-tenant-card-meta">'
        f"{status_badge(health)}"
        f'<span class="clarion-mono">{_esc(customer_id)}</span>'
        "</div>"
        '<div class="clarion-tenant-card-meta">'
        f'<span>{_esc(last_run_at)}</span>'
        "</div>"
        "</div>"
    )


# ---------- Incident row ----------


def incident_row(
    *,
    ts: str,
    severity: Status,
    tenant: str,
    summary: str,
) -> str:
    """One row in the recent-incidents stream."""
    return (
        '<div class="clarion-incident">'
        f'<div class="clarion-incident-time">{_esc(ts)}</div>'
        f"<div>{status_badge(severity)}</div>"
        f'<div class="clarion-incident-tenant">{_esc(tenant)}</div>'
        f'<div class="clarion-incident-summary">{_esc(summary)}</div>'
        "</div>"
    )


# ---------- Agent node ----------


def agent_node(
    *,
    name: str,
    state: AgentState,
    ms: int | None = None,
    cost_usd: float | None = None,
) -> str:
    """One node in the Agent Flow SVG (Phase D).

    The node itself is a styled card; the connector lines + animation
    are owned by the view, which composes multiple nodes into the
    full diagram.
    """
    meta_parts: list[str] = []
    if ms is not None:
        meta_parts.append(f"<span>{ms} ms</span>")
    if cost_usd is not None:
        meta_parts.append(f"<span>${cost_usd:.4f}</span>")
    meta_html = (
        f'<div class="clarion-agent-node-meta">{" ".join(meta_parts)}</div>'
        if meta_parts
        else ""
    )
    return (
        f'<div class="clarion-agent-node" data-state="{state}">'
        f'<div class="clarion-agent-node-name">{_esc(name)}</div>'
        f"{meta_html}"
        "</div>"
    )


# ---------- Cost chip ----------


def cost_chip(*, usd: float, period: str = "turn") -> str:
    """Monospace cost pill — used everywhere we surface a $ value."""
    return (
        '<span class="clarion-cost-chip">'
        f'<span class="clarion-cost-chip-value">${usd:.4f}</span>'
        f'<span class="clarion-cost-chip-period">/ {_esc(period)}</span>'
        "</span>"
    )


# ---------- Mono span ----------


def mono(text: str) -> str:
    """Wrap a string in the mono-fg class — trace IDs, conv IDs."""
    return f'<span class="clarion-mono">{_esc(text)}</span>'


# ---------- Panel + KPI strip (layout helpers) ----------


def panel(*, title: str, body_html: str) -> str:
    """Named-section card. Title goes above; body fills the
    interior. The body is expected to be pre-rendered HTML — usually
    a concat of primitives from this module."""
    return (
        '<div class="clarion-panel">'
        f'<div class="clarion-panel-title">{_esc(title)}</div>'
        f"{body_html}"
        "</div>"
    )


def kpi_strip(tiles_html: list[str]) -> str:
    """Wrap N pre-rendered KPI tile strings in the strip layout."""
    inner = "".join(tiles_html)
    return f'<div class="clarion-kpi-strip">{inner}</div>'


__all__ = [
    "AgentState",
    "SignalWeight",
    "Status",
    "Trend",
    "agent_node",
    "cost_chip",
    "incident_row",
    "kpi_strip",
    "kpi_tile",
    "latency_ring",
    "mono",
    "panel",
    "signal_bar",
    "status_badge",
    "tenant_card",
    "trust_gauge",
]
