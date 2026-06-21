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
    sparkline: list[float] | None = None,
) -> str:
    """Render one KPI tile for the Mission Control top strip.

    Args:
        label:      short uppercase label ("CONTAINMENT").
        value:      primary number ("74.0%"). Already formatted -
                    callers pick the precision + unit.
        delta:      optional change vs. previous period ("+2.1%").
                    None hides the delta row.
        trend:      "up" / "down" / "flat" - sets the delta color.
        status:     border-left color: healthy/warning/critical/info.
        sparkline:  optional list of recent values for an inline
                    SVG sparkline. Auto-scaled to the tile width;
                    values are unitless. Empty / single-element
                    lists are dropped silently.

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
    spark_html = ""
    if sparkline is not None and len(sparkline) >= 2:
        spark_html = _spark_svg(sparkline, trend=trend, status=status)
    return (
        f'<div class="clarion-kpi-tile" data-status="{status}">'
        f'<div class="clarion-kpi-label">{_esc(label)}</div>'
        f'<div class="clarion-kpi-value">{_esc(value)}</div>'
        f"{delta_html}"
        f"{spark_html}"
        "</div>"
    )


def _spark_svg(values: list[float], *, trend: Trend, status: Status) -> str:
    """Render a fixed-size sparkline polyline as inline SVG.

    Auto-scales the y-axis to the value range; a flat series
    renders as a horizontal middle-line. Stroke colour follows
    the tile's status / trend so the spark line ties visually
    to the value above it.
    """
    width = 120
    height = 28
    pad = 2
    lo = min(values)
    hi = max(values)
    span = hi - lo if hi > lo else 1.0
    n = len(values)
    pts: list[str] = []
    for i, v in enumerate(values):
        x = pad + (width - 2 * pad) * (i / (n - 1))
        # SVG y grows downward; invert.
        y = pad + (height - 2 * pad) * (1.0 - (v - lo) / span)
        pts.append(f"{x:.1f},{y:.1f}")
    points = " ".join(pts)
    color = (
        "var(--c-critical)"
        if status == "critical"
        else "var(--c-warning)"
        if status == "warning"
        else "var(--c-healthy)"
        if (status == "healthy" or trend == "up")
        else "var(--c-text-muted)"
        if trend == "down"
        else "var(--c-accent)"
    )
    return (
        f'<svg class="clarion-kpi-spark" '
        f'width="100%" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'preserveAspectRatio="none" aria-hidden="true">'
        f'<polyline points="{points}" '
        f'fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'opacity="0.85"/>'
        "</svg>"
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

    # Five tick marks at 0%, 25%, 50%, 75%, 100% around the arc.
    # Each is a short radial line drawn slightly outside the track.
    ticks: list[str] = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        t_deg = 180.0 - 180.0 * t
        t_rad = math.radians(t_deg)
        inner = r + 9
        outer = r + 16
        x1 = cx + inner * math.cos(t_rad)
        y1 = cy - inner * math.sin(t_rad)
        x2 = cx + outer * math.cos(t_rad)
        y2 = cy - outer * math.sin(t_rad)
        ticks.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{PALETTE["text_muted"]}" stroke-width="1.5" '
            'stroke-linecap="round" opacity="0.5"/>'
        )
    tick_marks = "".join(ticks)

    # Use a unique gradient ID per render to avoid SVG ID collisions
    # if multiple gauges sit on the same page.
    grad_id = f"clarion-gauge-grad-{int(score * 1000):04d}"

    svg = (
        '<svg viewBox="0 0 180 120" width="180" height="120" '
        'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        "<defs>"
        f'<linearGradient id="{grad_id}" x1="0%" y1="0%" x2="100%" y2="0%">'
        f'<stop offset="0%" stop-color="{fill}" stop-opacity="0.65"/>'
        f'<stop offset="100%" stop-color="{fill}" stop-opacity="1.0"/>'
        "</linearGradient>"
        "</defs>"
        # Tick marks (drawn first so they sit under the arc visually).
        + tick_marks
        # Track (full semicircle).
        + f'<path d="M {track_start_x} {track_start_y} '
        f"A {r} {r} 0 0 1 {track_end_x} {track_end_y}\" "
        f'fill="none" stroke="{PALETTE["bg_subtle"]}" stroke-width="14" '
        'stroke-linecap="round"/>'
        # Fill (partial arc, score-driven, gradient-stroked).
        f'<path d="M {track_start_x} {track_start_y} '
        f"A {r} {r} 0 0 1 {end_x:.2f} {end_y:.2f}\" "
        f'fill="none" stroke="url(#{grad_id})" stroke-width="14" '
        'stroke-linecap="round"/>'
        # Threshold hairline.
        f'<line x1="{th_x:.2f}" y1="{cy - r - 4}" x2="{th_x:.2f}" '
        f'y2="{cy - r + 18}" stroke="{PALETTE["text_muted"]}" '
        'stroke-width="2" stroke-dasharray="3 3"/>'
        # Centered score - large numeric inside the arc.
        f'<text x="{cx}" y="{cy - 6}" text-anchor="middle" '
        f'fill="{PALETTE["text_strong"]}" font-size="26" '
        f'font-weight="700" font-family="ui-sans-serif, system-ui, sans-serif">'
        f"{score:.2f}"
        "</text>"
        # Subtitle below the score.
        f'<text x="{cx}" y="{cy + 10}" text-anchor="middle" '
        f'fill="{PALETTE["text_muted"]}" font-size="9" '
        'font-family="ui-monospace, monospace" letter-spacing="0.12em">'
        f"THRESHOLD {int(threshold * 100)}"
        "</text>"
        "</svg>"
    )
    return (
        '<div class="clarion-gauge">'
        f'<div class="clarion-gauge-label">{_esc(label)}</div>'
        f"{svg}"
        "</div>"
    )


# ---------- Donut chart (SVG) ----------


def donut_chart(
    *,
    segments: list[tuple[str, float, str]],
    center_value: str = "",
    center_label: str = "",
    size: int = 180,
) -> str:
    """Inline-SVG donut chart for share/composition visuals.

    Args:
        segments: list of ``(label, value, color)``. Values are
                  summed and each segment's arc spans the
                  proportional sweep. Negative values are clamped
                  to zero. A list summing to zero falls back to
                  the empty placeholder.
        center_value: optional big string in the donut hole
                      (typically the total).
        center_label: optional small caption under the center value.
        size: SVG side in pixels; the donut is sized to fit.

    The legend is rendered as a sibling list below the SVG -
    one row per segment with the colour dot, label, and value.
    """
    total = sum(max(0.0, v) for _, v, _ in segments)
    cx = cy = size / 2
    radius = size * 0.42
    stroke = size * 0.12  # donut ring width

    if total <= 0:
        return (
            f'<div class="clarion-donut" style="width: {size}px;">'
            f'<svg width="{size}" height="{size}" '
            f'viewBox="0 0 {size} {size}" '
            'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
            f'<circle cx="{cx}" cy="{cy}" r="{radius:.2f}" '
            'fill="none" stroke="var(--c-bg-subtle)" '
            f'stroke-width="{stroke:.2f}"/>'
            "</svg>"
            '<div class="clarion-donut-empty">No data</div>'
            "</div>"
        )

    # Build each arc segment. Each is a circle stroke-dasharray
    # trick: we draw a full circle but use stroke-dashoffset to
    # rotate it into position and stroke-dasharray to size it.
    circ = 2 * math.pi * radius
    offset = 0.0
    arcs: list[str] = []
    for _label, value, color in segments:
        v = max(0.0, value)
        if v == 0:
            continue
        arc_len = (v / total) * circ
        gap = circ - arc_len
        # The dasharray "arc_len gap" plus dashoffset rotates the
        # segment into place. Negative dashoffset moves clockwise.
        arcs.append(
            f'<circle cx="{cx}" cy="{cy}" r="{radius:.2f}" '
            f'fill="none" stroke="{color}" '
            f'stroke-width="{stroke:.2f}" '
            f'stroke-dasharray="{arc_len:.2f} {gap:.2f}" '
            f'stroke-dashoffset="{-offset:.2f}" '
            'transform="rotate(-90 ' + f"{cx} {cy})\"/>"
        )
        offset += arc_len

    center_html = ""
    if center_value or center_label:
        center_html = (
            # SVG <text> nodes inside the donut hole.
            f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" '
            'fill="var(--c-text-strong)" font-size="20" '
            'font-weight="700" '
            'font-family="ui-sans-serif, system-ui, sans-serif">'
            f"{_esc(center_value)}"
            "</text>"
            f'<text x="{cx}" y="{cy + 14}" text-anchor="middle" '
            'fill="var(--c-text-muted)" font-size="9" '
            'font-family="ui-monospace, monospace" '
            'letter-spacing="0.1em">'
            f"{_esc(center_label.upper())}"
            "</text>"
        )

    legend_rows = "".join(
        '<div class="clarion-donut-legend-row">'
        f'<span class="clarion-donut-swatch" style="background:{color};"></span>'
        f'<span class="clarion-donut-legend-label">{_esc(label)}</span>'
        f'<span class="clarion-donut-legend-value">'
        f"{_fmt_share(value, total)}</span>"
        "</div>"
        for label, value, color in segments
        if value > 0
    )

    return (
        f'<div class="clarion-donut" style="width: {size}px;">'
        f'<svg width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" '
        'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        # Track ring (full grey circle behind the segments).
        f'<circle cx="{cx}" cy="{cy}" r="{radius:.2f}" '
        'fill="none" stroke="var(--c-bg-subtle)" '
        f'stroke-width="{stroke:.2f}"/>'
        + "".join(arcs)
        + center_html
        + "</svg>"
        + '<div class="clarion-donut-legend">'
        + legend_rows
        + "</div>"
        "</div>"
    )


def _fmt_share(value: float, total: float) -> str:
    if total <= 0:
        return "—"
    pct = (value / total) * 100.0
    return f"{pct:.1f}%"


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
    """One node in the Agent Flow SVG (Phase D, polished in H7).

    The node is a card with a state-colored indicator dot in the
    top-left, the name beside it, and a meta footer below
    showing latency + cost. CSS owns the gradient + border.
    Connector lines + animation are owned by the view, which
    composes multiple nodes into the full diagram.
    """
    meta_parts: list[str] = []
    if ms is not None:
        meta_parts.append(
            f'<span class="clarion-agent-node-meta-item">'
            f'<span class="clarion-agent-node-meta-key">latency</span>'
            f"<span>{ms} ms</span></span>"
        )
    if cost_usd is not None:
        meta_parts.append(
            f'<span class="clarion-agent-node-meta-item">'
            f'<span class="clarion-agent-node-meta-key">cost</span>'
            f"<span>${cost_usd:.4f}</span></span>"
        )
    meta_html = (
        f'<div class="clarion-agent-node-meta">{"".join(meta_parts)}</div>'
        if meta_parts
        else ""
    )
    return (
        f'<div class="clarion-agent-node" data-state="{state}">'
        f'<div class="clarion-agent-node-header">'
        f'<span class="clarion-agent-node-dot" aria-hidden="true"></span>'
        f'<span class="clarion-agent-node-name">{_esc(name)}</span>'
        f"</div>"
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


# ---------- Phase G: top brand strip ----------


def brand_strip(
    *,
    version: str,
    env: str = "live",
    env_status: Status = "healthy",
    tagline: str = "Configurable Multi-Agent Healthcare Operations Platform",
) -> str:
    """Top app strip — brand mark + name + version + environment badge.

    Sits above the KPI strip in the Phase G shell. The brand strip
    is the single piece of chrome that NEVER swaps (it's not
    customer-bound), so it's safe to render statically once and
    never rebuild.
    """
    # SVG mark — a faceted hex/diamond made of cyan gradient
    # planes. Reads as a stylised mission-control sigil; crisp at
    # any size and free of Unicode font dependencies.
    logo_svg = (
        '<svg class="clarion-brand-mark" viewBox="0 0 36 36" '
        'width="38" height="38" xmlns="http://www.w3.org/2000/svg" '
        'aria-label="Clarion Vision">'
        "<defs>"
        '<linearGradient id="clarion-logo-grad" x1="0%" y1="0%" '
        'x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#22D3EE"/>'
        '<stop offset="100%" stop-color="#0E7490"/>'
        "</linearGradient>"
        '<linearGradient id="clarion-logo-face" x1="0%" y1="0%" '
        'x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#67E8F9" stop-opacity="0.95"/>'
        '<stop offset="100%" stop-color="#0891B2" stop-opacity="0.9"/>'
        "</linearGradient>"
        "</defs>"
        # Outer hexagon — the badge frame
        '<polygon points="18,3 31,10.5 31,25.5 18,33 5,25.5 5,10.5" '
        'fill="rgba(6, 182, 212, 0.10)" '
        'stroke="url(#clarion-logo-grad)" stroke-width="1.5" '
        'stroke-linejoin="round"/>'
        # Inner diamond — facet 1 (front)
        '<polygon points="18,9 26,18 18,27 10,18" '
        'fill="url(#clarion-logo-face)" '
        'stroke="#67E8F9" stroke-width="1" '
        'stroke-linejoin="round"/>'
        # Inner highlight — facet 2 (top-right reflection)
        '<polygon points="18,9 26,18 18,18" '
        'fill="rgba(255, 255, 255, 0.18)"/>'
        "</svg>"
    )
    # Theme toggle: sun/moon SVG that flips body's `theme-light`
    # class on click. Inline onclick is the simplest cross-route
    # to JS that survives Gradio's renderer.
    theme_toggle = (
        '<button class="clarion-theme-toggle" type="button" '
        'title="Toggle light / dark theme" aria-label="Toggle theme" '
        "onclick=\"document.body.classList.toggle('theme-light')\">"
        '<svg class="clarion-theme-icon-dark" width="16" height="16" '
        'viewBox="0 0 16 16" fill="none" stroke="currentColor" '
        'stroke-width="1.5" stroke-linecap="round" '
        'stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M13 9.5A5 5 0 0 1 6.5 3a5.5 5.5 0 1 0 6.5 6.5z"/>'
        "</svg>"
        '<svg class="clarion-theme-icon-light" width="16" height="16" '
        'viewBox="0 0 16 16" fill="none" stroke="currentColor" '
        'stroke-width="1.5" stroke-linecap="round" '
        'stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="8" cy="8" r="3"/>'
        '<path d="M8 1v2M8 13v2M1 8h2M13 8h2'
        "M3.05 3.05l1.4 1.4M11.55 11.55l1.4 1.4"
        'M3.05 12.95l1.4-1.4M11.55 4.45l1.4-1.4"/>'
        "</svg>"
        "</button>"
    )
    return (
        '<div class="clarion-brand-strip">'
        '<div class="clarion-brand-left">'
        + logo_svg
        + '<div class="clarion-brand-text">'
        '<div class="clarion-brand-name">'
        'Clarion <span class="clarion-brand-suffix">Vision</span>'
        "</div>"
        f'<div class="clarion-brand-tagline">{_esc(tagline)}</div>'
        "</div>"
        "</div>"
        '<div class="clarion-brand-right">'
        f'<span class="clarion-brand-version">v{_esc(version)}</span>'
        + status_badge(env_status, label=env.upper())
        + theme_toggle
        + "</div>"
        "</div>"
    )


# ---------- Empty-state illustration ----------


def empty_state(
    *,
    glyph: str = "data",
    title: str,
    detail: str = "",
    hint: str = "",
) -> str:
    """Polished empty-state block with a small SVG illustration.

    Replaces the plain "No data" copy that used to live inline in
    each view. ``glyph`` picks an SVG vignette:

      - "data"     : stacked rectangles (a "no records" file metaphor)
      - "patient"  : circle + arc avatar silhouette
      - "trace"    : zig-zag waveform on a baseline
      - "donut"    : ring with a gap (matches the cost donut)
    """
    glyphs = {
        "data": (
            '<svg width="72" height="72" viewBox="0 0 72 72" '
            'fill="none" xmlns="http://www.w3.org/2000/svg" '
            'aria-hidden="true">'
            '<rect x="18" y="20" width="36" height="6" rx="2" '
            'fill="var(--c-bg-subtle)" stroke="var(--c-border)"/>'
            '<rect x="18" y="32" width="36" height="6" rx="2" '
            'fill="var(--c-bg-subtle)" stroke="var(--c-border)"/>'
            '<rect x="18" y="44" width="22" height="6" rx="2" '
            'fill="var(--c-bg-subtle)" stroke="var(--c-border)" '
            'opacity="0.6"/>'
            '<circle cx="56" cy="48" r="9" fill="none" '
            'stroke="var(--c-accent)" stroke-width="2"/>'
            '<path d="M62 54L67 59" stroke="var(--c-accent)" '
            'stroke-width="2" stroke-linecap="round"/>'
            "</svg>"
        ),
        "patient": (
            '<svg width="72" height="72" viewBox="0 0 72 72" '
            'fill="none" xmlns="http://www.w3.org/2000/svg" '
            'aria-hidden="true">'
            '<circle cx="36" cy="28" r="10" stroke="var(--c-border)" '
            'fill="var(--c-bg-subtle)" stroke-width="2"/>'
            '<path d="M16 58c0-11 9-18 20-18s20 7 20 18" '
            'stroke="var(--c-border)" fill="var(--c-bg-subtle)" '
            'stroke-width="2" stroke-linecap="round"/>'
            '<circle cx="52" cy="20" r="5" fill="var(--c-accent)" '
            'opacity="0.8"/>'
            "</svg>"
        ),
        "trace": (
            '<svg width="120" height="48" viewBox="0 0 120 48" '
            'fill="none" xmlns="http://www.w3.org/2000/svg" '
            'aria-hidden="true">'
            '<line x1="4" y1="40" x2="116" y2="40" '
            'stroke="var(--c-border)" stroke-width="1" '
            'stroke-dasharray="2 3"/>'
            '<polyline points="6,28 18,22 30,30 42,16 54,32 '
            "66,18 78,26 90,14 102,28 114,22\" "
            'stroke="var(--c-accent)" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round" '
            'fill="none" opacity="0.6"/>'
            '<circle cx="42" cy="16" r="3" fill="var(--c-accent)" '
            'opacity="0.7"/>'
            '<circle cx="90" cy="14" r="3" fill="var(--c-accent)" '
            'opacity="0.7"/>'
            "</svg>"
        ),
        "donut": (
            '<svg width="72" height="72" viewBox="0 0 72 72" '
            'fill="none" xmlns="http://www.w3.org/2000/svg" '
            'aria-hidden="true">'
            '<circle cx="36" cy="36" r="22" stroke="var(--c-border)" '
            'stroke-width="8" fill="none" stroke-dasharray="100 38" '
            'transform="rotate(-90 36 36)"/>'
            '<circle cx="36" cy="36" r="6" fill="var(--c-bg-subtle)"/>'
            "</svg>"
        ),
    }
    svg = glyphs.get(glyph, glyphs["data"])
    detail_html = (
        f'<div style="font-size: var(--fs-sm); color: var(--c-text-muted); '
        f'max-width: 380px; text-align: center;">{_esc(detail)}</div>'
        if detail
        else ""
    )
    hint_html = (
        f'<div style="font-family: var(--font-mono); font-size: var(--fs-xs); '
        f"color: var(--c-text-muted); background: var(--c-bg-subtle); "
        f"border: 1px solid var(--c-border); padding: 6px 10px; "
        f'border-radius: var(--r-sm);">{_esc(hint)}</div>'
        if hint
        else ""
    )
    return (
        '<div class="clarion-empty-state" '
        'style="display: flex; flex-direction: column; align-items: center; '
        'gap: 14px; padding: 36px 16px;">'
        + svg
        + f'<div style="font-size: var(--fs-lg); color: var(--c-text-strong); '
        f'font-weight: var(--fw-semibold);">{_esc(title)}</div>'
        + detail_html
        + hint_html
        + "</div>"
    )


__all__ = [
    "AgentState",
    "SignalWeight",
    "Status",
    "Trend",
    "agent_node",
    "brand_strip",
    "cost_chip",
    "donut_chart",
    "empty_state",
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
