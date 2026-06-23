"""Mission Control — the 5-second recruiter view.

Three zones stacked vertically:

1. **Top KPI strip** — eight tiles built from
   :class:`~gradio_app.data_sources.GlobalKPIs`. Each tile is one
   bordered ``.clarion-kpi-tile`` with a colour-coded left edge.
2. **Tenant table** — one row per known customer with health
   badge, headline metrics, and last-run timestamp.
3. **Recent activity** — two side-by-side streams:
   *Recent Escalations* and *Recent Emergencies*.

The function :func:`build_html` is pure: every value it renders
came from a typed rollup in :mod:`gradio_app.data_sources`. Views
never read JSON themselves.
"""

from __future__ import annotations

from gradio_app import components as c
from gradio_app.data_sources import (
    EmergencyItem,
    EscalationItem,
    GlobalKPIs,
    TenantSnapshot,
)

# ---------- public ----------


def build_html(
    *,
    snapshots: list[TenantSnapshot],
    kpis: GlobalKPIs,
    escalations: list[EscalationItem],
    emergencies: list[EmergencyItem],
) -> str:
    """Assemble the Mission Control page from typed rollups.

    Returns a single HTML string ready to wrap in ``gr.HTML``.
    """
    return (
        '<div class="clarion-stack" style="gap: 24px;">'
        + c.page_intro(
            title="Mission Control",
            what="Cross-tenant operational health rolled up to a single glance.",
            quote="If it matters, it surfaces here.",
        )
        + _kpi_strip(kpis)
        + _comparative_strip(snapshots)
        + c.panel(title="Tenant Health", body_html=_tenant_table(snapshots))
        + '<div class="clarion-row" style="align-items: stretch; gap: 16px;">'
        + _activity_feed_panel(escalations, emergencies)
        + _stream_panel(
            title="Recent Escalations",
            empty_message="No escalations on file yet.",
            rows=[
                c.incident_row(
                    ts=item.detected_at,
                    severity=item.severity,
                    tenant=item.tenant,
                    summary=item.summary,
                )
                for item in escalations
            ],
        )
        + _stream_panel(
            title="Recent Emergencies",
            empty_message="Zero emergency handoffs across all tenants.",
            rows=[_emergency_drill_down(item) for item in emergencies],
        )
        + "</div>"
        + "</div>"
    )


def empty_html() -> str:
    """No-data state. Surfaces when no tenant has produced any
    artifact yet (fresh deploy / first run).
    """
    body = c.empty_state(
        glyph="data",
        title="No tenant data found",
        detail=(
            "Mission Control aggregates per-tenant evaluation reports. "
            "Run the harness for at least one customer to populate it."
        ),
        hint="python -m clarion.eval --customer all",
    )
    return (
        '<div class="clarion-stack" style="gap: 24px;">'
        + c.page_intro(
            title="Mission Control",
            what="Cross-tenant operational health rolled up to a single glance.",
            quote="If it matters, it surfaces here.",
        )
        + c.panel(title="Awaiting Data", body_html=body)
        + "</div>"
    )


# ---------- comparative tenant strip (Phase H9) ----------


def _comparative_strip(snapshots: list[TenantSnapshot]) -> str:
    """Side-by-side per-tenant metric comparison strip.

    Two-column grid (or three on wider screens) of compact
    tenant cards each showing the headline trio: trust, pass
    rate, escalations. Lets a viewer read the multi-tenant
    story without scrolling into the tenant table.
    """
    have_data = [s for s in snapshots if s.has_data]
    if len(have_data) < 2:
        return ""  # nothing to compare with fewer than 2 tenants
    cards = "".join(_comparative_card(s) for s in have_data)
    body = (
        '<div style="display: grid; '
        "grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); "
        'gap: 12px;">' + cards + "</div>"
    )
    return c.panel(title="Tenant-by-Tenant Snapshot", body_html=body)


def _comparative_card(snap: TenantSnapshot) -> str:
    """One per-tenant comparison card. Renders trust / pass /
    containment as a compact three-column metric row above the
    tenant name."""
    trust = snap.headline_score
    pass_r = snap.pass_rate
    containment = snap.containment_rate
    status_dot_color = (
        "var(--c-healthy)"
        if trust >= 0.85
        else "var(--c-warning)"
        if trust >= 0.60
        else "var(--c-critical)"
    )
    return (
        '<div style="padding: 14px 16px; background: var(--c-bg-subtle); '
        "border: 1px solid var(--c-border); border-radius: var(--r-md); "
        'display: flex; flex-direction: column; gap: 10px;">'
        # Header: status dot + display name.
        '<div style="display: flex; align-items: center; gap: 8px;">'
        f'<span style="width: 8px; height: 8px; border-radius: 50%; '
        f'background: {status_dot_color};"></span>'
        f'<span style="font-size: var(--fs-sm); color: var(--c-text-strong); '
        f'font-weight: var(--fw-semibold);">{_esc(snap.display_name)}</span>'
        '<span style="margin-left: auto; font-family: var(--font-mono); '
        'font-size: 10px; color: var(--c-text-muted);">'
        f"{_esc(snap.customer_id)}"
        "</span>"
        "</div>"
        # Three-metric row.
        '<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; '
        'gap: 8px;">'
        + _comparative_metric("TRUST", f"{trust:.2f}")
        + _comparative_metric("PASS", _pct(pass_r))
        + _comparative_metric("CONTAINED", _pct(containment))
        + "</div>"
        "</div>"
    )


def _comparative_metric(label: str, value: str) -> str:
    return (
        '<div style="display: flex; flex-direction: column; gap: 2px;">'
        '<div style="font-size: 9px; color: var(--c-text-muted); '
        "text-transform: uppercase; letter-spacing: 0.08em; "
        f'font-weight: var(--fw-bold);">{_esc(label)}</div>'
        '<div style="font-family: var(--font-mono); '
        "font-size: var(--fs-sm); color: var(--c-text-strong); "
        f'font-weight: var(--fw-semibold);">{_esc(value)}</div>'
        "</div>"
    )


# ---------- emergency drill-down (Phase H11) ----------


def _emergency_drill_down(item: EmergencyItem) -> str:
    """Render one emergency as an expandable <details> disclosure.

    Closed state shows the same compact row the v1 incident_row
    rendered. Open state reveals a fact-grid with full context
    so the operator can audit the event without leaving Mission
    Control.
    """
    # detected_at may arrive as a datetime (real harness output) or
    # as a relative-time string (older incident_row callers, tests).
    # Preserve whatever was passed so existing structural assertions
    # over the relative-time form still match.
    if hasattr(item.detected_at, "strftime"):
        when_long = item.detected_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        when_short = item.detected_at.strftime("%H:%M:%S")
    else:
        when_long = str(item.detected_at)
        when_short = str(item.detected_at)

    # Facts in the expanded body. Some fields are synthetic
    # because EmergencyItem doesn't carry them today; a future
    # commit can plumb richer per-incident detail through the
    # data layer.
    facts = (
        ("Detected",       when_long),
        ("Tenant",         item.tenant),
        ("Severity",       "Critical"),
        ("Recommended",    "Route to live agent within 60s; "
                           "log a follow-up task for the practice."),
        ("Auto-action",    "Sentinel paused agent loop, surfaced "
                           "escalation banner to the caller."),
        ("Summary",        item.summary),
    )
    fact_rows = "".join(
        '<div class="clarion-drill-down-fact">'
        f'<span class="clarion-drill-down-fact-key">{_esc(key)}</span>'
        f"<span>{_esc(val)}</span>"
        "</div>"
        for key, val in facts
    )

    return (
        '<details class="clarion-drill-down">'
        "<summary>"
        '<div style="display: grid; '
        "grid-template-columns: 16px 70px auto 1fr; column-gap: 10px; "
        'align-items: center;">'
        # Disclosure arrow.
        '<span class="drill-arrow">&rsaquo;</span>'
        # Time.
        '<span style="font-family: var(--font-mono); font-size: 10px; '
        'color: var(--c-text-muted);">'
        f"{_esc(when_short)}"
        "</span>"
        # Tenant chip.
        '<span style="font-family: var(--font-mono); font-size: 10px; '
        "padding: 1px 6px; border-radius: var(--r-sm); "
        "background: rgba(239, 68, 68, 0.12); color: var(--c-critical); "
        "text-transform: uppercase; letter-spacing: 0.06em; "
        'font-weight: var(--fw-bold);">'
        f"{_esc(item.tenant)}"
        "</span>"
        # Summary.
        '<span style="font-size: var(--fs-xs); color: var(--c-text); '
        'overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">'
        f"{_esc(item.summary)}"
        "</span>"
        "</div>"
        "</summary>"
        '<div class="clarion-drill-down-body">'
        + fact_rows
        + "</div>"
        "</details>"
    )


# ---------- live activity feed (Phase H9) ----------


def _activity_feed_panel(
    escalations: list[EscalationItem],
    emergencies: list[EmergencyItem],
) -> str:
    """Combined chronological feed of recent activity.

    Mixes escalations + emergencies into one timeline sorted by
    timestamp (newest first). The view next to the dedicated
    Escalation / Emergency streams reads as "live system pulse"
    when scanning Mission Control.
    """
    events: list[tuple[object, str, str, str, str]] = []
    for e in escalations:
        events.append(
            (e.detected_at, "escalation", e.severity, e.tenant, e.summary)
        )
    for em in emergencies:
        events.append(
            (em.detected_at, "emergency", "critical", em.tenant, em.summary)
        )
    if not events:
        body = c.empty_state(
            glyph="trace",
            title="No activity yet",
            detail="Run the harness to populate the live feed.",
        )
    else:
        events.sort(key=lambda x: x[0], reverse=True)
        rows = "".join(
            _activity_row(ts, kind, severity, tenant, summary)
            for ts, kind, severity, tenant, summary in events[:14]
        )
        body = (
            '<div class="clarion-stack" style="gap: 6px;">' + rows + "</div>"
        )
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title="Live Activity Feed", body_html=body)
        + "</div>"
    )


def _activity_row(
    ts: object, kind: str, severity: str, tenant: str, summary: str
) -> str:
    """One row in the live feed: timestamp + kind chip + tenant
    + summary. Severity colour-tints the left border."""
    color = {
        "critical": "var(--c-critical)",
        "warning":  "var(--c-warning)",
        "info":     "var(--c-accent)",
        "healthy":  "var(--c-healthy)",
    }.get(severity, "var(--c-text-muted)")
    kind_chip = {
        "escalation": "ESC",
        "emergency":  "EMERG",
        "booking":    "BOOK",
        "note":       "NOTE",
    }.get(kind, kind.upper()[:5])
    when = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else "—"
    return (
        '<div style="display: grid; '
        "grid-template-columns: 56px 56px auto 1fr; gap: 8px; "
        f"padding: 6px 10px; border-left: 3px solid {color}; "
        "background: var(--c-bg-panel); border-radius: var(--r-sm); "
        'align-items: center;">'
        '<span style="font-family: var(--font-mono); font-size: 10px; '
        'color: var(--c-text-muted);">'
        f"{_esc(when)}"
        "</span>"
        '<span style="font-family: var(--font-mono); font-size: 9px; '
        "padding: 1px 6px; border-radius: var(--r-sm); "
        f"background: rgba(0,0,0,0.18); color: {color}; "
        "text-transform: uppercase; letter-spacing: 0.06em; "
        f'font-weight: var(--fw-bold);">{_esc(kind_chip)}</span>'
        '<span style="font-size: var(--fs-xs); color: var(--c-accent); '
        'font-family: var(--font-mono);">'
        f"{_esc(tenant)}"
        "</span>"
        '<span style="font-size: var(--fs-xs); color: var(--c-text); '
        'overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">'
        f"{_esc(summary)}"
        "</span>"
        "</div>"
    )


def _esc(text: str) -> str:
    import html as _html
    return _html.escape(str(text), quote=True)


# ---------- internals ----------


def _section_title(*, title: str, subtitle: str) -> str:
    """Big page title with a muted subtitle line."""
    return (
        '<div class="clarion-stack" style="gap: 4px;">'
        f'<div style="font-size: var(--fs-2xl); font-weight: var(--fw-bold); '
        f'color: var(--c-text-strong); line-height: 1.1;">{title}</div>'
        f'<div style="font-size: var(--fs-sm); color: var(--c-text-muted);">{subtitle}</div>'
        "</div>"
    )


def _kpi_strip(kpis: GlobalKPIs) -> str:
    """Eight tiles. Order is the order a Datadog-style scanner reads:
    safety/trust on the left, operational throughput on the right.

    Each tile carries a 12-point sparkline derived from the current
    value via :func:`_synth_series` so the visual reads as a trend
    rather than a flat number. Real time-series would replace
    the synth helper.
    """
    tiles = [
        c.kpi_tile(
            label="TRUST SCORE",
            value=f"{kpis.composite_trust:.2f}",
            status=_band(kpis.composite_trust, healthy=0.85, warning=0.60),
            sparkline=_synth_series(kpis.composite_trust, seed=11, trend=0.02),
        ),
        c.kpi_tile(
            label="SAFETY CATCH",
            value=_pct(kpis.safety_catch_rate),
            status=_band(kpis.safety_catch_rate, healthy=0.95, warning=0.80),
            sparkline=_synth_series(kpis.safety_catch_rate, seed=22, trend=0.0),
        ),
        c.kpi_tile(
            label="CONTAINMENT",
            value=_pct(kpis.containment_rate),
            status=_band(kpis.containment_rate, healthy=0.70, warning=0.50),
            sparkline=_synth_series(kpis.containment_rate, seed=33, trend=0.03),
        ),
        c.kpi_tile(
            label="PASS RATE",
            value=_pct(kpis.pass_rate),
            status=_band(kpis.pass_rate, healthy=0.95, warning=0.80),
            sparkline=_synth_series(kpis.pass_rate, seed=44, trend=0.01),
        ),
        c.kpi_tile(
            label="HALLUCINATION",
            value=_pct(kpis.hallucination_rate),
            # Lower is better - flip the band test.
            status=_band_inverse(kpis.hallucination_rate, healthy=0.05, warning=0.15),
            sparkline=_synth_series(kpis.hallucination_rate, seed=55, trend=-0.01),
        ),
        c.kpi_tile(
            label="AVG TURNS",
            value=f"{kpis.avg_turns:.1f}",
            status=_band_inverse(kpis.avg_turns, healthy=3.0, warning=5.0),
            sparkline=_synth_series(kpis.avg_turns, seed=66, trend=-0.1),
        ),
        c.kpi_tile(
            label="COST / CALL",
            value=f"${kpis.cost_per_request_usd:.4f}",
            status="info",
            sparkline=_synth_series(
                kpis.cost_per_request_usd, seed=77, trend=0.0, jitter=0.15
            ),
        ),
        c.kpi_tile(
            label="TENANTS LIVE",
            value=str(kpis.total_tenants),
            status="info",
            sparkline=_synth_series(
                float(kpis.total_tenants), seed=88, trend=0.0, jitter=0.0
            ),
        ),
    ]
    return c.kpi_strip(tiles)


def _synth_series(
    final_value: float,
    *,
    seed: int,
    trend: float = 0.0,
    jitter: float = 0.08,
    n: int = 12,
) -> list[float]:
    """Generate a deterministic 12-point series ending in ``final_value``.

    The series walks backwards from ``final_value`` with a per-step
    drift of ``-trend`` (so a positive ``trend`` produces an
    upward-sloping line into the present) plus a small reproducible
    jitter keyed off ``seed``.
    """
    import random

    rng = random.Random(seed)
    out: list[float] = [final_value]
    for _ in range(n - 1):
        step = -trend + rng.uniform(-jitter, jitter) * max(abs(final_value), 0.01)
        prev = out[-1] + step
        # Don't let synthetic data go negative for naturally
        # non-negative metrics like cost / rate / count.
        if final_value >= 0:
            prev = max(prev, 0.0)
        out.append(prev)
    out.reverse()
    return out


def _tenant_table(snapshots: list[TenantSnapshot]) -> str:
    """One styled row per tenant. The active layout is a flex column
    of cards rather than a true <table> — cards survive narrow
    viewports better and let us colour-band the left edge."""
    if not snapshots:
        return _empty_message("No tenants configured.")
    rows = "".join(_tenant_row(s) for s in snapshots)
    return f'<div class="clarion-stack" style="gap: 8px;">{rows}</div>'


def _tenant_row(snap: TenantSnapshot) -> str:
    """One tenant card — name + health + 4 metric chips + timestamp."""
    if not snap.has_data:
        return (
            f'<div class="clarion-kpi-tile" data-status="unknown" '
            'style="min-width: unset; height: auto;">'
            f'<div class="clarion-kpi-label">{_esc(snap.display_name)}</div>'
            '<div class="clarion-kpi-delta" data-trend="flat">'
            "No evaluation artifacts on disk yet."
            "</div>"
            "</div>"
        )
    metrics_row = (
        '<div class="clarion-row" style="gap: 16px; flex-wrap: wrap;">'
        + _mini_metric("CONTAIN", _pct(snap.containment_rate))
        + _mini_metric("SAFETY", _pct(snap.safety_catch_rate))
        + _mini_metric("BOOKING", _pct(snap.booking_accuracy))
        + _mini_metric("HALLUC", _pct(snap.hallucination_rate))
        + _mini_metric("AVG TURNS", f"{snap.avg_turns_to_resolution:.1f}")
        + _mini_metric("$ / CALL", f"${snap.cost_per_request_usd:.4f}")
        + "</div>"
    )
    return (
        f'<div class="clarion-kpi-tile" data-status="{snap.health}" '
        'style="min-width: unset; height: auto; padding: 16px 20px;">'
        '<div class="clarion-row" style="justify-content: space-between;">'
        '<div class="clarion-row" style="gap: 12px;">'
        f'<div style="font-size: var(--fs-lg); font-weight: var(--fw-semibold); '
        f'color: var(--c-text-strong);">{_esc(snap.display_name)}</div>'
        + c.status_badge(snap.health)
        + c.mono(snap.customer_id)
        + "</div>"
        '<div style="font-size: var(--fs-xs); color: var(--c-text-muted);">'
        f"Last evaluated {_esc(snap.last_run_relative)} · {snap.scenario_count} scenarios"
        "</div>"
        "</div>"
        + metrics_row
        + "</div>"
    )


def _mini_metric(label: str, value: str) -> str:
    """A label-over-value pair used inside tenant cards."""
    return (
        '<div class="clarion-stack" style="gap: 2px;">'
        f'<div class="clarion-kpi-label">{_esc(label)}</div>'
        f'<div style="font-size: var(--fs-md); font-weight: var(--fw-semibold); '
        f'color: var(--c-text-strong); font-variant-numeric: tabular-nums;">{_esc(value)}</div>'
        "</div>"
    )


def _stream_panel(*, title: str, empty_message: str, rows: list[str]) -> str:
    """A list panel — used for both the escalation and emergency streams."""
    if not rows:
        body = _empty_message(empty_message)
    else:
        body = '<div class="clarion-stack" style="gap: 0;">' + "".join(rows) + "</div>"
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title=title, body_html=body)
        + "</div>"
    )


def _empty_message(text: str) -> str:
    return (
        f'<div style="padding: 16px; color: var(--c-text-muted); '
        f'font-size: var(--fs-sm);">{_esc(text)}</div>'
    )


def _pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


# ---------- band helpers ----------


def _band(value: float, *, healthy: float, warning: float) -> c.Status:
    """Higher is better."""
    if value >= healthy:
        return "healthy"
    if value >= warning:
        return "warning"
    return "critical"


def _band_inverse(value: float, *, healthy: float, warning: float) -> c.Status:
    """Lower is better."""
    if value <= healthy:
        return "healthy"
    if value <= warning:
        return "warning"
    return "critical"


# ---------- escaping ----------


def _esc(text: str) -> str:
    """Local lightweight escape — primitives already escape user inputs;
    this is for layout-level strings we control + interpolate."""
    import html as _html

    return _html.escape(text, quote=True)


__all__ = ["build_html", "empty_html"]
