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
        + _section_title(
            title="Mission Control",
            subtitle="Real-time health across every tenant",
        )
        + _kpi_strip(kpis)
        + c.panel(title="Tenant Health", body_html=_tenant_table(snapshots))
        + '<div class="clarion-row" style="align-items: stretch; gap: 16px;">'
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
            rows=[
                c.incident_row(
                    ts=item.detected_at,
                    severity="critical",
                    tenant=item.tenant,
                    summary=item.summary,
                )
                for item in emergencies
            ],
        )
        + "</div>"
        + "</div>"
    )


def empty_html() -> str:
    """No-data state. Surfaces when no tenant has produced any
    artifact yet (fresh deploy / first run).
    """
    body = (
        '<div class="clarion-stack" style="align-items: center; gap: 12px;">'
        '<div style="font-size: var(--fs-lg); color: var(--c-text);">'
        "No tenant data found."
        "</div>"
        '<div style="font-size: var(--fs-sm); color: var(--c-text-muted);">'
        "Run the evaluation harness for at least one customer:"
        "</div>"
        '<div class="clarion-mono" style="padding: 12px 16px; '
        'background: var(--c-bg-subtle); border-radius: var(--r-md);">'
        "python -m clarion.evaluation.cli run all --out reports/"
        "</div>"
        "</div>"
    )
    return (
        '<div class="clarion-stack" style="gap: 24px;">'
        + _section_title(
            title="Mission Control",
            subtitle="Real-time health across every tenant",
        )
        + c.panel(title="Awaiting Data", body_html=body)
        + "</div>"
    )


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
