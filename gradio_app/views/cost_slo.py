"""Cost & SLO — the executive bottom strip.

Three zones:

1. **Top headline KPIs** — total cost, naive monthly projection,
   global p50/p95 latency.
2. **Per-tenant cost breakdown** — one card per tenant with
   total cost, avg cost/request, tokens in/out.
3. **Per-tenant latency breakdown** — one card per tenant with
   avg, p50, p95, sample count.

The "projection" is intentionally naive: today's total x 30. The
audit doc names the precise place a billing forecaster would
slot in later. The point of this view today is to make the cost
+ latency story *visible*, not to model spend.
"""

from __future__ import annotations

from gradio_app import components as c
from gradio_app.data_sources import (
    CostBreakdown,
    GlobalCostSLO,
    LatencyBreakdown,
)

# ---------- public ----------


def build_html(rollup: GlobalCostSLO) -> str:
    return (
        '<div class="clarion-stack" style="gap: 24px;">'
        + c.page_intro(
            title="Cost & SLO",
            what="Per-tenant spend, latency, and SLA evidence.",
            quote="Speed, savings, and an audit trail.",
        )
        + _kpi_strip(rollup)
        + '<div class="clarion-row" style="align-items: stretch; gap: 16px;">'
        + _cost_share_panel(rollup)
        + _cost_panel(rollup.per_tenant_cost)
        + "</div>"
        + _latency_panel(rollup.per_tenant_latency)
        + "</div>"
    )


def empty_html() -> str:
    body = (
        '<div class="clarion-stack" style="align-items: center; gap: 12px;">'
        '<div style="font-size: var(--fs-lg); color: var(--c-text);">'
        "No cost or latency data on disk."
        "</div>"
        '<div style="font-size: var(--fs-sm); color: var(--c-text-muted);">'
        "Cost &amp; latency rollups are derived from per-tenant trace reports."
        "</div>"
        "</div>"
    )
    return (
        '<div class="clarion-stack" style="gap: 24px;">'
        + c.page_intro(
            title="Cost & SLO",
            what="Per-tenant spend, latency, and SLA evidence.",
            quote="Speed, savings, and an audit trail.",
        )
        + c.panel(title="Awaiting Traces", body_html=body)
        + "</div>"
    )


# ---------- internals ----------


def _kpi_strip(rollup: GlobalCostSLO) -> str:
    """4 headline tiles. Lower is better for cost + latency; we
    flip the band test for those. The p95 budget target reflects
    the loadtest SLA (1500 ms scripted, much faster locally)."""
    p50_budget_ms = 1000
    p95_budget_ms = 1500
    tiles = [
        c.kpi_tile(
            label="TOTAL COST (DEMO)",
            value=f"${rollup.total_cost_usd:.4f}",
            status="info",
        ),
        c.kpi_tile(
            label="MONTHLY PROJECTION",
            value=f"${rollup.monthly_projection_usd:.2f}",
            status="info",
        ),
        c.kpi_tile(
            label="LATENCY P50",
            value=f"{int(rollup.global_p50_ms)} ms",
            status=_band_inverse(
                rollup.global_p50_ms, healthy=p50_budget_ms * 0.75, warning=p50_budget_ms
            ),
        ),
        c.kpi_tile(
            label="LATENCY P95",
            value=f"{int(rollup.global_p95_ms)} ms",
            status=_band_inverse(
                rollup.global_p95_ms, healthy=p95_budget_ms * 0.75, warning=p95_budget_ms
            ),
        ),
    ]
    return c.kpi_strip(tiles)


_DONUT_PALETTE = (
    "#06B6D4",  # cyan-500
    "#A78BFA",  # violet-400
    "#F59E0B",  # amber-500
    "#10B981",  # emerald-500
    "#F472B6",  # pink-400
    "#22D3EE",  # cyan-400
    "#FB7185",  # rose-400
    "#34D399",  # emerald-400
)


def _cost_share_panel(rollup: GlobalCostSLO) -> str:
    """Donut: per-tenant share of total cost.

    Sits to the left of the per-tenant cost cards as a visual
    summary - readers should grasp who drives the bill in one
    glance before scanning the detail rows.
    """
    segments: list[tuple[str, float, str]] = [
        (row.tenant, row.total_cost_usd, _DONUT_PALETTE[i % len(_DONUT_PALETTE)])
        for i, row in enumerate(rollup.per_tenant_cost)
    ]
    donut = c.donut_chart(
        segments=segments,
        center_value=f"${rollup.total_cost_usd:.2f}",
        center_label="Total",
        size=200,
    )
    return c.panel(title="Cost Share by Tenant", body_html=donut)


def _cost_panel(rows: list[CostBreakdown]) -> str:
    if not rows:
        body = _empty_message("No cost data on disk yet.")
    else:
        body = (
            '<div class="clarion-stack" style="gap: 8px;">'
            + "".join(_cost_row(r) for r in rows)
            + "</div>"
        )
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title="Cost by Tenant", body_html=body)
        + "</div>"
    )


def _cost_row(row: CostBreakdown) -> str:
    return (
        '<div class="clarion-kpi-tile" data-status="info" '
        'style="min-width: unset; height: auto; padding: 16px 20px;">'
        '<div class="clarion-row" style="justify-content: space-between;">'
        f'<div style="font-size: var(--fs-md); font-weight: var(--fw-semibold); '
        f'color: var(--c-text-strong);">{_esc(row.tenant)}</div>'
        + c.cost_chip(usd=row.total_cost_usd, period=f"{row.scenarios} runs")
        + "</div>"
        '<div class="clarion-row" style="gap: 16px; flex-wrap: wrap;">'
        + _mini_metric("AVG / REQ", f"${row.avg_cost_per_request_usd:.4f}")
        + _mini_metric("IN TOKENS", f"{row.total_input_tokens:,}")
        + _mini_metric("OUT TOKENS", f"{row.total_output_tokens:,}")
        + "</div>"
        "</div>"
    )


def _latency_panel(rows: list[LatencyBreakdown]) -> str:
    if not rows:
        body = _empty_message("No latency samples on disk yet.")
    else:
        body = (
            '<div class="clarion-stack" style="gap: 8px;">'
            + "".join(_latency_row(r) for r in rows)
            + "</div>"
        )
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title="Latency by Tenant", body_html=body)
        + "</div>"
    )


def _latency_row(row: LatencyBreakdown) -> str:
    p50_band = _band_inverse(row.p50_ms, healthy=750, warning=1000)
    return (
        f'<div class="clarion-kpi-tile" data-status="{p50_band}" '
        'style="min-width: unset; height: auto; padding: 16px 20px;">'
        '<div class="clarion-row" style="justify-content: space-between;">'
        f'<div style="font-size: var(--fs-md); font-weight: var(--fw-semibold); '
        f'color: var(--c-text-strong);">{_esc(row.tenant)}</div>'
        f'<div style="font-size: var(--fs-xs); color: var(--c-text-muted); '
        f'font-variant-numeric: tabular-nums;">'
        f"{row.sample_count} samples"
        "</div>"
        "</div>"
        '<div class="clarion-row" style="gap: 16px; flex-wrap: wrap;">'
        + _mini_metric("AVG", f"{int(row.avg_ms)} ms")
        + _mini_metric("P50", f"{int(row.p50_ms)} ms")
        + _mini_metric("P95", f"{int(row.p95_ms)} ms")
        + "</div>"
        "</div>"
    )


def _mini_metric(label: str, value: str) -> str:
    return (
        '<div class="clarion-stack" style="gap: 2px;">'
        f'<div class="clarion-kpi-label">{_esc(label)}</div>'
        f'<div style="font-size: var(--fs-md); font-weight: var(--fw-semibold); '
        f'color: var(--c-text-strong); font-variant-numeric: tabular-nums;">{_esc(value)}</div>'
        "</div>"
    )


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


def _band_inverse(value: float, *, healthy: float, warning: float) -> c.Status:
    if value <= healthy:
        return "healthy"
    if value <= warning:
        return "warning"
    return "critical"


def _esc(text: str) -> str:
    import html as _html

    return _html.escape(text, quote=True)


__all__ = ["build_html", "empty_html"]
