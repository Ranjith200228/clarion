"""Healthcare Operations — domain intelligence dashboard.

Closes the *doesn't feel healthcare-shaped* gap from the audit.
Four panels rendered top-to-bottom:

1. **Headline strip** — 4 tiles: TOTAL PROVIDERS, AVG UTILIZATION,
   NO-SHOW MEAN RISK, OPEN PMS TASKS.
2. **Provider availability heat map** — one row per provider with
   14 daily utilization cells colour-banded green/amber/red.
3. **No-show risk distribution** — three bars (low / medium /
   high) sized by fraction of upcoming appointments, with the
   mean risk shown as a chip.
4. **PMS task queue** — table of writeback-produced tasks
   sorted urgent-first, plus eligibility coverage donut on the
   side.

Per-tenant; binds to the customer dropdown. All data flows from
artifacts the engine already writes — SQLite store, M1 writeback
files, M3 predictions (or synthetic fallback) — no new endpoints.
"""

from __future__ import annotations

from gradio_app import components as c
from gradio_app.data_sources import (
    EligibilitySummary,
    HealthcareOpsSnapshot,
    NoShowRiskBucket,
    PmsTaskRow,
    ProviderUtilization,
)
from gradio_app.theme import PALETTE

# ---------- public ----------


def build_html(ops: HealthcareOpsSnapshot) -> str:
    """Render the Healthcare Operations view.

    Empty-state path still ships the headline strip and synthetic
    no-show distribution (the M3 generator runs even when no SQLite
    is on disk), so the page is never visually empty.
    """
    return (
        '<div class="clarion-stack" style="gap: 24px;">'
        + c.page_intro(
            title="Healthcare Operations",
            what=(
                "Bookings, no-shows, escalations, revenue recovered — "
                "the domain rollup."
            ),
            quote="Where front-line meets the front office.",
        )
        + _headline_strip(ops)
        + _provider_heatmap_panel(ops)
        + '<div class="clarion-row" style="align-items: stretch; gap: 16px;">'
        + _no_show_panel(ops.no_show_buckets, ops.no_show_total, ops.no_show_mean_risk)
        + _eligibility_panel(ops.eligibility, ops.eligibility_total)
        + "</div>"
        + _pms_panel(ops.pms_tasks)
        + "</div>"
    )


# ---------- headline strip ----------


def _headline_strip(ops: HealthcareOpsSnapshot) -> str:
    return c.kpi_strip(
        [
            c.kpi_tile(
                label="PROVIDERS",
                value=str(len(ops.providers)),
                status="info",
            ),
            c.kpi_tile(
                label="AVG UTILIZATION",
                value=f"{ops.avg_utilization * 100.0:.0f}%",
                status=_band_balanced(ops.avg_utilization),
            ),
            c.kpi_tile(
                label="MEAN NO-SHOW RISK",
                value=f"{ops.no_show_mean_risk * 100.0:.1f}%",
                status=_band_inverse(
                    ops.no_show_mean_risk, healthy=0.15, warning=0.30
                ),
            ),
            c.kpi_tile(
                label="OPEN PMS TASKS",
                value=str(ops.pms_open_count),
                status="info",
            ),
        ]
    )


# ---------- provider heat map ----------


def _provider_heatmap_panel(ops: HealthcareOpsSnapshot) -> str:
    if not ops.providers:
        body = _empty_message(
            "No structured.sqlite3 on disk yet. The provider availability "
            "panel reads from the per-tenant SQLite store seeded in Phase 3."
        )
        return c.panel(title="Provider Availability — Next 14 Days", body_html=body)

    rows = "".join(_provider_heatmap_row(p, days=ops.days_in_grid) for p in ops.providers)
    legend = (
        '<div class="clarion-row" style="gap: 8px; flex-wrap: wrap; '
        'margin-top: 12px;">'
        '<span style="font-size: var(--fs-xs); color: var(--c-text-muted);">'
        "Utilization:</span>"
        + _legend_swatch("under-utilised", PALETTE["bg_subtle"])
        + _legend_swatch("healthy 30-60%", PALETTE["healthy"])
        + _legend_swatch("warning 60-85%", PALETTE["warning"])
        + _legend_swatch("over-booked 85%+", PALETTE["critical"])
        + "</div>"
    )
    body = (
        '<div class="clarion-stack" style="gap: 6px;">' + rows + legend + "</div>"
    )
    return c.panel(title="Provider Availability — Next 14 Days", body_html=body)


def _provider_heatmap_row(prov: ProviderUtilization, *, days: int) -> str:
    """One row in the heat map: provider name + utilization chip +
    14 daily cells coloured by per-day utilization."""
    # Provider name lives in a fixed-width label column so cells
    # line up across rows.
    cells = "".join(
        _heatmap_cell(prov.daily[i] if i < len(prov.daily) else 0.0)
        for i in range(days)
    )
    util_pct = prov.utilization * 100.0
    return (
        '<div class="clarion-row" style="gap: 8px; align-items: center;">'
        f'<div style="flex: 0 0 180px; font-size: var(--fs-sm); '
        f'color: var(--c-text-strong); overflow: hidden; text-overflow: ellipsis; '
        f'white-space: nowrap;">{_esc(prov.provider_name)}</div>'
        f'<div class="clarion-row" style="gap: 2px; flex: 1 1 0; min-width: 0;">'
        + cells
        + "</div>"
        f'<div style="flex: 0 0 92px; text-align: right; '
        f'font-family: var(--font-mono); font-size: var(--fs-xs); '
        f'color: var(--c-text-muted);">{util_pct:.0f}% '
        f"({prov.slots_booked}/{prov.slots_total})"
        "</div>"
        "</div>"
    )


def _heatmap_cell(utilization: float) -> str:
    """One day's cell. Colour comes from the daily util band:
    > 0.85 critical, > 0.60 warning, > 0.30 healthy, else subtle."""
    if utilization >= 0.85:
        bg = PALETTE["critical"]
    elif utilization >= 0.60:
        bg = PALETTE["warning"]
    elif utilization >= 0.30:
        bg = PALETTE["healthy"]
    else:
        bg = PALETTE["bg_subtle"]
    pct = int(round(utilization * 100))
    return (
        f'<div title="{pct}% booked" '
        f'style="flex: 1 1 0; min-width: 12px; max-width: 36px; height: 20px; '
        f'background: {bg}; border-radius: 3px;"></div>'
    )


def _legend_swatch(label: str, color: str) -> str:
    return (
        '<span style="display: inline-flex; align-items: center; gap: 4px; '
        'font-size: var(--fs-xs); color: var(--c-text-muted);">'
        f'<span style="width: 12px; height: 12px; background: {color}; '
        f'border-radius: 3px;"></span>'
        f"{_esc(label)}"
        "</span>"
    )


# ---------- no-show panel ----------


def _no_show_panel(
    buckets: list[NoShowRiskBucket], total: int, mean_risk: float
) -> str:
    if total == 0:
        body = _empty_message("No no-show predictions available yet.")
    else:
        bars = "".join(_no_show_row(b) for b in buckets)
        body = (
            '<div class="clarion-stack" style="gap: 8px;">'
            + bars
            + '<div class="clarion-row" style="gap: 8px; flex-wrap: wrap; '
            'margin-top: 8px;">'
            + c.cost_chip(usd=mean_risk, period="mean risk")
            + '<span style="font-size: var(--fs-xs); color: var(--c-text-muted);">'
            f"{total} upcoming appointments scored"
            "</span>"
            "</div>"
            "</div>"
        )
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title="No-Show Risk Distribution", body_html=body)
        + "</div>"
    )


_BAND_COLOR: dict[str, str] = {
    "low":    PALETTE["healthy"],
    "medium": PALETTE["warning"],
    "high":   PALETTE["critical"],
}


def _no_show_row(bucket: NoShowRiskBucket) -> str:
    pct = bucket.fraction * 100.0
    color = _BAND_COLOR.get(bucket.band, PALETTE["accent"])
    return (
        '<div class="clarion-signal">'
        f'<div class="clarion-signal-name">{_esc(bucket.band.title())}</div>'
        '<div class="clarion-signal-track">'
        f'<div class="clarion-signal-fill" style="width: {pct:.1f}%; '
        f'background: {color};"></div>'
        "</div>"
        f'<div class="clarion-signal-value">{bucket.count} ({pct:.0f}%)</div>'
        "</div>"
    )


# ---------- eligibility panel ----------


_ELIG_COLOR: dict[str, str] = {
    "active":  PALETTE["healthy"],
    "pending": PALETTE["warning"],
    "denied":  PALETTE["critical"],
    "unknown": PALETTE["border_strong"],
}


def _eligibility_panel(
    summaries: list[EligibilitySummary], total: int
) -> str:
    if total == 0:
        body = _empty_message(
            "Eligibility coverage reads from SQLite — no records yet."
        )
        return (
            '<div style="flex: 1 1 0; min-width: 0;">'
            + c.panel(title="Eligibility Coverage", body_html=body)
            + "</div>"
        )

    donut = _eligibility_donut(summaries, total)
    legend_rows = "".join(_eligibility_legend_row(s) for s in summaries)
    legend = (
        '<div class="clarion-stack" style="gap: 4px;">' + legend_rows + "</div>"
    )
    body = (
        '<div class="clarion-row" style="gap: 24px; align-items: center;">'
        + donut
        + legend
        + "</div>"
    )
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title="Eligibility Coverage", body_html=body)
        + "</div>"
    )


def _eligibility_donut(
    summaries: list[EligibilitySummary], total: int
) -> str:
    """Inline-SVG donut chart. Each slice subtends an arc
    proportional to its fraction. Lazy approach: build the arcs as
    rotated stroke-dash patches over a single circle."""
    cx, cy, r = 60.0, 60.0, 48.0
    import math
    circumference = 2 * math.pi * r
    slices: list[str] = []
    rotation = -90.0  # start at top
    for s in summaries:
        arc_len = circumference * s.fraction
        gap = circumference - arc_len
        color = _ELIG_COLOR.get(s.status, PALETTE["accent"])
        slices.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" '
            f'fill="none" stroke="{color}" stroke-width="20" '
            f'stroke-dasharray="{arc_len:.2f} {gap:.2f}" '
            f'transform="rotate({rotation:.2f} {cx} {cy})"/>'
        )
        rotation += s.fraction * 360.0
    return (
        '<svg viewBox="0 0 120 120" width="120" height="120" '
        'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        + "".join(slices)
        + f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" '
        f'font-size="16" font-weight="600" fill="{PALETTE["text_strong"]}" '
        f'font-family="JetBrains Mono, monospace">{total}</text>'
        f'<text x="{cx}" y="{cy + 18}" text-anchor="middle" '
        f'font-size="9" fill="{PALETTE["text_muted"]}">total</text>'
        "</svg>"
    )


def _eligibility_legend_row(s: EligibilitySummary) -> str:
    color = _ELIG_COLOR.get(s.status, PALETTE["accent"])
    pct = s.fraction * 100.0
    return (
        '<div class="clarion-row" style="gap: 8px; align-items: center;">'
        f'<span style="width: 10px; height: 10px; background: {color}; '
        f'border-radius: 2px;"></span>'
        f'<span style="font-size: var(--fs-sm); color: var(--c-text);">'
        f"{_esc(s.status.title())}</span>"
        f'<span style="font-size: var(--fs-xs); color: var(--c-text-muted); '
        f'font-family: var(--font-mono); margin-left: auto;">'
        f"{s.count} ({pct:.0f}%)</span>"
        "</div>"
    )


# ---------- PMS panel ----------


def _pms_panel(tasks: list[PmsTaskRow]) -> str:
    if not tasks:
        body = _empty_message(
            "No PMS writeback tasks on disk. Module M1 produces these — "
            "enable modules.pms_writeback: true on a customer to populate "
            "this panel."
        )
        return c.panel(title="PMS Task Queue", body_html=body)

    header = (
        '<div class="clarion-row" style="gap: 12px; padding: 8px 12px; '
        'border-bottom: 1px solid var(--c-border);">'
        '<div style="flex: 0 0 80px; font-size: var(--fs-xs); '
        'color: var(--c-text-muted); text-transform: uppercase; '
        'letter-spacing: 0.04em;">Priority</div>'
        '<div style="flex: 1 1 0; min-width: 0; font-size: var(--fs-xs); '
        'color: var(--c-text-muted); text-transform: uppercase; '
        'letter-spacing: 0.04em;">Subject</div>'
        '<div style="flex: 0 0 120px; font-size: var(--fs-xs); '
        'color: var(--c-text-muted); text-transform: uppercase; '
        'letter-spacing: 0.04em;">Assignee</div>'
        '<div style="flex: 0 0 100px; font-size: var(--fs-xs); '
        'color: var(--c-text-muted); text-transform: uppercase; '
        'letter-spacing: 0.04em;">Created</div>'
        "</div>"
    )
    rows = "".join(_pms_row(t) for t in tasks)
    body = (
        '<div class="clarion-stack" style="gap: 0;">' + header + rows + "</div>"
    )
    return c.panel(title="PMS Task Queue", body_html=body)


def _pms_row(task: PmsTaskRow) -> str:
    badge = c.status_badge(
        "critical" if task.priority == "urgent" else "info",
        label=task.priority.upper(),
    )
    return (
        '<div class="clarion-row" style="gap: 12px; padding: 8px 12px; '
        'border-bottom: 1px solid var(--c-border);">'
        f'<div style="flex: 0 0 80px;">{badge}</div>'
        f'<div style="flex: 1 1 0; min-width: 0; font-size: var(--fs-sm); '
        f'color: var(--c-text-strong); overflow: hidden; text-overflow: ellipsis; '
        f'white-space: nowrap;">{_esc(task.subject)}</div>'
        f'<div style="flex: 0 0 120px; font-size: var(--fs-xs); '
        f'color: var(--c-text-muted); font-family: var(--font-mono);">'
        f"{_esc(task.assignee_group)}</div>"
        f'<div style="flex: 0 0 100px; font-size: var(--fs-xs); '
        f'color: var(--c-text-muted); font-family: var(--font-mono);">'
        f"{_esc(task.created_at)}</div>"
        "</div>"
    )


# ---------- helpers ----------


def _band_inverse(value: float, *, healthy: float, warning: float) -> c.Status:
    if value <= healthy:
        return "healthy"
    if value <= warning:
        return "warning"
    return "critical"


def _band_balanced(value: float) -> c.Status:
    """Utilisation is healthiest in the middle (60-85%); too low =
    under-booked, too high = over-booked / no walk-in capacity."""
    if 0.60 <= value <= 0.85:
        return "healthy"
    if 0.30 <= value < 0.60 or 0.85 < value <= 0.95:
        return "warning"
    return "critical"


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


__all__ = ["build_html"]
