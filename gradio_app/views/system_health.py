"""System Health — the Platform section's status board.

Three zones, top to bottom:

1. **Overall pulse header** — single sentence + status badge that
   summarises the worst subsystem.
2. **Subsystem grid** — one row per subsystem (API service, LLM
   provider, FAISS retriever, voice, sentinel, customer
   registry), each with a coloured indicator dot + status note
   + tiny metric tag.
3. **Resource bars** — CPU / memory / storage / concurrency
   utilisation as horizontal bars; synthetic values today.

All HTML built from a typed `SystemHealthSnapshot`; no I/O here.
"""

from __future__ import annotations

import html as _html

from gradio_app import components as c
from gradio_app.data_sources import (
    ResourceMetric,
    SubsystemStatus,
    SystemHealthSnapshot,
)


def build_html(snap: SystemHealthSnapshot) -> str:
    return (
        '<div class="clarion-stack" style="gap: 20px;">'
        + c.page_intro(
            title="System Health",
            what="Every subsystem's status, latency, and last heartbeat.",
            quote="Trust starts with what you can verify.",
        )
        + _header(snap)
        + _subsystem_panel(snap.subsystems)
        + _resource_panel(snap.resources)
        + "</div>"
    )


def _header(snap: SystemHealthSnapshot) -> str:
    color = {
        "healthy": "var(--c-healthy)",
        "warning": "var(--c-warning)",
        "critical": "var(--c-critical)",
        "unknown": "var(--c-text-muted)",
    }[snap.overall]
    label = {
        "healthy": "All systems operational",
        "warning": "Degraded — some subsystems in soft fallback",
        "critical": "Critical — one or more subsystems unavailable",
        "unknown": "Status unknown",
    }[snap.overall]
    return (
        '<div class="clarion-stack" style="gap: 6px;">'
        '<div style="display: flex; align-items: center; gap: 12px;">'
        f'<span style="width: 12px; height: 12px; border-radius: 50%; '
        f"background: {color}; box-shadow: 0 0 0 4px "
        f'rgba(6, 182, 212, 0.10);"></span>'
        '<div style="font-size: var(--fs-2xl); font-weight: var(--fw-bold); '
        'color: var(--c-text-strong); letter-spacing: -0.01em;">'
        "System Health"
        "</div>"
        "</div>"
        f'<div style="font-size: var(--fs-sm); color: {color}; '
        f'font-weight: var(--fw-medium);">{_esc(label)}</div>'
        f'<div style="font-size: 11px; color: var(--c-text-muted);">'
        f"Build v{_esc(snap.version_display)} · "
        f"{len(snap.subsystems)} subsystems monitored"
        "</div>"
        "</div>"
    )


def _subsystem_panel(rows: tuple[SubsystemStatus, ...]) -> str:
    body = '<div class="clarion-stack" style="gap: 6px;">' + "".join(
        _subsystem_row(s) for s in rows
    ) + "</div>"
    return c.panel(title="Subsystems", body_html=body)


def _subsystem_row(s: SubsystemStatus) -> str:
    color = {
        "healthy": "var(--c-healthy)",
        "warning": "var(--c-warning)",
        "critical": "var(--c-critical)",
        "unknown": "var(--c-text-muted)",
    }[s.status]
    badge_label = {
        "healthy": "OPERATIONAL",
        "warning": "DEGRADED",
        "critical": "DOWN",
        "unknown": "UNKNOWN",
    }[s.status]
    return (
        '<div style="display: grid; '
        "grid-template-columns: 12px 1fr auto auto; column-gap: 12px; "
        "padding: 10px 12px; border-radius: var(--r-md); "
        "background: var(--c-bg-panel); align-items: center; "
        'border: 1px solid var(--c-border);">'
        f'<span style="width: 10px; height: 10px; border-radius: 50%; '
        f'background: {color};"></span>'
        '<div class="clarion-stack" style="gap: 2px;">'
        f'<div style="font-size: var(--fs-sm); color: var(--c-text); '
        f'font-weight: var(--fw-semibold);">{_esc(s.name)}</div>'
        f'<div style="font-size: var(--fs-xs); color: var(--c-text-muted);">'
        f"{_esc(s.note)}</div>"
        "</div>"
        '<div style="font-family: var(--font-mono); font-size: 11px; '
        'color: var(--c-text-muted);">'
        f"{_esc(s.metric_display)}</div>"
        '<span style="font-size: 10px; padding: 2px 8px; '
        "border-radius: var(--r-sm); "
        f"background: rgba(0, 0, 0, 0.25); color: {color}; "
        "text-transform: uppercase; letter-spacing: 0.08em; "
        'font-weight: var(--fw-bold);">'
        f"{badge_label}"
        "</span>"
        "</div>"
    )


def _resource_panel(rows: tuple[ResourceMetric, ...]) -> str:
    body = '<div class="clarion-stack" style="gap: 12px;">' + "".join(
        _resource_row(r) for r in rows
    ) + "</div>"
    return c.panel(title="Resource Utilisation", body_html=body)


def _resource_row(r: ResourceMetric) -> str:
    pct = max(0.0, min(1.0, r.used_pct))
    pct_display = int(round(pct * 100))
    if pct >= 0.85:
        bar_color = "var(--c-critical)"
    elif pct >= 0.65:
        bar_color = "var(--c-warning)"
    else:
        bar_color = "var(--c-accent)"
    return (
        '<div class="clarion-stack" style="gap: 4px;">'
        '<div style="display: flex; justify-content: space-between; '
        'align-items: baseline;">'
        f'<span style="font-size: var(--fs-sm); color: var(--c-text); '
        f'font-weight: var(--fw-medium);">{_esc(r.name)}</span>'
        '<span style="font-size: 11px; font-family: var(--font-mono); '
        'color: var(--c-text-muted);">'
        f"{_esc(r.detail)}</span>"
        "</div>"
        '<div style="height: 6px; background: var(--c-bg-subtle); '
        'border-radius: 999px; overflow: hidden;">'
        f'<div style="height: 100%; width: {pct_display}%; '
        f"background: {bar_color}; "
        f'border-radius: 999px;"></div>'
        "</div>"
        "</div>"
    )


def _esc(text: str) -> str:
    return _html.escape(text, quote=True)


__all__ = ["build_html"]
