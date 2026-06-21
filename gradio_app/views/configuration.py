"""Configuration - the Platform section's per-tenant settings view.

Reads the active tenant's YAML config (``configs/<customer>.yaml``)
through :func:`clarion.config.loader.load_customer` and renders
the operational knobs that drive the rest of the platform:
display name, enabled tools, escalation thresholds, languages,
agent persona. All read-only - this is the "look what's wired"
view, not an editor.
"""

from __future__ import annotations

import html as _html

from gradio_app import components as c


def build_html(customer_id: str) -> str:
    """Build the Configuration HTML for one tenant.

    Falls back to an empty-state panel if the config can't be
    loaded - the renderer should never crash the tab.
    """
    try:
        from clarion.config.loader import load_customer
        cfg = load_customer(customer_id)
    except Exception as e:
        return empty_html(detail=str(e))

    return (
        '<div class="clarion-stack" style="gap: 20px;">'
        + _section_title(
            title="Configuration",
            subtitle=(
                f"Tenant YAML drives the platform · "
                f"{_esc(customer_id)}"
            ),
        )
        + _identity_panel(cfg)
        + '<div class="clarion-row" style="align-items: stretch; gap: 16px;">'
        + _tools_panel(cfg)
        + _escalation_panel(cfg)
        + "</div>"
        + _persona_panel(cfg)
        + "</div>"
    )


def empty_html(*, detail: str = "") -> str:
    body = (
        '<div class="clarion-stack" '
        'style="align-items: center; gap: 8px; padding: 24px 16px;">'
        '<div style="font-size: var(--fs-lg); color: var(--c-text);">'
        "Configuration unavailable."
        "</div>"
    )
    if detail:
        body += (
            f'<div style="font-size: var(--fs-xs); '
            f'color: var(--c-text-muted); font-family: var(--font-mono);">'
            f"{_esc(detail)}</div>"
        )
    body += "</div>"
    return (
        '<div class="clarion-stack" style="gap: 20px;">'
        + _section_title(
            title="Configuration",
            subtitle="Tenant YAML driver",
        )
        + c.panel(title="Status", body_html=body)
        + "</div>"
    )


# ---------- sections ----------


def _section_title(*, title: str, subtitle: str) -> str:
    return (
        '<div class="clarion-stack" style="gap: 2px;">'
        f'<div style="font-size: var(--fs-2xl); font-weight: var(--fw-bold); '
        f'color: var(--c-text-strong); letter-spacing: -0.01em;">'
        f"{_esc(title)}</div>"
        f'<div style="font-size: var(--fs-sm); color: var(--c-text-muted);">'
        f"{_esc(subtitle)}</div>"
        "</div>"
    )


def _identity_panel(cfg) -> str:  # type: ignore[no-untyped-def]
    facts = "".join(
        _fact_row(label, value)
        for label, value in (
            ("Customer ID", cfg.customer_id),
            ("Display name", cfg.display_name),
            (
                "Specialties",
                ", ".join(cfg.specialties[:4])
                + ("..." if len(cfg.specialties) > 4 else ""),
            ),
            ("Languages", ", ".join(lang.upper() for lang in cfg.languages)),
            ("Rules path", str(cfg.rules_path)),
        )
    )
    body = (
        '<div style="display: grid; '
        "grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); "
        'gap: 12px 24px;">' + facts + "</div>"
    )
    return c.panel(title="Identity", body_html=body)


def _fact_row(label: str, value: str) -> str:
    return (
        '<div class="clarion-stack" style="gap: 2px;">'
        f'<div style="font-size: 10px; color: var(--c-text-muted); '
        f"text-transform: uppercase; letter-spacing: 0.06em; "
        f'font-weight: var(--fw-semibold);">{_esc(label)}</div>'
        f'<div style="font-size: var(--fs-sm); color: var(--c-text);">'
        f"{_esc(value)}</div>"
        "</div>"
    )


def _tools_panel(cfg) -> str:  # type: ignore[no-untyped-def]
    """Each tool name renders as a chip. The colour shows
    intent: a small green dot on the left = "this LLM call sees
    this tool in its registry"."""
    chips = "".join(_tool_chip(t) for t in cfg.enabled_tools)
    body = (
        '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
        + chips
        + "</div>"
    )
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(
            title=f"Enabled Tools ({len(cfg.enabled_tools)})",
            body_html=body,
        )
        + "</div>"
    )


def _tool_chip(name: str) -> str:
    return (
        '<span style="display: inline-flex; align-items: center; gap: 8px; '
        "padding: 6px 12px; border-radius: var(--r-md); "
        "background: var(--c-bg-subtle); border: 1px solid var(--c-border); "
        'font-size: var(--fs-xs); color: var(--c-text); '
        'font-family: var(--font-mono);">'
        '<span style="width: 6px; height: 6px; border-radius: 50%; '
        'background: var(--c-healthy);"></span>'
        f"{_esc(name)}"
        "</span>"
    )


def _escalation_panel(cfg) -> str:  # type: ignore[no-untyped-def]
    """The three escalation knobs as horizontal threshold bars
    so the operator can see how strict each one is."""
    thr = cfg.escalation
    rows = "".join(
        _threshold_row(label, value, hint)
        for label, value, hint in (
            (
                "Low-confidence trigger",
                float(thr.low_confidence),
                "Self-reported confidence below this -> escalate",
            ),
            (
                "Frustration trigger",
                float(thr.frustration),
                "Sentinel frustration score above this -> escalate",
            ),
            (
                "Max clarifications",
                float(thr.max_clarifications) / 10.0,
                f"After {thr.max_clarifications} clarifying turns -> escalate",
            ),
        )
    )
    body = '<div class="clarion-stack" style="gap: 12px;">' + rows + "</div>"
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title="Escalation Thresholds", body_html=body)
        + "</div>"
    )


def _threshold_row(label: str, value: float, hint: str) -> str:
    pct = max(0.0, min(1.0, value)) * 100.0
    if value >= 0.85:
        color = "var(--c-critical)"
    elif value >= 0.6:
        color = "var(--c-warning)"
    else:
        color = "var(--c-accent)"
    return (
        '<div class="clarion-stack" style="gap: 4px;">'
        '<div style="display: flex; justify-content: space-between; '
        'align-items: baseline;">'
        f'<span style="font-size: var(--fs-sm); color: var(--c-text); '
        f'font-weight: var(--fw-medium);">{_esc(label)}</span>'
        '<span style="font-family: var(--font-mono); '
        'font-size: var(--fs-xs); color: var(--c-text-strong);">'
        f"{value:.2f}</span>"
        "</div>"
        '<div style="height: 6px; background: var(--c-bg-subtle); '
        'border-radius: 999px; overflow: hidden;">'
        f'<div style="height: 100%; width: {pct:.1f}%; '
        f'background: {color}; border-radius: 999px;"></div>'
        "</div>"
        f'<div style="font-size: 10px; color: var(--c-text-muted);">'
        f"{_esc(hint)}</div>"
        "</div>"
    )


def _persona_panel(cfg) -> str:  # type: ignore[no-untyped-def]
    persona = cfg.agent_persona
    # Render preserving newlines but escaping HTML.
    safe = _esc(persona).replace("\n", "<br>")
    body = (
        '<div style="font-family: var(--font-mono); '
        "font-size: var(--fs-xs); color: var(--c-text); "
        "line-height: 1.6; background: var(--c-bg); "
        "padding: 12px 14px; border-radius: var(--r-md); "
        "border: 1px solid var(--c-border); "
        'white-space: pre-wrap;">'
        f"{safe}"
        "</div>"
    )
    return c.panel(title="Agent Persona", body_html=body)


def _esc(text: str) -> str:
    return _html.escape(text, quote=True)


__all__ = ["build_html", "empty_html"]
