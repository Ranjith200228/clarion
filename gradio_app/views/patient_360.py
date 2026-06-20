"""Patient 360 — per-patient longitudinal record view.

Four zones, top to bottom:

1. **Patient picker strip** — chip-row of patients in this tenant
   so a viewer can quickly switch profiles. Selected chip is
   highlighted; clicking is purely visual today (the dropdown is
   the source of truth for "selected").
2. **Profile card** — name, DOB/age, phone, email, address, language,
   engagement / sentiment / trust scores rendered as gauge tiles.
3. **Care team + insurance** — two side-by-side panels with the
   patient's clinical team and their payer / member / eligibility
   state.
4. **Journey timeline** — chronological list of touchpoints
   (voice calls, appointments, eligibility checks, escalations,
   PMS tasks, care-team notes), each color-tinted by severity.

All HTML is built from typed `data_sources.Patient360Snapshot` —
this module never reads disk and never touches Gradio components.
"""

from __future__ import annotations

import html as _html

from gradio_app import components as c
from gradio_app.data_sources import (
    Patient360Snapshot,
    PatientProfile,
    PatientTimelineEvent,
)

# ---------- public ----------


def build_html(snap: Patient360Snapshot) -> str:
    if not snap.patients or snap.selected is None:
        return empty_html()
    selected = snap.selected
    return (
        '<div class="clarion-stack" style="gap: 20px;">'
        + _section_title(
            title="Patient 360",
            subtitle=(
                f"Unified longitudinal record · "
                f"{_esc(snap.customer_id)} tenant"
            ),
        )
        + _picker(snap.patients, selected)
        + _profile_card(selected)
        + '<div class="clarion-row" style="align-items: stretch; gap: 16px;">'
        + _care_team_panel(selected)
        + _insurance_panel(selected)
        + "</div>"
        + _timeline_panel(selected)
        + "</div>"
    )


def empty_html() -> str:
    body = (
        '<div class="clarion-stack" '
        'style="align-items: center; gap: 12px; padding: 32px 16px;">'
        '<div style="font-size: var(--fs-lg); color: var(--c-text);">'
        "No patients on file for this tenant."
        "</div>"
        '<div style="font-size: var(--fs-sm); color: var(--c-text-muted);">'
        "Run the simulator to seed a synthetic roster."
        "</div>"
        "</div>"
    )
    return (
        '<div class="clarion-stack" style="gap: 20px;">'
        + _section_title(
            title="Patient 360",
            subtitle="Unified longitudinal record",
        )
        + c.panel(title="Roster", body_html=body)
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


def _picker(
    patients: tuple[PatientProfile, ...],
    selected: PatientProfile,
) -> str:
    chips = "".join(_chip(p, p.patient_id == selected.patient_id) for p in patients)
    return (
        '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
        + chips
        + "</div>"
    )


def _chip(p: PatientProfile, is_selected: bool) -> str:
    bg = "var(--c-bg-subtle)" if is_selected else "var(--c-bg-panel)"
    border = "var(--c-accent)" if is_selected else "var(--c-border)"
    color = "var(--c-accent)" if is_selected else "var(--c-text)"
    weight = "var(--fw-semibold)" if is_selected else "var(--fw-medium)"
    return (
        f'<div style="display: inline-flex; align-items: center; gap: 8px; '
        f"padding: 6px 12px; border-radius: 999px; "
        f"background: {bg}; border: 1px solid {border}; "
        f'color: {color}; font-size: var(--fs-sm); font-weight: {weight};">'
        f'<span style="font-family: var(--font-mono); font-size: 11px; '
        f'opacity: 0.7;">{_esc(p.patient_id)}</span>'
        f"<span>{_esc(p.display_name)}</span>"
        "</div>"
    )


def _profile_card(p: PatientProfile) -> str:
    initials = "".join(part[0] for part in p.display_name.split()[:2]).upper()
    facts = "".join(
        _fact_row(label, value)
        for label, value in (
            ("Patient ID", p.patient_id),
            ("DOB", f"{p.dob_display} · age {p.age_years}"),
            ("Phone", p.phone_display),
            ("Email", p.email),
            ("Address", p.address),
            ("Language", p.preferred_language.upper()),
        )
    )
    scores = (
        '<div class="clarion-kpi-strip" style="gap: 12px;">'
        + _score_tile("Engagement", p.engagement_score)
        + _score_tile("Sentiment", p.sentiment_score)
        + _score_tile("Trust", p.trust_score)
        + "</div>"
    )
    body = (
        '<div style="display: grid; '
        "grid-template-columns: 64px 1fr; column-gap: 16px; row-gap: 16px; "
        'align-items: start;">'
        '<div style="width: 64px; height: 64px; border-radius: 50%; '
        "background: linear-gradient(135deg, #22D3EE 0%, #0E7490 100%); "
        "display: flex; align-items: center; justify-content: center; "
        "color: white; font-weight: var(--fw-bold); font-size: 22px; "
        'letter-spacing: 0.04em;">'
        f"{_esc(initials)}"
        "</div>"
        '<div class="clarion-stack" style="gap: 4px;">'
        f'<div style="font-size: var(--fs-xl); font-weight: var(--fw-bold); '
        f'color: var(--c-text-strong);">{_esc(p.display_name)}</div>'
        f'<div style="font-size: var(--fs-sm); color: var(--c-text-muted);">'
        f"{_esc(p.customer_id)} · {_esc(p.preferred_language.upper())}"
        "</div>"
        "</div>"
        '<div style="grid-column: 1 / -1;">'
        '<div style="display: grid; '
        "grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); "
        'gap: 8px 16px;">'
        + facts
        + "</div>"
        "</div>"
        '<div style="grid-column: 1 / -1;">' + scores + "</div>"
        "</div>"
    )
    return c.panel(title="Profile", body_html=body)


def _fact_row(label: str, value: str) -> str:
    return (
        '<div class="clarion-stack" style="gap: 2px;">'
        f'<div style="font-size: 11px; color: var(--c-text-muted); '
        f"text-transform: uppercase; letter-spacing: 0.04em; "
        f'font-weight: var(--fw-medium);">{_esc(label)}</div>'
        f'<div style="font-size: var(--fs-sm); color: var(--c-text);">'
        f"{_esc(value)}</div>"
        "</div>"
    )


def _score_tile(label: str, score: float) -> str:
    pct = max(0, min(100, int(round(score * 100))))
    if score >= 0.75:
        status = "healthy"
        color = "var(--c-healthy)"
    elif score >= 0.5:
        status = "info"
        color = "var(--c-accent)"
    elif score >= 0.3:
        status = "warning"
        color = "var(--c-warning)"
    else:
        status = "critical"
        color = "var(--c-critical)"
    return (
        f'<div class="clarion-kpi-tile" data-status="{status}" '
        'style="flex: 1; min-width: 120px;">'
        f'<div class="clarion-kpi-label">{_esc(label)}</div>'
        f'<div class="clarion-kpi-value" style="color: {color};">{pct}%</div>'
        f'<div class="clarion-kpi-delta" data-trend="flat">/100</div>'
        "</div>"
    )


def _care_team_panel(p: PatientProfile) -> str:
    if not p.care_team:
        body = (
            '<div style="font-size: var(--fs-sm); color: var(--c-text-muted);">'
            "No care-team members on file."
            "</div>"
        )
    else:
        rows = "".join(
            (
                '<div style="display: flex; align-items: center; gap: 12px; '
                "padding: 10px 12px; border-radius: var(--r-md); "
                f"background: {'var(--c-bg-subtle)' if m.is_primary else 'transparent'};"
                '">'
                '<div style="width: 32px; height: 32px; border-radius: 50%; '
                "background: var(--c-bg-subtle); display: flex; "
                "align-items: center; justify-content: center; color: var(--c-accent); "
                'font-weight: var(--fw-semibold); font-size: var(--fs-xs);">'
                + _esc("".join(part[0] for part in m.name.split()[:2]).upper())
                + "</div>"
                '<div class="clarion-stack" style="gap: 2px; flex: 1;">'
                f'<div style="font-size: var(--fs-sm); color: var(--c-text); '
                f'font-weight: var(--fw-medium);">{_esc(m.name)}</div>'
                f'<div style="font-size: 11px; color: var(--c-text-muted);">'
                f"{_esc(m.role)}</div>"
                "</div>"
                + (
                    '<span style="font-size: 10px; padding: 2px 6px; '
                    "border-radius: var(--r-sm); background: rgba(6, 182, 212, 0.15); "
                    "color: var(--c-accent); text-transform: uppercase; "
                    "letter-spacing: 0.06em; "
                    'font-weight: var(--fw-semibold);">PRIMARY</span>'
                    if m.is_primary
                    else ""
                )
                + "</div>"
            )
            for m in p.care_team
        )
        body = '<div class="clarion-stack" style="gap: 4px;">' + rows + "</div>"
    return c.panel(title="Care Team", body_html=body)


def _insurance_panel(p: PatientProfile) -> str:
    if p.insurance is None:
        body = (
            '<div style="font-size: var(--fs-sm); color: var(--c-text-muted);">'
            "No insurance information on file."
            "</div>"
        )
        return c.panel(title="Insurance", body_html=body)
    ins = p.insurance
    status_color = {
        "active": "var(--c-healthy)",
        "pending": "var(--c-warning)",
        "lapsed": "var(--c-critical)",
        "unknown": "var(--c-text-muted)",
    }[ins.eligibility_status]
    last = (
        ins.last_verified_at.strftime("%Y-%m-%d")
        if ins.last_verified_at is not None
        else "—"
    )
    rows = "".join(
        _fact_row(label, value)
        for label, value in (
            ("Payer", ins.payer),
            ("Plan", ins.plan),
            ("Member ID", ins.member_id),
            ("Last verified", last),
        )
    )
    body = (
        '<div class="clarion-stack" style="gap: 12px;">'
        '<div style="display: inline-flex; align-items: center; gap: 8px;">'
        f'<span style="width: 8px; height: 8px; border-radius: 50%; '
        f'background: {status_color};"></span>'
        f'<span style="font-size: var(--fs-sm); font-weight: var(--fw-semibold); '
        f"color: var(--c-text-strong); text-transform: uppercase; "
        f'letter-spacing: 0.04em;">{_esc(ins.eligibility_status)}</span>'
        "</div>"
        '<div style="display: grid; '
        "grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); "
        'gap: 8px 16px;">' + rows + "</div>"
        "</div>"
    )
    return c.panel(title="Insurance", body_html=body)


def _timeline_panel(p: PatientProfile) -> str:
    if not p.timeline:
        body = (
            '<div style="font-size: var(--fs-sm); color: var(--c-text-muted);">'
            "No timeline events recorded."
            "</div>"
        )
    else:
        rows = "".join(_timeline_row(e) for e in p.timeline)
        body = '<div class="clarion-stack" style="gap: 8px;">' + rows + "</div>"
    return c.panel(title="Journey Timeline", body_html=body)


def _timeline_row(e: PatientTimelineEvent) -> str:
    icon = {
        "voice_call": "🎙",
        "appointment": "📅",
        "eligibility": "🛡",
        "escalation": "⚠",
        "pms_task": "✓",
        "note": "✎",
    }.get(e.kind, "•")
    severity_color = {
        "healthy": "var(--c-healthy)",
        "info": "var(--c-accent)",
        "warning": "var(--c-warning)",
        "critical": "var(--c-critical)",
    }[e.severity]
    when = e.ts.strftime("%Y-%m-%d %H:%M UTC")
    return (
        '<div style="display: grid; '
        "grid-template-columns: 28px 1fr auto; column-gap: 12px; "
        "padding: 10px 12px; border-left: 3px solid "
        f"{severity_color}; "
        "background: var(--c-bg-panel); border-radius: var(--r-md); "
        'align-items: start;">'
        f'<div style="font-size: 16px; line-height: 1.2; '
        f'color: {severity_color};">{icon}</div>'
        '<div class="clarion-stack" style="gap: 2px;">'
        f'<div style="font-size: var(--fs-sm); color: var(--c-text); '
        f'font-weight: var(--fw-semibold);">{_esc(e.title)}</div>'
        f'<div style="font-size: var(--fs-xs); color: var(--c-text-muted);">'
        f"{_esc(e.detail)}</div>"
        "</div>"
        '<div style="font-family: var(--font-mono); font-size: 11px; '
        'color: var(--c-text-muted); white-space: nowrap;">'
        f"{_esc(when)}</div>"
        "</div>"
    )


def _esc(text: str) -> str:
    return _html.escape(text, quote=True)


__all__ = ["build_html", "empty_html"]
