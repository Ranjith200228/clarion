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
from datetime import UTC, datetime

from gradio_app import components as c
from gradio_app.data_sources import (
    Patient360Snapshot,
    PatientProfile,
    PatientTimelineEvent,
)
from gradio_app.views._confirmation import (
    ConfirmationContext,
    confirmation_data_uri,
    default_instructions,
)

# ---------- public ----------


def build_html(snap: Patient360Snapshot) -> str:
    if not snap.patients or snap.selected is None:
        return empty_html()
    selected = snap.selected
    return (
        '<div class="clarion-stack" style="gap: 20px;">'
        + c.page_intro(
            title="Patient 360",
            what=(
                "Every fact this tenant knows about a patient — "
                "chart, history, payer, care team, appointments."
            ),
            quote="Care, in context.",
        )
        + _picker(snap.patients, selected)
        + _profile_card(selected)
        + '<div class="clarion-row" style="align-items: stretch; gap: 16px;">'
        + _care_team_panel(selected)
        + _insurance_panel(selected)
        + "</div>"
        + _confirmation_panel(selected, snap.customer_id)
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
        + c.page_intro(
            title="Patient 360",
            what=(
                "Every fact this tenant knows about a patient — "
                "chart, history, payer, care team, appointments."
            ),
            quote="Care, in context.",
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
    """Roster picker with H10 client-side search + payer filter.

    Each chip carries data-name / data-payer / data-risk attributes
    the inline JS handler reads to toggle ``display: none``. The
    counter at the right of the filter bar updates live.

    All JS is inline + scoped via the wrapping element id so it
    doesn't conflict with anything else on the page; no script
    elsewhere in the app reads or writes the same handles.
    """
    chips = "".join(_chip(p, p.patient_id == selected.patient_id) for p in patients)
    payers = sorted({
        p.insurance.payer for p in patients if p.insurance is not None
    })
    payer_options = "".join(
        f'<option value="{_esc(payer)}">{_esc(payer)}</option>'
        for payer in payers
    )
    total = len(patients)
    # Inline JS: read input + select, toggle chip visibility, update
    # counter. Scoped by container id so it stays self-contained.
    js = (
        "(function(){"
        "var box = document.getElementById('clarion-p360-roster');"
        "if(!box) return;"
        "var q = (box.querySelector('.clarion-p360-search') || {}).value;"
        "var payer = (box.querySelector('.clarion-p360-payer') || {}).value;"
        "q = (q || '').toLowerCase().trim();"
        "var chips = box.querySelectorAll('.clarion-p360-chip');"
        "var shown = 0;"
        "chips.forEach(function(c){"
        "  var name = (c.dataset.name || '').toLowerCase();"
        "  var pid = (c.dataset.pid || '').toLowerCase();"
        "  var p = c.dataset.payer || '';"
        "  var matchQ = !q || name.indexOf(q) > -1 || pid.indexOf(q) > -1;"
        "  var matchPayer = !payer || p === payer;"
        "  var show = matchQ && matchPayer;"
        "  c.style.display = show ? '' : 'none';"
        "  if(show) shown++;"
        "});"
        "var counter = box.querySelector('.clarion-p360-counter');"
        f"if(counter) counter.textContent = shown + ' of {total}';"
        "})()"
    )
    return (
        '<div id="clarion-p360-roster" class="clarion-stack" '
        'style="gap: 10px;">'
        # Filter bar.
        '<div style="display: flex; flex-wrap: wrap; align-items: center; '
        'gap: 10px;">'
        '<input class="clarion-p360-search" type="search" '
        'placeholder="Search by name or patient id..." '
        f'oninput="{js}" '
        'style="flex: 1 1 220px; min-width: 200px; padding: 8px 12px; '
        "border-radius: var(--r-md); border: 1px solid var(--c-border); "
        "background: var(--c-bg-panel); color: var(--c-text-strong); "
        'font-size: var(--fs-sm);">'
        '<select class="clarion-p360-payer" '
        f'onchange="{js}" '
        'style="padding: 8px 12px; border-radius: var(--r-md); '
        "border: 1px solid var(--c-border); "
        "background: var(--c-bg-panel); color: var(--c-text); "
        'font-size: var(--fs-sm);">'
        '<option value="">All payers</option>'
        + payer_options
        + "</select>"
        '<span class="clarion-p360-counter" '
        'style="font-family: var(--font-mono); font-size: var(--fs-xs); '
        "color: var(--c-text-muted); padding: 4px 10px; "
        "background: var(--c-bg-subtle); border-radius: var(--r-sm); "
        'white-space: nowrap;">'
        f"{total} of {total}"
        "</span>"
        "</div>"
        # Chip strip.
        '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
        + chips
        + "</div>"
        "</div>"
    )


def _chip(p: PatientProfile, is_selected: bool) -> str:
    bg = "var(--c-bg-subtle)" if is_selected else "var(--c-bg-panel)"
    border = "var(--c-accent)" if is_selected else "var(--c-border)"
    color = "var(--c-accent)" if is_selected else "var(--c-text)"
    weight = "var(--fw-semibold)" if is_selected else "var(--fw-medium)"
    payer = p.insurance.payer if p.insurance is not None else ""
    risk_band = (
        "high"
        if p.trust_score < 0.5
        else "medium"
        if p.trust_score < 0.75
        else "low"
    )
    return (
        '<div class="clarion-p360-chip" '
        f'data-name="{_esc(p.display_name)}" '
        f'data-pid="{_esc(p.patient_id)}" '
        f'data-payer="{_esc(payer)}" '
        f'data-risk="{risk_band}" '
        f'style="display: inline-flex; align-items: center; gap: 8px; '
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


def _confirmation_panel(p: PatientProfile, customer_id: str) -> str:
    """Render a printable appointment-confirmation preview with
    a download link.

    Picks the patient's nearest *upcoming* appointment from
    their timeline. If they have no upcoming appointment, the
    panel renders a friendly empty state instead of vanishing -
    a viewer should always see this section so the download
    affordance is discoverable.
    """
    now = datetime.now(UTC)

    # Prefer the nearest UPCOMING appointment, but fall back to
    # the most recent past one so the download affordance always
    # shows up when a patient has any appointment history. The
    # confirmation document calls out past vs upcoming clearly
    # so the UX still reads correctly either way.
    upcoming: PatientTimelineEvent | None = None
    most_recent_past: PatientTimelineEvent | None = None
    for e in p.timeline:
        if e.kind != "appointment":
            continue
        ts = e.ts if e.ts.tzinfo is not None else e.ts.replace(tzinfo=UTC)
        if ts >= now:
            if upcoming is None or ts < upcoming.ts:
                upcoming = e
        else:
            if most_recent_past is None or ts > most_recent_past.ts:
                most_recent_past = e
    chosen = upcoming or most_recent_past
    is_upcoming = chosen is not None and chosen is upcoming

    if chosen is None:
        body = (
            '<div style="font-size: var(--fs-sm); color: var(--c-text-muted);">'
            "No appointment on file for this patient — once one is "
            "booked, a downloadable confirmation will appear here."
            "</div>"
        )
        return c.panel(title="Appointment Confirmation", body_html=body)

    upcoming = chosen  # rename for the rest of the function

    # Parse appointment type + provider name from the timeline
    # event's title/detail (set by data_sources._patient_360_from_store).
    # Title shape: "Appointment booked: <type>"; detail shape:
    # "<type> with <provider> — <notes>".
    apt_type = upcoming.title.split(":", 1)[-1].strip() or "Appointment"
    provider_name = ""
    if " with " in upcoming.detail:
        after = upcoming.detail.split(" with ", 1)[1]
        provider_name = after.split(" — ", 1)[0].strip()
    if not provider_name and p.care_team:
        provider_name = p.care_team[0].name
    if not provider_name:
        provider_name = "Care team"

    ctx = ConfirmationContext(
        customer_display_name=customer_id.title() + " Practice",
        customer_id=customer_id,
        patient_id=p.patient_id,
        patient_name=p.display_name,
        patient_dob=p.dob_display,
        patient_phone=p.phone_display,
        patient_email=p.email,
        patient_address=p.address,
        patient_language=p.preferred_language,
        payer=p.insurance.payer if p.insurance else "Self-pay",
        plan=p.insurance.plan if p.insurance else "—",
        member_id=p.insurance.member_id if p.insurance else "—",
        eligibility_status=p.insurance.eligibility_status if p.insurance else "unknown",
        provider_name=provider_name,
        appointment_type=apt_type,
        appointment_at=upcoming.ts,
        duration_minutes=30,
        appointment_id=f"appt_{p.patient_id}_{int(upcoming.ts.timestamp())}",
        location=f"{customer_id.title()} Front Desk · Suite 200",
        instructions=default_instructions(apt_type),
        generated_at=now,
    )

    uri = confirmation_data_uri(ctx)
    when = ctx.appointment_at.strftime("%a, %b %d %Y · %I:%M %p")
    filename = (
        f"confirmation_{p.patient_id}_"
        f"{ctx.appointment_at.strftime('%Y%m%d')}.html"
    )

    body = (
        '<div class="clarion-stack" style="gap: 14px;">'
        # Compact preview card.
        '<div style="display: grid; grid-template-columns: 1fr auto; '
        "gap: 16px; align-items: center; padding: 14px 16px; "
        "background: var(--c-bg-subtle); border: 1px solid var(--c-border); "
        'border-radius: var(--r-md);">'
        '<div class="clarion-stack" style="gap: 4px;">'
        '<div style="font-size: 10px; color: var(--c-accent); '
        "text-transform: uppercase; letter-spacing: 0.08em; "
        f'font-weight: var(--fw-bold);">{"Next appointment" if is_upcoming else "Most recent appointment"}</div>'
        f'<div style="font-size: var(--fs-lg); color: var(--c-text-strong); '
        f'font-weight: var(--fw-semibold);">{_esc(apt_type)}</div>'
        f'<div style="font-size: var(--fs-sm); color: var(--c-text);">'
        f"{_esc(when)} · with {_esc(provider_name)}"
        "</div>"
        f'<div style="font-size: var(--fs-xs); color: var(--c-text-muted);">'
        f"{_esc(ctx.location)}"
        "</div>"
        "</div>"
        # Download link styled as a button.
        '<a href="' + uri + '" '
        f'download="{filename}" '
        'style="display: inline-flex; align-items: center; gap: 8px; '
        "padding: 10px 16px; border-radius: var(--r-md); "
        "background: var(--c-accent); color: white; "
        "text-decoration: none; font-size: var(--fs-sm); "
        "font-weight: var(--fw-semibold); white-space: nowrap; "
        'box-shadow: 0 1px 0 rgba(255,255,255,0.10) inset;">'
        # Download icon (inline SVG).
        '<svg width="14" height="14" viewBox="0 0 16 16" '
        'xmlns="http://www.w3.org/2000/svg" fill="none" '
        'stroke="currentColor" stroke-width="1.75" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M8 2v9"/>'
        '<path d="M4 7l4 4 4-4"/>'
        '<path d="M2 14h12"/>'
        "</svg>"
        "Download confirmation"
        "</a>"
        "</div>"
        # Notice strip beneath the card.
        '<div style="font-size: var(--fs-xs); color: var(--c-text-muted); '
        'line-height: 1.5;">'
        "The downloaded HTML file is fully self-contained — open it in any "
        "browser to view, print, or attach to a patient record. Includes "
        "the patient profile, provider, time/location, insurance details, "
        "and appointment-specific prep instructions."
        "</div>"
        "</div>"
    )
    return c.panel(title="Appointment Confirmation", body_html=body)


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
