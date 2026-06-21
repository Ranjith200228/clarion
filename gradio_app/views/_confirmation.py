"""Appointment confirmation document generator.

Builds a standalone, printable HTML document for one booked
appointment. The document embeds its own minimal CSS so it
prints cleanly on paper and looks consistent when opened
detached from the dashboard.

The patient_360 view wraps the output in a data: URI so the
browser can save it directly without a server endpoint -
zero extra Gradio wiring.
"""

from __future__ import annotations

import base64
import html as _html
from datetime import datetime
from typing import NamedTuple


class ConfirmationContext(NamedTuple):
    """Everything the confirmation needs in one immutable bundle."""

    customer_display_name: str
    customer_id: str
    patient_id: str
    patient_name: str
    patient_dob: str
    patient_phone: str
    patient_email: str
    patient_address: str
    patient_language: str
    payer: str
    plan: str
    member_id: str
    eligibility_status: str
    provider_name: str
    appointment_type: str
    appointment_at: datetime
    duration_minutes: int
    appointment_id: str
    location: str
    instructions: str
    generated_at: datetime


def build_confirmation_html(ctx: ConfirmationContext) -> str:
    """Return a self-contained printable HTML document.

    The output is a complete HTML page (with DOCTYPE, <head>,
    inline CSS). It's intended to be downloaded and opened in
    a browser or printed; it does NOT inherit the dashboard's
    CSS variables. Colours are hard-coded so the printed copy
    looks the same on every machine.
    """
    when_local = ctx.appointment_at.strftime("%A, %B %d, %Y · %I:%M %p")
    generated = ctx.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Appointment Confirmation · {_esc(ctx.patient_name)}</title>
<style>
  :root {{
    --ink: #0F172A;
    --ink-muted: #475569;
    --line: #E2E8F0;
    --bg: #FFFFFF;
    --accent: #0E7490;
    --accent-soft: #ECFEFF;
    --healthy: #047857;
    --warning: #B45309;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", Roboto, sans-serif;
    font-size: 14px;
    line-height: 1.55;
  }}
  .page {{
    max-width: 720px;
    margin: 32px auto;
    padding: 40px 44px;
    border: 1px solid var(--line);
    border-radius: 12px;
  }}
  header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 2px solid var(--accent);
    padding-bottom: 16px;
    margin-bottom: 24px;
  }}
  .brand {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .brand-mark {{
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: linear-gradient(135deg, #22D3EE 0%, #0E7490 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 17px;
  }}
  .brand-name {{
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
  }}
  .brand-name .accent {{ color: var(--accent); }}
  .brand-tag {{ font-size: 11px; color: var(--ink-muted); }}
  .meta {{
    font-size: 11px;
    color: var(--ink-muted);
    text-align: right;
    font-family: ui-monospace, monospace;
  }}
  h1 {{
    margin: 0 0 6px;
    font-size: 22px;
    color: var(--ink);
  }}
  .subtitle {{
    color: var(--ink-muted);
    margin-bottom: 28px;
  }}
  .section {{
    margin-bottom: 24px;
  }}
  .section-title {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-muted);
    font-weight: 700;
    margin-bottom: 10px;
  }}
  .when-card {{
    background: var(--accent-soft);
    border: 1px solid #A5F3FC;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 24px;
  }}
  .when-card .when-label {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: 4px;
  }}
  .when-card .when-value {{
    font-size: 20px;
    font-weight: 700;
    color: var(--ink);
  }}
  .when-card .when-detail {{
    font-size: 13px;
    color: var(--ink-muted);
    margin-top: 4px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px 24px;
  }}
  .fact {{
    border-bottom: 1px solid var(--line);
    padding-bottom: 8px;
  }}
  .fact-label {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-muted);
    font-weight: 600;
    margin-bottom: 2px;
  }}
  .fact-value {{
    font-size: 14px;
    color: var(--ink);
  }}
  .instructions {{
    background: #FEF3C7;
    border-left: 3px solid var(--warning);
    border-radius: 4px;
    padding: 12px 14px;
    font-size: 13px;
    color: var(--ink);
  }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
  }}
  .badge-healthy {{
    background: #D1FAE5;
    color: var(--healthy);
  }}
  .badge-warning {{
    background: #FEF3C7;
    color: var(--warning);
  }}
  footer {{
    border-top: 1px solid var(--line);
    margin-top: 28px;
    padding-top: 14px;
    font-size: 11px;
    color: var(--ink-muted);
    display: flex;
    justify-content: space-between;
  }}
  @media print {{
    body {{ background: white; }}
    .page {{ border: none; margin: 0; padding: 0; }}
  }}
</style>
</head>
<body>
<div class="page">
  <header>
    <div class="brand">
      <div class="brand-mark">C</div>
      <div>
        <div class="brand-name">Clarion <span class="accent">Vision</span></div>
        <div class="brand-tag">{_esc(ctx.customer_display_name)}</div>
      </div>
    </div>
    <div class="meta">
      Confirmation #{_esc(ctx.appointment_id)}<br>
      Generated {_esc(generated)}
    </div>
  </header>

  <h1>Appointment Confirmation</h1>
  <div class="subtitle">
    Please review the details below and arrive 10 minutes before
    your scheduled time. Bring a photo ID and your insurance card.
  </div>

  <div class="when-card">
    <div class="when-label">Scheduled for</div>
    <div class="when-value">{_esc(when_local)}</div>
    <div class="when-detail">
      {_esc(ctx.appointment_type)} ·
      {ctx.duration_minutes} minutes ·
      with {_esc(ctx.provider_name)}
    </div>
  </div>

  <div class="section">
    <div class="section-title">Patient</div>
    <div class="grid">
      <div class="fact">
        <div class="fact-label">Name</div>
        <div class="fact-value">{_esc(ctx.patient_name)}</div>
      </div>
      <div class="fact">
        <div class="fact-label">Patient ID</div>
        <div class="fact-value">{_esc(ctx.patient_id)}</div>
      </div>
      <div class="fact">
        <div class="fact-label">Date of Birth</div>
        <div class="fact-value">{_esc(ctx.patient_dob)}</div>
      </div>
      <div class="fact">
        <div class="fact-label">Preferred Language</div>
        <div class="fact-value">{_esc(ctx.patient_language.upper())}</div>
      </div>
      <div class="fact">
        <div class="fact-label">Phone</div>
        <div class="fact-value">{_esc(ctx.patient_phone)}</div>
      </div>
      <div class="fact">
        <div class="fact-label">Email</div>
        <div class="fact-value">{_esc(ctx.patient_email)}</div>
      </div>
    </div>
    <div class="fact" style="grid-column: 1 / -1; margin-top: 8px;">
      <div class="fact-label">Address</div>
      <div class="fact-value">{_esc(ctx.patient_address)}</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Visit</div>
    <div class="grid">
      <div class="fact">
        <div class="fact-label">Provider</div>
        <div class="fact-value">{_esc(ctx.provider_name)}</div>
      </div>
      <div class="fact">
        <div class="fact-label">Appointment Type</div>
        <div class="fact-value">{_esc(ctx.appointment_type)}</div>
      </div>
      <div class="fact">
        <div class="fact-label">Duration</div>
        <div class="fact-value">{ctx.duration_minutes} minutes</div>
      </div>
      <div class="fact">
        <div class="fact-label">Location</div>
        <div class="fact-value">{_esc(ctx.location)}</div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Insurance</div>
    <div class="grid">
      <div class="fact">
        <div class="fact-label">Payer</div>
        <div class="fact-value">{_esc(ctx.payer)}</div>
      </div>
      <div class="fact">
        <div class="fact-label">Plan</div>
        <div class="fact-value">{_esc(ctx.plan)}</div>
      </div>
      <div class="fact">
        <div class="fact-label">Member ID</div>
        <div class="fact-value">{_esc(ctx.member_id)}</div>
      </div>
      <div class="fact">
        <div class="fact-label">Eligibility</div>
        <div class="fact-value">
          <span class="badge badge-{
            'healthy' if ctx.eligibility_status == 'active' else 'warning'
          }">{_esc(ctx.eligibility_status.upper())}</span>
        </div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Important Instructions</div>
    <div class="instructions">{_esc(ctx.instructions)}</div>
  </div>

  <footer>
    <span>Clarion Vision Platform · automated confirmation</span>
    <span>Need to reschedule? Reply to your booking call or call the front desk.</span>
  </footer>
</div>
</body>
</html>
"""


def confirmation_data_uri(ctx: ConfirmationContext) -> str:
    """Build a `data:text/html;base64,...` URI carrying the confirmation
    HTML. A download link with this URI as its href + a `download`
    attribute lets the browser save the file with zero server logic.
    """
    html = build_confirmation_html(ctx)
    encoded = base64.b64encode(html.encode("utf-8")).decode("ascii")
    return f"data:text/html;charset=utf-8;base64,{encoded}"


def default_instructions(appointment_type: str) -> str:
    """Tailored prep notes per appointment type. Falls back to a
    generic message for unknown types."""
    lower = appointment_type.lower()
    if "cataract" in lower or "pre-op" in lower:
        return (
            "Do not eat or drink anything after midnight the night "
            "before. Bring a list of all current medications and "
            "arrange a ride home — you will not be able to drive after."
        )
    if "glaucoma" in lower or "dilation" in lower:
        return (
            "Your eyes will be dilated during the exam. Bring "
            "sunglasses and avoid driving for ~4 hours after. Bring "
            "your current medication list."
        )
    if "screening" in lower or "follow-up" in lower or "follow up" in lower:
        return (
            "Bring any recent test results or imaging. Pupil dilation "
            "is possible — consider arranging transportation."
        )
    if "consult" in lower:
        return (
            "Bring your insurance card, photo ID, and a list of "
            "questions or symptoms you would like to discuss."
        )
    return (
        "Arrive 10 minutes early to complete intake forms. Bring "
        "your photo ID and insurance card."
    )


def _esc(text: str) -> str:
    return _html.escape(str(text), quote=True)


__all__ = [
    "ConfirmationContext",
    "build_confirmation_html",
    "confirmation_data_uri",
    "default_instructions",
]
