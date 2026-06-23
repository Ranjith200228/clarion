"""Phase G Gradio Blocks app — Clarion mission-control shell.

Entry point::

    python -m gradio_app

Shell layout (Phase G):

  ┌───────────────────────────────────────────────────────────┐
  │ ◐ Clarion · v1.1.0 · LIVE                                 │
  ├───────────────────────────────────────────────────────────┤
  │ Customer: [ ophthalmology ▼ ]                             │
  ├──────────┬────────────────────────────────────────────────┤
  │ NAV      │ MAIN CANVAS (single active view)               │
  │  Mission │   Mission Control  · Sentinel Ops  · Agent     │
  │  Sentinel│   Flow · Voice Intelligence · Healthcare Ops   │
  │  Agents  │   · Live Agent · Voice Agent · Cost & SLO      │
  │  ...     │                                                │
  └──────────┴────────────────────────────────────────────────┘

Tab list (final): Mission Control, Sentinel Ops, Agent Flow, Voice
Intelligence, Healthcare Ops, Live Agent, Voice Agent, Cost & SLO.
The three v1 tabs (Quality Metrics, Escalations, Trace Explorer)
are retired in Phase G — their content fully lives in the v2 views.

Reads ``report_<customer>.json`` + ``trace_<customer>.json`` from
``CLARION_DATA_DIR`` (default ``data/``). The Live Agent tab talks to
FastAPI at ``CLARION_API_URL`` (default ``http://localhost:8000``).
"""

from __future__ import annotations

import html as _html
import logging
import os
from importlib import metadata

import gradio as gr

from gradio_app import (
    components,
    data,
    data_sources,
    tab_cost_ocr,
    tab_live_agent,
    tab_voice_agent,
)
from gradio_app.theme import CLARION_THEME, CSS
from gradio_app.views import (
    agent_flow,
    configuration,
    cost_slo,
    healthcare_ops,
    mission_control,
    patient_360,
    sentinel_ops,
    system_health,
    voice_intel,
)

log = logging.getLogger(__name__)

TITLE = "Clarion — Configurable Multi-Agent Healthcare Operations Platform"


def _resolve_version() -> str:
    """Return the installed package version, falling back to ``dev``.

    The brand strip shows this — looking it up via ``importlib.metadata``
    means the displayed version stays in sync with ``pyproject.toml``
    without a duplicate constant in code.
    """
    try:
        return metadata.version("clarion")
    except metadata.PackageNotFoundError:
        return "dev"


def build_app() -> gr.Blocks:
    """Construct the gr.Blocks app.

    Phase G shell: top brand strip is always visible; customer
    switcher sits below it; main canvas hosts the eight live tabs.
    The left-rail nav appearance comes from CSS that rotates the
    Gradio tab list to vertical (see style.css §"Phase G — shell").
    """
    customers = data.available_customers()
    default_customer = customers[0]
    version = _resolve_version()

    with gr.Blocks(title=TITLE, theme=CLARION_THEME, css=CSS) as demo:
        # ---------- Top strip ----------
        # The brand strip never rebuilds — it's static chrome.
        gr.HTML(
            components.brand_strip(
                version=version,
                env="live",
                env_status="healthy",
            )
        )

        # ----- Tenant-aware identity layer ------------------------
        # Three skinny components sit between the brand strip and
        # the customer dropdown. All three rebuild on every customer
        # switch so the page identity shifts with the active tenant.
        #
        #   1. accent_css     - hidden <style> override that swaps
        #                       --c-accent / --c-accent-dim site-wide.
        #   2. greeting_html  - time-of-day greeting + per-tenant
        #                       attention count ("3 items need eyes").
        #   3. standout_html  - the single most important fact for
        #                       this tenant right now (lands recruiters
        #                       on a number, not a navigation tree).
        accent_css = gr.HTML(_render_accent_css(default_customer))
        greeting_html = gr.HTML(_render_greeting(default_customer))
        standout_html = gr.HTML(_render_standout(default_customer))

        # Customer switcher sits between the brand strip and the
        # main canvas so it reads as a tenant context selector,
        # not a tab control.
        customer_dd = gr.Dropdown(
            choices=customers,
            value=default_customer,
            label="Customer",
        )

        # ---------- Main canvas ----------
        #
        # Tab order is load-bearing — the CSS in style.css injects
        # section headers via :nth-of-type() before specific tabs.
        # Sections (in order):
        #   1. Mission Control   (tabs 1-6)
        #   2. Healthcare Ops    (tab 7)
        #   3. Interactive       (tabs 8-9)
        #   4. Platform          (tab 10)
        # Reorder = update the section-header CSS too.
        with gr.Tabs():
            # ---- MISSION CONTROL section ----
            with gr.Tab("Mission Control"):
                mc_html = gr.HTML(_render_mission_control())
            with gr.Tab("Sentinel Ops"):
                so_html = gr.HTML(_render_sentinel_ops(default_customer))
            with gr.Tab("Agent Flow"):
                af_html = gr.HTML(_render_agent_flow(default_customer))
            with gr.Tab("Voice Intelligence"):
                vi_html = gr.HTML(
                    _render_voice_intel(default_customer),
                    elem_id="clarion-vi-canvas",
                )
            with gr.Tab("Patient 360"):
                p360_html = gr.HTML(_render_patient_360(default_customer))
            with gr.Tab("Cost & SLO"):
                cs_html = gr.HTML(_render_cost_slo())
                # OCR invoice extractor sits below the cost rollup.
                # Self-contained: the upload + button + result panel
                # all live in tab_cost_ocr.build() so app.py stays
                # focused on the shell wiring. The returned handles
                # are not held - the click handler is wired inside
                # build() so the components manage themselves.
                tab_cost_ocr.build()
            # ---- HEALTHCARE OPS section ----
            with gr.Tab("Healthcare Ops"):
                ho_html = gr.HTML(_render_healthcare_ops(default_customer))
            # ---- INTERACTIVE section ----
            with gr.Tab("Live Agent"):
                live = tab_live_agent.build()
            with gr.Tab("Voice Agent"):
                voice = tab_voice_agent.build()
            # ---- PLATFORM section ----
            with gr.Tab("System Health"):
                gr.HTML(_render_system_health())
            with gr.Tab("Configuration"):
                cfg_html = gr.HTML(_render_configuration(default_customer))

        # ---------- Footer strip ----------
        gr.HTML(_render_footer(version=version))

        def _compute_views(  # type: ignore[no-untyped-def]
            customer_id: str,
            live_state: tab_live_agent.LiveAgentState,
            voice_state: tab_voice_agent.VoiceAgentState,
        ):
            """Build the full tuple of view outputs for a customer.

            Mission Control + Cost & SLO are cross-tenant — they
            still rebuild on every switch so a viewer toggling
            ophthalmology -> orthopedics sees the executive view
            stay current with whatever was just loaded.

            Returns 13 values in the order matching ``outputs``:
              identity layer:  accent_css, greeting, standout
              tab views:       mc, so, af, vi, ho, p360
              session states:  live_state, voice_state
              tail views:      cs, cfg
            """
            new_voice_state = tab_voice_agent.VoiceAgentState(
                customer_id=customer_id,
                # Switching customer mid-session resets the conversation
                # so the next voice turn starts a fresh transcript.
                session_id="",
            )
            new_live_state = tab_live_agent.set_customer(live_state, customer_id)
            accent = _render_accent_css(customer_id)
            greeting = _render_greeting(customer_id)
            standout = _render_standout(customer_id)
            mc = _render_mission_control()
            so = _render_sentinel_ops(customer_id)
            af = _render_agent_flow(customer_id)
            vi = _render_voice_intel(customer_id)
            ho = _render_healthcare_ops(customer_id)
            p360 = _render_patient_360(customer_id)
            cs = _render_cost_slo()
            cfg = _render_configuration(customer_id)
            return (
                accent, greeting, standout,
                mc, so, af, vi, ho, p360,
                new_live_state, new_voice_state, cs, cfg,
            )

        def refresh_on_switch(  # type: ignore[no-untyped-def]
            customer_id: str,
            live_state: tab_live_agent.LiveAgentState,
            voice_state: tab_voice_agent.VoiceAgentState,
        ):
            """Generator for the customer-switch event.

            First yield paints the Voice Intelligence skeleton (other
            panels keep their prior HTML via ``gr.update()``) so the
            operator sees shimmer feedback while we compute the real
            per-tenant roll-ups; second yield writes the real data
            on top.
            """
            new_live_state = tab_live_agent.set_customer(live_state, customer_id)
            new_voice_state = tab_voice_agent.VoiceAgentState(
                customer_id=customer_id,
                session_id="",
            )
            # First yield: skeleton only on Voice Intelligence; the
            # identity-layer trio (accent / greeting / standout)
            # rebuilds immediately so the page shifts persona before
            # the real per-tenant views finish computing.
            yield (
                _render_accent_css(customer_id),    # accent_css
                _render_greeting(customer_id),      # greeting_html
                _render_standout(customer_id),      # standout_html
                gr.update(),                        # mc_html
                gr.update(),                        # so_html
                gr.update(),                        # af_html
                _skeleton_view("Voice Intelligence"),  # vi_html
                gr.update(),                        # ho_html
                gr.update(),                        # p360_html
                new_live_state,
                new_voice_state,
                gr.update(),                        # cs_html
                gr.update(),                        # cfg_html
            )
            yield _compute_views(customer_id, live_state, voice_state)

        def initial_load():  # type: ignore[no-untyped-def]
            """Non-generator callback wired to ``demo.load``.

            ``demo.load`` in Gradio 4.44 doesn't reliably pass
            ``inputs=`` to the function on the very first render, so
            we read the default customer + fresh state values from
            the enclosing closure instead. Two payoffs:

            - the page-load handler never receives ``None`` for
              customer_id and crashes inside ``_humanize``;
            - we don't need ``inputs=`` on demo.load at all, which
              sidesteps the "Too many arguments" Gradio warning.
            """
            seed_live_state = tab_live_agent.LiveAgentState(
                customer_id=default_customer,
            )
            seed_voice_state = tab_voice_agent.VoiceAgentState(
                customer_id=default_customer,
                session_id="",
            )
            return _compute_views(
                default_customer, seed_live_state, seed_voice_state,
            )

        outputs = [
            accent_css,
            greeting_html,
            standout_html,
            mc_html,
            so_html,
            af_html,
            vi_html,
            ho_html,
            p360_html,
            live.state,
            voice.state,
            cs_html,
            cfg_html,
        ]

        # Voice Intelligence skeleton-on-switch: refresh_on_switch
        # is a generator, so Gradio streams its yields back to the
        # client. First yield paints the shimmer into the VI canvas;
        # second yield overwrites it with the real per-tenant HTML.
        customer_dd.change(
            fn=refresh_on_switch,
            inputs=[customer_dd, live.state, voice.state],
            outputs=outputs,
        )

        # demo.load uses the non-generator path and reads its own
        # defaults from the closure - passing inputs= here caused
        # Gradio to send None for customer_id on the first render.
        demo.load(
            fn=initial_load,
            outputs=outputs,
        )

    return demo


# ---------- v2 view renderers ----------


def _render_mission_control() -> str:
    """Build the Mission Control HTML.

    Wraps the data-source roll-up so the app callback doesn't have
    to know anything about typed snapshots. Returns the empty-state
    HTML when no tenant has data on disk yet — surfacing a clear
    "run the harness" hint instead of an empty page.
    """
    snapshots = data_sources.all_tenant_snapshots()
    if not any(s.has_data for s in snapshots):
        return mission_control.empty_html()
    kpis = data_sources.build_global_kpis(snapshots)
    return mission_control.build_html(
        snapshots=snapshots,
        kpis=kpis,
        escalations=data_sources.recent_escalations(snapshots),
        emergencies=data_sources.recent_emergencies(snapshots),
    )


def _render_sentinel_ops(customer_id: str) -> str:
    """Build the Sentinel Operations Center HTML for one customer.

    The hero view is per-tenant — it binds to the active customer
    dropdown selection. The empty-state fall-back inside the view
    handles "no trace report on disk" without crashing.
    """
    ops = data_sources.build_sentinel_ops(customer_id)
    return sentinel_ops.build_html(ops)


def _render_agent_flow(customer_id: str) -> str:
    """Build the Agent Flow HTML for one customer.

    Defaults to the first scenario in the tenant's trace report.
    The empty-state fall-back inside the view handles "no trace
    on disk" cleanly.
    """
    flow = data_sources.build_agent_flow(customer_id)
    return agent_flow.build_html(flow)


def _render_voice_intel(customer_id: str) -> str:
    """Build the Voice Intelligence HTML for one customer.

    Per-tenant; emotion + frustration + escalation rate rolled from
    the chat trace, plus static voice-pipeline reference + sample
    transcript that ship regardless of data state.
    """
    vi = data_sources.build_voice_intelligence(customer_id)
    return voice_intel.build_html(vi)


def _render_healthcare_ops(customer_id: str) -> str:
    """Build the Healthcare Operations HTML for one customer.

    Per-tenant; aggregates over SQLite + M1 PMS writeback + M3
    predictions (or synthetic fallback). All read-only.
    """
    ops = data_sources.build_healthcare_ops(customer_id)
    return healthcare_ops.build_html(ops)


def _render_patient_360(customer_id: str) -> str:
    """Build the Patient 360 HTML for one customer.

    Per-tenant; renders a small synthetic patient roster + the
    first patient's profile/timeline/care-team/insurance. A future
    task can extend `data_sources.build_patient_360` to read from
    a real per-tenant patient store.
    """
    snap = data_sources.build_patient_360(customer_id)
    return patient_360.build_html(snap)


def _render_system_health() -> str:
    """Build the System Health HTML (cross-tenant Platform view).

    Not customer-bound; rebuilds once at startup. A future task
    can wire this to a refresh button so the operator can re-poll
    subsystem status without restarting the app.
    """
    return system_health.build_html(data_sources.build_system_health())


def _render_configuration(customer_id: str) -> str:
    """Build the Configuration HTML for one tenant.

    Reads the tenant's YAML config directly via
    ``clarion.config.loader.load_customer`` and renders it. The
    view module handles the empty / load-error state, so this
    helper is a thin pass-through.
    """
    return configuration.build_html(customer_id)


def _render_accent_css(customer_id: str) -> str:
    """Inline ``<style>`` block that swaps ``--c-accent`` site-wide
    to the active tenant's identity color.

    Lives as its own ``gr.HTML`` component so Gradio can replace it
    independently on customer switch - that way the accent ripple
    through KPI tile edges, badges, sparklines, focus rings happens
    in the same paint as the view content swap below it.
    """
    accent, accent_dim = data_sources.customer_accent(customer_id)
    return (
        "<style>"
        ":root, .gradio-container {"
        f"  --c-accent: {accent};"
        f"  --c-accent-dim: {accent_dim};"
        "}"
        "</style>"
    )


def _render_greeting(customer_id: str) -> str:
    """Time-of-day greeting + per-tenant attention count.

    Sits between the brand strip and the customer dropdown so the
    operator lands on a personalized line, not a static label. The
    attention count rolls escalations + emergencies for the active
    tenant from disk; falls back to a calm "all clear" line when
    nothing needs eyes.

    Server-time based - good enough for a single-operator demo /
    portfolio piece; a multi-region deployment would render this
    client-side from the browser's clock instead.
    """
    from datetime import datetime as _dt

    hour = _dt.now().hour
    if hour < 5 or hour >= 22:
        period = "Working late"
    elif hour < 12:
        period = "Good morning"
    elif hour < 17:
        period = "Good afternoon"
    else:
        period = "Good evening"

    display = data_sources._humanize(customer_id)
    try:
        snapshots = data_sources.all_tenant_snapshots()
        escalations = data_sources.recent_escalations(snapshots)
        emergencies = data_sources.recent_emergencies(snapshots)
        attention = sum(
            1 for x in (*escalations, *emergencies) if x.tenant == display
        )
    except Exception:
        attention = 0

    if attention > 0:
        attention_html = (
            f'<span class="clarion-greeting-attention">'
            f'<span class="clarion-greeting-count">{attention}</span>'
            f' item{"s" if attention != 1 else ""} need your eyes</span>'
        )
    else:
        attention_html = (
            '<span class="clarion-greeting-allclear">'
            'All clear on this tenant.</span>'
        )

    return (
        '<div class="clarion-greeting">'
        f'<span class="clarion-greeting-hello">{_html.escape(period)}, '
        'operator.</span> '
        f'<span class="clarion-greeting-context">'
        f'{_html.escape(display)}'
        '</span>'
        ' &middot; '
        f"{attention_html}"
        "</div>"
    )


def _render_standout(customer_id: str) -> str:
    """The single most important fact for this tenant right now.

    Picks one headline from the active tenant's data using a small
    priority ladder:

      1. emergencies pending  -> "X emergencies awaiting review"
      2. escalations pending  -> "X escalations need triage"
      3. containment dipped   -> "Containment X% - watch for trend"
      4. healthy + steady     -> "Containment X% / Pass rate Y%"
      5. no data on disk      -> calm hint to run the harness

    A second line carries a numeric back-up that adds confidence
    without crowding the headline. Rendered as a card with a left
    accent strip that picks up ``--c-accent`` (the per-tenant color
    swap from ``_render_accent_css``), so the card visibly belongs
    to the active tenant.
    """
    try:
        snap = data_sources.build_tenant_snapshot(customer_id)
    except Exception:
        snap = None

    if snap is None or not snap.has_data:
        return (
            '<div class="clarion-standout" data-tone="info">'
            '<div class="clarion-standout-eyebrow">Today\'s standout</div>'
            '<div class="clarion-standout-headline">'
            f'No artifacts on disk yet for {_html.escape(_humanize_safe(customer_id))}.'
            '</div>'
            '<div class="clarion-standout-sub">'
            "Run <code>python -m clarion.eval --customer "
            f"{_html.escape(customer_id)}</code> to populate the dashboard."
            '</div>'
            '</div>'
        )

    try:
        snapshots = data_sources.all_tenant_snapshots()
        escalations = data_sources.recent_escalations(snapshots)
        emergencies = data_sources.recent_emergencies(snapshots)
        tenant_emergencies = [
            e for e in emergencies if e.tenant == snap.display_name
        ]
        tenant_escalations = [
            e for e in escalations if e.tenant == snap.display_name
        ]
    except Exception:
        tenant_emergencies = []
        tenant_escalations = []

    containment_pct = f"{snap.containment_rate * 100:.1f}%"
    pass_pct = f"{snap.pass_rate * 100:.1f}%"

    if tenant_emergencies:
        n = len(tenant_emergencies)
        headline = f"{n} emergenc{'ies' if n != 1 else 'y'} awaiting review"
        sub = (
            f"Containment {containment_pct} &middot; "
            f"Pass rate {pass_pct}"
        )
        tone = "critical"
    elif tenant_escalations:
        n = len(tenant_escalations)
        headline = f"{n} escalation{'s' if n != 1 else ''} need triage"
        sub = (
            f"Pass rate {pass_pct} &middot; "
            f"Containment {containment_pct}"
        )
        tone = "warning"
    elif snap.containment_rate < 0.55:
        headline = f"Containment dipped to {containment_pct}"
        sub = f"Last run {_html.escape(snap.last_run_relative)} &middot; watch for trend"
        tone = "warning"
    else:
        headline = f"All systems steady &middot; containment {containment_pct}"
        sub = (
            f"Pass rate {pass_pct} &middot; "
            f"{snap.scenario_count} scenarios scored "
            f"{_html.escape(snap.last_run_relative)}"
        )
        tone = "healthy"

    return (
        f'<div class="clarion-standout" data-tone="{tone}">'
        '<div class="clarion-standout-eyebrow">Today\'s standout</div>'
        f'<div class="clarion-standout-headline">{headline}</div>'
        f'<div class="clarion-standout-sub">{sub}</div>'
        "</div>"
    )


def _humanize_safe(customer_id: str | None) -> str:
    """Display-name helper local to ``_render_standout``'s empty path."""
    return data_sources._humanize(customer_id)


def _skeleton_view(label: str) -> str:
    """Return a skeleton placeholder for one tab during customer
    switch. Cheap to render (pure HTML, no I/O) so the swap is
    near-instant; the `.clarion-skeleton` shimmer animation
    keeps the user oriented while refresh_all finishes.

    The shape roughly tracks a typical view: title + subtitle,
    KPI strip of 4 tiles, then a two-up panel row.
    """
    tile = (
        '<div class="clarion-skeleton clarion-skeleton-block" '
        'style="flex: 1 1 0; min-width: 120px; height: 96px;"></div>'
    )
    big_panel = (
        '<div class="clarion-skeleton clarion-skeleton-block" '
        'style="flex: 1 1 0; min-width: 0; height: 220px; '
        'border-radius: var(--r-lg);"></div>'
    )
    return (
        '<div class="clarion-stack" style="gap: 20px;">'
        # Title + subtitle pair.
        '<div class="clarion-stack" style="gap: 8px;">'
        '<div class="clarion-skeleton" '
        f'style="height: 22px; width: 280px;" aria-label="Loading {_html.escape(label, quote=True)}"></div>'
        '<div class="clarion-skeleton" '
        'style="height: 12px; width: 360px;"></div>'
        "</div>"
        # KPI strip (4 tiles).
        '<div class="clarion-row" style="gap: 12px; flex-wrap: wrap;">'
        + (tile * 4)
        + "</div>"
        # Two big panels side by side.
        '<div class="clarion-row" style="gap: 16px;">'
        + big_panel
        + big_panel
        + "</div>"
        "</div>"
    )


def _render_footer(*, version: str) -> str:
    """Build the persistent footer status strip.

    Left side: a rotating quote band cycling through three short,
    confident product affirmations. Right side: a System Health
    pulse + status. The version tag tucks in at the far right so
    the chrome stays compact.

    The quote rotation is pure CSS — three ``span`` rows stacked
    in absolute position, fading in turn via ``@keyframes`` (see
    style.css §"Engagement quote band"). No JS, no Gradio coupling.
    """
    pulse_svg = (
        '<svg width="120" height="14" viewBox="0 0 120 14" '
        'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        '<polyline points="0,7 18,7 24,2 32,12 40,5 48,9 56,7 '
        '80,7 86,3 94,11 102,6 110,8 120,7" '
        'fill="none" stroke="#22D3EE" stroke-width="1.5" '
        'stroke-linecap="round" stroke-linejoin="round" '
        'opacity="0.85"/>'
        "</svg>"
    )
    quotes = (
        "Augmenting clinicians. Never replacing them.",
        "Every decision backed by an audit trail.",
        "Front-line operations. Back-office trust.",
        "Real evidence. Real-time decisions.",
        "Where compliance meets velocity.",
    )
    quote_spans = "".join(
        f'<span class="clarion-quote clarion-quote-{i}">'
        f"&ldquo;{_html.escape(q, quote=True)}&rdquo;</span>"
        for i, q in enumerate(quotes)
    )
    return (
        '<div class="clarion-footer">'
        '<div class="clarion-footer-left">'
        '<div class="clarion-quote-band" aria-live="polite">'
        + quote_spans
        + "</div>"
        "</div>"
        '<div class="clarion-footer-right">'
        '<span class="clarion-footer-label">System Health</span>'
        + pulse_svg
        + '<span class="clarion-footer-status">Operational</span>'
        f'<span class="clarion-footer-version">v{_html.escape(version, quote=True)}</span>'
        "</div>"
        "</div>"
    )


def _render_cost_slo() -> str:
    """Build the Cost & SLO HTML.

    Same shape as Mission Control — typed rollup, empty-state when
    no traces are on disk.
    """
    snapshots = data_sources.all_tenant_snapshots()
    if not any(s.has_data for s in snapshots):
        return cost_slo.empty_html()
    return cost_slo.build_html(data_sources.build_cost_slo(snapshots))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    demo = build_app()
    host = os.environ.get("GRADIO_HOST", "0.0.0.0")
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    demo.launch(server_name=host, server_port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
