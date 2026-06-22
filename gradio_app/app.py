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

        # Customer switcher sits between the brand strip and the
        # main canvas so it reads as a tenant context selector,
        # not a tab control.
        customer_dd = gr.Dropdown(
            choices=customers,
            value=default_customer,
            label="Customer",
            info="Switches every tenant-bound view to the selected customer.",
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

        def refresh_all(  # type: ignore[no-untyped-def]
            customer_id: str,
            live_state: tab_live_agent.LiveAgentState,
            voice_state: tab_voice_agent.VoiceAgentState,
        ):
            """Reload every view that depends on the customer dropdown.

            Mission Control + Cost & SLO are cross-tenant — they
            still rebuild on every switch so a viewer toggling
            ophthalmology -> orthopedics sees the executive view
            stay current with whatever was just loaded.
            """
            new_voice_state = tab_voice_agent.VoiceAgentState(
                customer_id=customer_id,
                # Switching customer mid-session resets the conversation
                # so the next voice turn starts a fresh transcript.
                session_id="",
            )
            mc = _render_mission_control()
            so = _render_sentinel_ops(customer_id)
            af = _render_agent_flow(customer_id)
            vi = _render_voice_intel(customer_id)
            ho = _render_healthcare_ops(customer_id)
            p360 = _render_patient_360(customer_id)
            cs = _render_cost_slo()
            cfg = _render_configuration(customer_id)
            new_live_state = tab_live_agent.set_customer(live_state, customer_id)
            return (
                mc, so, af, vi, ho, p360,
                new_live_state, new_voice_state, cs, cfg,
            )

        outputs = [
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

        # Skeleton-on-switch is implemented in pure JS (via the
        # `js=` param on the change handler). The JS runs
        # synchronously in the browser BEFORE the Python callback
        # fires, paints the .clarion-skeleton blocks into every
        # gr.HTML view container, then refresh_all returns real
        # data and Gradio swaps it in. This sidesteps Gradio's
        # event-queue coalescing - the user always sees the
        # shimmer for the duration of the round-trip.
        customer_dd.change(
            fn=refresh_all,
            inputs=[customer_dd, live.state, voice.state],
            outputs=outputs,
        )

        # Scoped skeleton: a tiny <script> tag attaches a native
        # change listener to the customer dropdown and paints
        # shimmer blocks into #clarion-vi-canvas whenever the
        # value changes. Gradio's natural change handler then
        # runs refresh_all and the real HTML replaces the
        # skeleton when the round-trip returns.
        #
        # We use gr.HTML(elem_id="clarion-skel-bootstrap") so
        # Gradio doesn't escape the <script> tag (its sanitiser
        # leaves <script> alone inside gr.HTML by default).
        gr.HTML(
            """
<script>
(function attachVISkeleton() {
  if (window.__clarionVISkelArmed) return;  // idempotent
  const skel = `
    <div class="clarion-stack" style="gap: 20px;">
      <div class="clarion-stack" style="gap: 8px;">
        <div class="clarion-skeleton" style="height: 24px; width: 220px;"></div>
        <div class="clarion-skeleton" style="height: 12px; width: 340px;"></div>
      </div>
      <div class="clarion-row" style="gap: 12px; flex-wrap: wrap;">
        <div class="clarion-skeleton clarion-skeleton-block" style="flex:1 1 0;min-width:140px;height:104px;"></div>
        <div class="clarion-skeleton clarion-skeleton-block" style="flex:1 1 0;min-width:140px;height:104px;"></div>
        <div class="clarion-skeleton clarion-skeleton-block" style="flex:1 1 0;min-width:140px;height:104px;"></div>
        <div class="clarion-skeleton clarion-skeleton-block" style="flex:1 1 0;min-width:140px;height:104px;"></div>
      </div>
      <div class="clarion-row" style="gap: 16px;">
        <div class="clarion-skeleton clarion-skeleton-block" style="flex:1 1 0;min-width:0;height:260px;border-radius:var(--r-lg);"></div>
        <div class="clarion-skeleton clarion-skeleton-block" style="flex:1 1 0;min-width:0;height:260px;border-radius:var(--r-lg);"></div>
      </div>
    </div>`;

  function tryArm() {
    const dd = document.querySelector('.gradio-container select');
    const vi = document.getElementById('clarion-vi-canvas');
    if (!dd || !vi) {
      setTimeout(tryArm, 200);
      return;
    }
    dd.addEventListener('change', function() {
      const target = document.getElementById('clarion-vi-canvas');
      if (target) { target.innerHTML = skel; }
    });
    window.__clarionVISkelArmed = true;
  }
  tryArm();
})();
</script>
            """
        )
        # Initial population on app load - no skeleton flash here
        # because the first paint is already the empty html_outputs
        # default; refresh_all writes the real data on top.
        demo.load(
            fn=refresh_all,
            inputs=[customer_dd, live.state, voice.state],
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

    Right side: an SVG pulse-line that animates via CSS to
    suggest the system is alive. Left side: copyright + version.
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
    return (
        '<div class="clarion-footer">'
        '<div class="clarion-footer-left">'
        f'<span>&copy; 2026 Clarion Vision Platform</span>'
        f'<span class="clarion-footer-version">v{_html.escape(version, quote=True)}</span>'
        "</div>"
        '<div class="clarion-footer-right">'
        '<span class="clarion-footer-label">System Health</span>'
        + pulse_svg
        + '<span class="clarion-footer-status">Operational</span>'
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
