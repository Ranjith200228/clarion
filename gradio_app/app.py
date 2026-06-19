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
    cost_slo,
    healthcare_ops,
    mission_control,
    sentinel_ops,
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
        with gr.Tabs():
            # v2 default landing — cross-tenant snapshot.
            with gr.Tab("Mission Control"):
                mc_html = gr.HTML(_render_mission_control())
            # v2 hero #1 — the trust engine surfaced.
            with gr.Tab("Sentinel Ops"):
                so_html = gr.HTML(_render_sentinel_ops(default_customer))
            # v2 hero #2 — multi-agent reasoning visualised.
            with gr.Tab("Agent Flow"):
                af_html = gr.HTML(_render_agent_flow(default_customer))
            # v2 hero #3 — emotion + frustration + voice pipeline.
            with gr.Tab("Voice Intelligence"):
                vi_html = gr.HTML(_render_voice_intel(default_customer))
            # v2 domain intelligence — provider availability, no-show
            # risk, eligibility, PMS queue.
            with gr.Tab("Healthcare Ops"):
                ho_html = gr.HTML(_render_healthcare_ops(default_customer))
            # v1 interactive surfaces — the only callable agents.
            # Kept because they cannot be replaced by a read-only
            # view: the user types or speaks into them.
            with gr.Tab("Live Agent"):
                live = tab_live_agent.build()
            with gr.Tab("Voice Agent"):
                voice = tab_voice_agent.build()
            # Executive bottom strip — cross-tenant cost + SLO.
            with gr.Tab("Cost & SLO"):
                cs_html = gr.HTML(_render_cost_slo())

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
            cs = _render_cost_slo()
            new_live_state = tab_live_agent.set_customer(live_state, customer_id)
            return (mc, so, af, vi, ho, new_live_state, new_voice_state, cs)

        outputs = [
            mc_html,
            so_html,
            af_html,
            vi_html,
            ho_html,
            live.state,
            voice.state,
            cs_html,
        ]

        # Switcher change fans out to every customer-bound view.
        customer_dd.change(
            fn=refresh_all,
            inputs=[customer_dd, live.state, voice.state],
            outputs=outputs,
        )
        # Initial population on app load.
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
