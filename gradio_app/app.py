"""Phase 14 Gradio Blocks app — four tabs + customer switcher.

Entry point::

    python -m gradio_app

Reads ``report_<customer>.json`` + ``trace_<customer>.json`` from
``CLARION_DATA_DIR`` (default ``data/``) for the Quality, Escalations,
and Trace Explorer tabs. The Live Agent tab talks to FastAPI at
``CLARION_API_URL`` (default ``http://localhost:8000``).
"""

from __future__ import annotations

import logging
import os

import gradio as gr

from gradio_app import (
    data,
    data_sources,
    tab_escalations,
    tab_live_agent,
    tab_quality,
    tab_trace_explorer,
    tab_voice_agent,
)
from gradio_app.data import SchemaVersionMismatchError
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

TITLE = "Clarion — Configurable Multi-Agent Voice Automation Platform"


def build_app() -> gr.Blocks:
    """Construct the gr.Blocks app.

    Tab order (Phase B): Mission Control sits **first** so the app
    opens to the recruiter view; Cost & SLO sits **last** as the
    executive bottom strip. The five v1 tabs (Live Agent, Voice,
    Quality, Escalations, Trace Explorer) live between, unchanged.
    Phase G will retire the v1 tabs once their content is fully
    represented in the v2 views; until then they stay live for
    backwards compatibility.
    """
    customers = data.available_customers()
    default_customer = customers[0]

    with gr.Blocks(title=TITLE, theme=CLARION_THEME, css=CSS) as demo:
        gr.Markdown(f"# {TITLE}")

        customer_dd = gr.Dropdown(
            choices=customers,
            value=default_customer,
            label="Customer",
            info="Switches all four tabs to the selected customer's report + traces.",
        )

        with gr.Tabs():
            # v2: opens here. Cross-tenant snapshot, no customer
            # binding — Mission Control is intentionally global.
            with gr.Tab("Mission Control"):
                mc_html = gr.HTML(_render_mission_control())
            # v2 hero — primary feature per the Phase 0 plan. The
            # trust engine has zero UI in v1; this is its dedicated
            # surface. Per-tenant (binds to the customer dropdown).
            with gr.Tab("Sentinel Ops"):
                so_html = gr.HTML(_render_sentinel_ops(default_customer))
            # v2 hero #2 — makes the multi-agent reasoning visible.
            # Per-tenant; binds to the customer dropdown. Phase E or
            # G will add a scenario-id picker for turn switching.
            with gr.Tab("Agent Flow"):
                af_html = gr.HTML(_render_agent_flow(default_customer))
            # v2 hero #3 — emotion analytics + frustration trace +
            # escalation prediction + voice pipeline reference.
            # Per-tenant; aggregates over the chat trace as a proxy
            # for live voice data (Phase 0 doc: trace doesn't
            # currently carry voice-specific data).
            with gr.Tab("Voice Intelligence"):
                vi_html = gr.HTML(_render_voice_intel(default_customer))
            # v2 domain intelligence — provider availability,
            # no-show risk, PMS queue, eligibility coverage.
            # Closes the "doesn't feel healthcare-shaped" gap.
            with gr.Tab("Healthcare Ops"):
                ho_html = gr.HTML(_render_healthcare_ops(default_customer))
            with gr.Tab("Live Agent"):
                live = tab_live_agent.build()
            with gr.Tab("Voice Agent"):
                voice = tab_voice_agent.build()
            with gr.Tab("Quality Metrics"):
                quality = tab_quality.build()
            with gr.Tab("Escalations"):
                escalations = tab_escalations.build()
            with gr.Tab("Trace Explorer"):
                traces = tab_trace_explorer.build()
            with gr.Tab("Cost & SLO"):
                cs_html = gr.HTML(_render_cost_slo())

        def refresh_all(  # type: ignore[no-untyped-def]
            customer_id: str,
            live_state: tab_live_agent.LiveAgentState,
            voice_state: tab_voice_agent.VoiceAgentState,
        ):
            """Reload every tab.

            Mission Control + Cost & SLO are cross-tenant — they
            rebuild on every customer switch too so a viewer who
            toggles ophthalmology -> orthopedics sees the executive
            view stay current with whatever was just loaded.
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
            try:
                artifacts = data.load_artifacts(customer_id)
            except FileNotFoundError as e:
                msg_q = tab_quality.render_empty(customer_id, str(e))
                msg_e = tab_escalations.render_empty(customer_id, str(e))
                msg_t = tab_trace_explorer.render_empty(customer_id, str(e))
                new_state = tab_live_agent.set_customer(live_state, customer_id)
                return (
                    mc,
                    so,
                    af,
                    vi,
                    ho,
                    *msg_q,
                    *msg_e,
                    *msg_t,
                    new_state,
                    new_voice_state,
                    cs,
                )
            except SchemaVersionMismatchError as e:
                reason = f"schema version mismatch: {e}"
                msg_q = tab_quality.render_empty(customer_id, reason)
                msg_e = tab_escalations.render_empty(customer_id, reason)
                msg_t = tab_trace_explorer.render_empty(customer_id, reason)
                new_state = tab_live_agent.set_customer(live_state, customer_id)
                return (
                    mc,
                    so,
                    af,
                    vi,
                    ho,
                    *msg_q,
                    *msg_e,
                    *msg_t,
                    new_state,
                    new_voice_state,
                    cs,
                )

            q = tab_quality.render(artifacts.report)
            esc = tab_escalations.render(artifacts.report)
            t = tab_trace_explorer.render(artifacts.trace_report)
            new_state = tab_live_agent.set_customer(live_state, customer_id)
            return (mc, so, af, vi, ho, *q, *esc, *t, new_state, new_voice_state, cs)

        outputs = [
            mc_html,
            so_html,
            af_html,
            vi_html,
            ho_html,
            quality.headline_md,
            quality.headline_table,
            quality.outcome_table,
            escalations.summary_md,
            escalations.reasons_table,
            escalations.escalated_table,
            escalations.threshold_md,
            traces.summary_md,
            traces.table,
            live.state,
            voice.state,
            cs_html,
        ]

        # Switcher change fans out to every tab.
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
