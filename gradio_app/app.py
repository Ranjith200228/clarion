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
    tab_escalations,
    tab_live_agent,
    tab_quality,
    tab_trace_explorer,
    tab_voice_agent,
)
from gradio_app.data import SchemaVersionMismatchError

log = logging.getLogger(__name__)

TITLE = "Clarion — Configurable Multi-Agent Voice Automation Platform"


def build_app() -> gr.Blocks:
    """Construct the gr.Blocks app with four tabs + customer switcher."""
    customers = data.available_customers()
    default_customer = customers[0]

    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}")

        customer_dd = gr.Dropdown(
            choices=customers,
            value=default_customer,
            label="Customer",
            info="Switches all four tabs to the selected customer's report + traces.",
        )

        with gr.Tabs():
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

        def refresh_all(  # type: ignore[no-untyped-def]
            customer_id: str,
            live_state: tab_live_agent.LiveAgentState,
            voice_state: tab_voice_agent.VoiceAgentState,
        ):
            """Reload the three report-driven tabs + reset the live + voice states."""
            new_voice_state = tab_voice_agent.VoiceAgentState(
                customer_id=customer_id,
                # Switching customer mid-session resets the conversation
                # so the next voice turn starts a fresh transcript.
                session_id="",
            )
            try:
                artifacts = data.load_artifacts(customer_id)
            except FileNotFoundError as e:
                msg_q = tab_quality.render_empty(customer_id, str(e))
                msg_e = tab_escalations.render_empty(customer_id, str(e))
                msg_t = tab_trace_explorer.render_empty(customer_id, str(e))
                new_state = tab_live_agent.set_customer(live_state, customer_id)
                return (
                    *msg_q,
                    *msg_e,
                    *msg_t,
                    new_state,
                    new_voice_state,
                )
            except SchemaVersionMismatchError as e:
                reason = f"schema version mismatch: {e}"
                msg_q = tab_quality.render_empty(customer_id, reason)
                msg_e = tab_escalations.render_empty(customer_id, reason)
                msg_t = tab_trace_explorer.render_empty(customer_id, reason)
                new_state = tab_live_agent.set_customer(live_state, customer_id)
                return (
                    *msg_q,
                    *msg_e,
                    *msg_t,
                    new_state,
                    new_voice_state,
                )

            q = tab_quality.render(artifacts.report)
            esc = tab_escalations.render(artifacts.report)
            t = tab_trace_explorer.render(artifacts.trace_report)
            new_state = tab_live_agent.set_customer(live_state, customer_id)
            return (*q, *esc, *t, new_state, new_voice_state)

        outputs = [
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
