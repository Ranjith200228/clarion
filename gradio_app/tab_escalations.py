"""Escalations tab.

Renders the Phase 13 EvaluationReport fields:
* Reason Frequency      — report.escalation_reason_frequency
* Escalated Calls       — report.escalated_scenario_ids
* Threshold Analysis    — report.metrics.escalation_precision / recall / f1
                          (the at-current-threshold snapshot)

Data source: ``report.json`` only. No re-computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr
from clarion.schemas import EvaluationReport


@dataclass
class EscalationsTab:
    summary_md: gr.Markdown
    reasons_table: gr.Dataframe
    escalated_table: gr.Dataframe
    threshold_md: gr.Markdown


def build() -> EscalationsTab:
    gr.Markdown("## Escalations\n\nLoaded from `report_<customer>.json`.")
    summary_md = gr.Markdown("_Select a customer to load escalations._")
    with gr.Row():
        reasons_table = gr.Dataframe(
            headers=["Reason", "Count"],
            label="Reason frequency",
            interactive=False,
            wrap=True,
        )
        escalated_table = gr.Dataframe(
            headers=["Scenario id"],
            label="Escalated calls",
            interactive=False,
            wrap=True,
        )
    threshold_md = gr.Markdown(
        "_Threshold analysis loaded after selecting a customer._",
        label=None,
    )
    return EscalationsTab(
        summary_md=summary_md,
        reasons_table=reasons_table,
        escalated_table=escalated_table,
        threshold_md=threshold_md,
    )


def render(
    report: EvaluationReport,
) -> tuple[str, list[list[str]], list[list[str]], str]:
    """Return the four populated component values for the tab."""
    m = report.metrics
    summary = (
        f"### `{report.customer_id}` — escalations at current threshold\n"
        f"- **{len(report.escalated_scenario_ids)}** of "
        f"**{report.scenario_count}** scenarios flagged for escalation"
    )

    reasons_rows = [
        [reason, str(count)]
        for reason, count in sorted(
            report.escalation_reason_frequency.items(), key=lambda kv: (-kv[1], kv[0])
        )
    ] or [["(no reasons fired)", "0"]]

    escalated_rows = [[sid] for sid in report.escalated_scenario_ids] or [["(none)"]]

    threshold_md = (
        "### Threshold analysis (current run)\n"
        f"- **escalation_precision:** {m.escalation_precision:.3f}\n"
        f"- **escalation_recall:** {m.escalation_recall:.3f}\n"
        f"- **escalation_f1:** {m.escalation_f1:.3f}\n"
        f"- **escalation_accuracy:** {m.escalation_accuracy:.3f}\n\n"
        "Threshold sweep across alternative cutoffs lands in a follow-up "
        "release once the reporter ships per-threshold P/R counts."
    )

    return summary, reasons_rows, escalated_rows, threshold_md


def render_empty(
    customer_id: str, reason: str
) -> tuple[str, list[list[str]], list[list[str]], str]:
    msg = (
        f"### `{customer_id}` — no escalations available\n"
        f"_{reason}_\n\n"
        f"Generate one with: `python -m clarion.eval --customer {customer_id}`"
    )
    return msg, [], [], ""
