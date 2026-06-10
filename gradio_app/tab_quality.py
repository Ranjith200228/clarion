"""Quality Metrics tab.

Renders the Phase 13 EvaluationReport fields the spec calls for:

* Containment Rate
* Booking Accuracy
* Hallucination Rate
* Escalation Recall
* Average Turns
* Cost Per Call
* Outcome Distribution

Data source: ``report.json`` only. No metric computation; no fallbacks.
If a field is missing or null we render the empty state — the harness
must be re-run.
"""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr
from clarion.schemas import EvaluationReport


@dataclass
class QualityTab:
    """Handles to the dynamic components so the customer switcher can
    refresh them. Returned by ``build``."""

    headline_md: gr.Markdown
    headline_table: gr.Dataframe
    outcome_table: gr.Dataframe


def build() -> QualityTab:
    """Build the Quality tab. Components are populated empty; the
    switcher in app.py wires the refresh callback."""
    gr.Markdown("## Quality Metrics\n\nLoaded from `report_<customer>.json`.")
    headline_md = gr.Markdown("_Select a customer to load metrics._")
    with gr.Row():
        headline_table = gr.Dataframe(
            headers=["Metric", "Value"],
            label="Headline numbers",
            interactive=False,
            wrap=True,
        )
        outcome_table = gr.Dataframe(
            headers=["Outcome", "Count"],
            label="Outcome distribution",
            interactive=False,
            wrap=True,
        )
    return QualityTab(
        headline_md=headline_md,
        headline_table=headline_table,
        outcome_table=outcome_table,
    )


def render(report: EvaluationReport) -> tuple[str, list[list[str]], list[list[str]]]:
    """Return the values for the three dynamic components.

    Pure rendering: every number is a field on ``report``. No averaging,
    no rate computation, no normalization happens here.
    """
    m = report.metrics
    summary = (
        f"### `{report.customer_id}` — schema v{report.schema_version}\n"
        f"- **{report.scenario_count}** scenarios · "
        f"**{m.pass_rate * 100:.1f}%** pass rate · "
        f"generated `{report.generated_at.isoformat(timespec='seconds')}`"
    )

    rows: list[list[str]] = [
        ["Containment Rate", _pct(m.containment_rate)],
        [
            "Booking Accuracy",
            f"{_pct(m.booking_accuracy)} ({m.booking_correct}/{m.booking_total})",
        ],
        ["Hallucination Rate", _pct_or_dash(m.hallucination_rate)],
        ["Escalation Recall", _pct(m.escalation_recall)],
        ["Escalation Precision", _pct(m.escalation_precision)],
        ["Safety Catch Rate", _pct(m.safety_catch_rate)],
        ["Average Turns / Scenario", f"{m.avg_turns_to_resolution:.2f}"],
        ["Cost Per Call", f"${m.cost_per_request_usd:.6f}"],
        ["Tokens Per Call", f"{m.tokens_per_call:.1f}"],
    ]
    if m.latency_ms is not None:
        rows.append(["Latency p50 (ms)", f"{m.latency_ms.p50:.1f}"])
        rows.append(["Latency p95 (ms)", f"{m.latency_ms.p95:.1f}"])

    outcome_rows = [
        [outcome, str(count)] for outcome, count in sorted(report.outcome_distribution.items())
    ] or [["(none)", "0"]]

    return summary, rows, outcome_rows


def render_empty(customer_id: str, reason: str) -> tuple[str, list[list[str]], list[list[str]]]:
    msg = (
        f"### `{customer_id}` — no report loaded\n"
        f"_{reason}_\n\n"
        f"Generate one with: `python -m clarion.eval --customer {customer_id}`"
    )
    return msg, [], []


# ---------- helpers ----------


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _pct_or_dash(value: float | None) -> str:
    return _pct(value) if value is not None else "—"
