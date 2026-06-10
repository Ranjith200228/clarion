"""Trace Explorer tab.

Renders the Phase 13 TraceReport entries as a flat table. One row per
scenario, columns the spec calls out:
  agent / tool / tokens / latency / cost / escalation / judge_result

Data source: ``trace.json`` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr
from clarion.schemas import TraceEntry, TraceReport

_HEADERS = [
    "scenario_id",
    "difficulty",
    "intent",
    "tools_called",
    "outcome",
    "passed",
    "escalation",
    "judge_hallucination",
    "judge_violations",
    "tokens (in+out)",
    "duration_ms",
    "cost_usd",
]


@dataclass
class TraceExplorerTab:
    summary_md: gr.Markdown
    table: gr.Dataframe


def build() -> TraceExplorerTab:
    gr.Markdown("## Trace Explorer\n\nLoaded from `trace_<customer>.json`.")
    summary_md = gr.Markdown("_Select a customer to load traces._")
    table = gr.Dataframe(
        headers=_HEADERS,
        label="Traces",
        interactive=False,
        wrap=True,
        # Tall by default so the user can scan many rows.
        max_height=600,
    )
    return TraceExplorerTab(summary_md=summary_md, table=table)


def render(trace_report: TraceReport) -> tuple[str, list[list[str]]]:
    """Return summary + table rows for the trace tab."""
    summary = (
        f"### `{trace_report.customer_id}` — {len(trace_report.entries)} traces, "
        f"schema v{trace_report.schema_version}"
    )
    rows = [_row(entry) for entry in trace_report.entries] or [
        ["(no traces — re-run eval)", "", "", "", "", "", "", "", "", "", "", ""]
    ]
    return summary, rows


def render_empty(customer_id: str, reason: str) -> tuple[str, list[list[str]]]:
    return (
        (
            f"### `{customer_id}` — no traces available\n"
            f"_{reason}_\n\n"
            f"Generate one with: `python -m clarion.eval --customer {customer_id}`"
        ),
        [],
    )


# ---------- helpers ----------


def _row(e: TraceEntry) -> list[str]:
    return [
        e.scenario_id,
        e.difficulty,
        e.intent,
        ", ".join(e.tools_called) or "—",
        e.actual_outcome,
        "✅" if e.passed else "❌",
        _esc_label(e),
        _opt_float(e.judge_hallucination),
        ", ".join(e.judge_violations) or "—",
        f"{e.input_tokens + e.output_tokens}",
        _opt_float(e.duration_ms, fmt="{:.1f}"),
        _opt_float(e.cost_usd, fmt="${:.6f}"),
    ]


def _esc_label(e: TraceEntry) -> str:
    if e.escalation_score is None:
        return "—"
    score = f"{e.escalation_score:.2f}"
    if e.escalation_reasons:
        return f"{score} ({', '.join(e.escalation_reasons[:2])})"
    return score


def _opt_float(value: float | None, *, fmt: str = "{:.3f}") -> str:
    if value is None:
        return "—"
    return fmt.format(value)
