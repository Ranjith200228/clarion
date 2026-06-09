"""Build the ``TraceReport`` sidecar JSON that the Phase 14 UI consumes.

The Phase 13 spec calls for two output files per evaluation run:

* ``report_<customer>.json`` — the aggregated EvaluationReport (metrics)
* ``trace_<customer>.json``  — per-scenario TraceReport (rows for the
  Trace Explorer tab)

This module owns the second one. Like ``reporter.py``, it does no
metric calculation of its own; it walks the HarnessResult + traces
summaries that have already been computed and packs them into the
locked ``TraceReport`` wire shape.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clarion.evaluation.metrics import load_trace_summaries
from clarion.schemas import (
    HarnessReport,
    HarnessResult,
    TraceEntry,
    TraceReport,
)

log = logging.getLogger(__name__)


def build_trace_report(
    customer_id: str,
    report: HarnessReport,
    *,
    traces_path: Path | None = None,
) -> TraceReport:
    """Assemble the per-scenario ``TraceReport`` for one run.

    ``traces_path`` is the customer's ``traces.jsonl`` (Phase 7); when
    None or missing, the resulting entries carry duration_ms / cost_usd
    / tokens as None / 0 but every other field is still populated.
    """
    summaries = load_trace_summaries(traces_path) if traces_path is not None else {}

    entries: list[TraceEntry] = []
    for result in report.results:
        # A scenario can produce multiple traces (one per chat turn);
        # we aggregate across them. Phase 9 scenarios are single-turn,
        # so this is usually one trace per result — but the
        # aggregation works for the multi-turn case too.
        duration_ms = 0.0
        cost_usd = 0.0
        input_tokens = 0
        output_tokens = 0
        step_count = 0
        saw_any_summary = False
        for trace_id in result.trace_ids or []:
            summary = summaries.get(trace_id)
            if summary is None:
                continue
            saw_any_summary = True
            duration_ms += float(summary.get("duration_ms") or 0.0)
            cost_usd += float(summary.get("cost_usd") or 0.0)
            input_tokens += int(summary.get("input_tokens") or 0)
            output_tokens += int(summary.get("output_tokens") or 0)
            step_count += int(summary.get("step_count") or 0)

        entries.append(
            TraceEntry(
                scenario_id=result.scenario_id,
                customer_id=result.customer_id,
                trace_id=(result.trace_ids[0] if result.trace_ids else ""),
                difficulty=result.difficulty,
                intent=result.intent,
                agent_replies=list(result.agent_replies),
                tools_called=list(result.actual_tools),
                actual_outcome=result.actual_outcome,
                passed=result.passed,
                escalation_score=_escalation_score(result),
                escalation_reasons=_escalation_reasons(result),
                judge_hallucination=_judge_field(result, "hallucination"),
                judge_booking_correct=_judge_field(result, "booking_correct"),
                judge_violations=_judge_violations(result),
                duration_ms=duration_ms if saw_any_summary else None,
                cost_usd=cost_usd if saw_any_summary else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                step_count=step_count,
            )
        )

    return TraceReport(
        customer_id=customer_id,
        generated_at=datetime.now(UTC),
        entries=entries,
    )


def write_trace_report(report: TraceReport, out_path: Path) -> Path:
    """Persist the ``TraceReport`` as JSON. Returns the path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return out_path


# ---------- helpers ----------


def _escalation_score(result: HarnessResult) -> float | None:
    esc = result.escalation
    if not isinstance(esc, dict):
        return None
    val = esc.get("score")
    if val is None:
        return None
    try:
        # Clamp defensively — the Pydantic side already enforces [0, 1]
        # but a malformed JSON might slip through.
        return max(0.0, min(1.0, float(val)))
    except (TypeError, ValueError):
        return None


def _escalation_reasons(result: HarnessResult) -> list[str]:
    esc = result.escalation
    if not isinstance(esc, dict):
        return []
    reasons = esc.get("reasons") or []
    return [str(r) for r in reasons]


def _judge_field(result: HarnessResult, key: str) -> float | None:
    verdict = result.judge_verdict
    if not isinstance(verdict, dict):
        return None
    val = verdict.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _judge_violations(result: HarnessResult) -> list[str]:
    verdict = result.judge_verdict
    if not isinstance(verdict, dict):
        return []
    raw = verdict.get("policy_violations") or []
    out: list[str] = []
    for v in raw:
        if isinstance(v, dict):
            kind = v.get("kind")
            if kind:
                out.append(str(kind))
        else:
            # Defensive: if the verdict was sliced through model_dump
            # already, kind might be a plain string.
            out.append(str(v))
    return out


# Keep mypy happy on the Any cast that load_trace_summaries returns.
_: Any = None
