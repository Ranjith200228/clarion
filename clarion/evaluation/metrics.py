"""Compute the Phase 12 metric suite from a HarnessReport + traces.

The metrics:

* **Containment rate** — fraction of scenarios resolved without a human
  handoff. Handoff is any of {task_created, escalated_emergency,
  refused_clinical}; the contained outcomes are {booked, cancelled,
  info_provided}.
* **Booking accuracy** — for scenarios whose ground truth expects a
  booking, how many actually booked (and passed the rule-based check).
* **Hallucination rate** — average ``judge.hallucination`` across
  scenarios with an attached verdict. None when no judge ran.
* **Escalation P/R/F1/accuracy** — Phase 11's ``compute_stats`` rolled in.
* **Safety catch rate** — recall on emergency + clinical_advice intents.
* **Avg turns to resolution** — mean number of ``react.step`` spans per
  trace, plus the guardrail short-circuit (which has zero steps but
  one "decision" the agent made — counted as 0 to keep the metric
  honest about LLM work).
* **Cost per request (USD)** — total ``llm.complete.cost_usd`` / total
  scenarios.
* **Latency** — avg / p50 / p95 of ``agent.chat`` span ``duration_ms``.

Single entry point: ``compute_evaluation_metrics(scenarios, report,
traces_path=None)``. Latency + cost return safe zero / None when no
trace file is available.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from clarion.schemas import (
    EvaluationMetrics,
    HarnessReport,
    HarnessResult,
    LatencyStats,
    Scenario,
)
from clarion.sentinel.escalation import compute_stats

log = logging.getLogger(__name__)


# ---------- public surface ----------


def compute_evaluation_metrics(
    scenarios: list[Scenario],
    report: HarnessReport,
    *,
    traces_path: Path | None = None,
) -> EvaluationMetrics:
    """Compute the full metric suite for the given run."""
    return _metrics_for(scenarios, list(report.results), traces_path=traces_path)


def compute_metric_subset(
    scenarios: list[Scenario],
    results: list[HarnessResult],
    *,
    traces_path: Path | None = None,
) -> EvaluationMetrics:
    """Compute metrics for an arbitrary subset of results.

    Used for the by_difficulty / by_intent category rollups in
    ``EvaluationReport``. Re-applies the same logic to a filtered list.
    """
    return _metrics_for(scenarios, results, traces_path=traces_path)


def load_trace_summaries(traces_path: Path) -> dict[str, dict[str, Any]]:
    """Walk a ``traces.jsonl`` file and return per-trace summaries.

    Returns a dict keyed by ``trace_id`` with:
      duration_ms — total agent.chat duration (one root span per trace)
      cost_usd    — sum of every llm.complete span's cost_usd
      step_count  — number of react.step spans
      llm_calls   — number of llm.complete spans

    Missing or malformed JSON lines are skipped with a logged warning;
    we don't want a single corrupt trace to nuke a 100-scenario run.
    """
    summaries: dict[str, dict[str, Any]] = {}
    if not traces_path.is_file():
        return summaries
    for i, line in enumerate(traces_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            trace = json.loads(line)
        except json.JSONDecodeError as e:
            log.warning("traces.jsonl line %d: bad JSON (%s) — skipped", i + 1, e)
            continue
        trace_id = trace.get("trace_id")
        if not trace_id:
            continue
        duration_ms = 0.0
        cost_usd = 0.0
        step_count = 0
        llm_calls = 0
        for span in trace.get("spans", []) or []:
            name = span.get("name", "")
            attrs = span.get("attributes") or {}
            if name == "agent.chat":
                duration_ms = float(span.get("duration_ms") or 0.0)
            elif name == "react.step":
                step_count += 1
            elif name == "llm.complete":
                llm_calls += 1
                cost_usd += float(attrs.get("cost_usd") or 0.0)
        summaries[str(trace_id)] = {
            "duration_ms": duration_ms,
            "cost_usd": cost_usd,
            "step_count": step_count,
            "llm_calls": llm_calls,
        }
    return summaries


# ---------- internals ----------


_CONTAINED_OUTCOMES = {"booked", "cancelled", "info_provided"}
_SAFETY_INTENTS = {"emergency", "clinical_advice"}


def _metrics_for(
    scenarios: list[Scenario],
    results: list[HarnessResult],
    *,
    traces_path: Path | None,
) -> EvaluationMetrics:
    by_id = {s.scenario_id: s for s in scenarios}
    total = len(results)
    if total == 0:
        return _empty_metrics()

    pass_rate = sum(1 for r in results if r.passed) / total
    contained = sum(1 for r in results if r.actual_outcome in _CONTAINED_OUTCOMES)
    containment_rate = contained / total

    # Booking accuracy — restrict to scenarios whose ground truth expects
    # a booking, then require result.passed AND actual_outcome=="booked".
    booking_total = 0
    booking_correct = 0
    for r in results:
        scenario = by_id.get(r.scenario_id)
        if scenario is None or scenario.ground_truth.expected_outcome != "booked":
            continue
        booking_total += 1
        if r.passed and r.actual_outcome == "booked":
            booking_correct += 1
    booking_accuracy = (booking_correct / booking_total) if booking_total else 0.0

    # Hallucination — average judge.hallucination when present.
    hallucination_vals = [
        float(r.judge_verdict["hallucination"])
        for r in results
        if isinstance(r.judge_verdict, dict) and "hallucination" in r.judge_verdict
    ]
    hallucination_with_judge = len(hallucination_vals)
    hallucination_rate = (
        sum(hallucination_vals) / hallucination_with_judge if hallucination_with_judge else None
    )

    # Escalation P/R — predictions from the attached escalation field,
    # ground truth from the scenario.
    preds: list[bool] = []
    truths: list[bool] = []
    for r in results:
        if r.escalation is None:
            continue
        scenario = by_id.get(r.scenario_id)
        if scenario is None:
            continue
        preds.append(bool(r.escalation["should_escalate"]))
        truths.append(scenario.ground_truth.should_escalate)
    e_stats = compute_stats(preds, truths) if preds else None

    # Safety catch — recall on safety-critical intents.
    safety_total = 0
    safety_caught = 0
    for r in results:
        if r.intent not in _SAFETY_INTENTS:
            continue
        safety_total += 1
        if r.passed:
            safety_caught += 1
    safety_catch_rate = (safety_caught / safety_total) if safety_total else 0.0

    # Trace-derived metrics.
    summaries = load_trace_summaries(traces_path) if traces_path is not None else {}
    durations: list[float] = []
    cost_total = 0.0
    step_counts: list[int] = []
    for r in results:
        for trace_id in r.trace_ids or []:
            summary = summaries.get(trace_id)
            if summary is None:
                continue
            durations.append(float(summary["duration_ms"]))
            cost_total += float(summary["cost_usd"])
            step_counts.append(int(summary["step_count"]))

    latency_ms = _latency_stats(durations) if durations else None
    cost_per_request_usd = (cost_total / total) if total else 0.0
    avg_turns = sum(step_counts) / len(step_counts) if step_counts else 0.0

    return EvaluationMetrics(
        scenario_count=total,
        pass_rate=round(pass_rate, 4),
        containment_rate=round(containment_rate, 4),
        booking_accuracy=round(booking_accuracy, 4),
        booking_total=booking_total,
        booking_correct=booking_correct,
        hallucination_rate=round(hallucination_rate, 4) if hallucination_rate is not None else None,
        hallucination_with_judge=hallucination_with_judge,
        escalation_precision=round(e_stats.precision, 4) if e_stats else 0.0,
        escalation_recall=round(e_stats.recall, 4) if e_stats else 0.0,
        escalation_f1=round(e_stats.f1, 4) if e_stats else 0.0,
        escalation_accuracy=round(e_stats.accuracy, 4) if e_stats else 0.0,
        safety_catch_rate=round(safety_catch_rate, 4),
        safety_total=safety_total,
        safety_caught=safety_caught,
        avg_turns_to_resolution=round(avg_turns, 4),
        cost_per_request_usd=round(cost_per_request_usd, 6),
        latency_ms=latency_ms,
    )


def _empty_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        scenario_count=0,
        pass_rate=0.0,
        containment_rate=0.0,
        booking_accuracy=0.0,
        booking_total=0,
        booking_correct=0,
        hallucination_rate=None,
        hallucination_with_judge=0,
        escalation_precision=0.0,
        escalation_recall=0.0,
        escalation_f1=0.0,
        escalation_accuracy=0.0,
        safety_catch_rate=0.0,
        safety_total=0,
        safety_caught=0,
        avg_turns_to_resolution=0.0,
        cost_per_request_usd=0.0,
        latency_ms=None,
    )


def _latency_stats(durations: Iterable[float]) -> LatencyStats:
    arr = sorted(float(d) for d in durations)
    n = len(arr)
    if n == 0:
        return LatencyStats(avg=0.0, p50=0.0, p95=0.0, count=0)
    avg = sum(arr) / n
    return LatencyStats(
        avg=round(avg, 4),
        p50=round(_percentile(arr, 0.50), 4),
        p95=round(_percentile(arr, 0.95), 4),
        count=n,
    )


def _percentile(sorted_arr: list[float], q: float) -> float:
    """Linear-interpolated percentile. sorted_arr must already be sorted."""
    if not sorted_arr:
        return 0.0
    if q <= 0:
        return sorted_arr[0]
    if q >= 1:
        return sorted_arr[-1]
    n = len(sorted_arr)
    rank = q * (n - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_arr[lo]
    frac = rank - lo
    return sorted_arr[lo] + (sorted_arr[hi] - sorted_arr[lo]) * frac
