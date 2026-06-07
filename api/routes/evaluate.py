"""``POST /evaluate`` — run a scripted sequence of messages, return metrics.

Placeholder for the richer evaluation harness in Phase 9 (synthetic
personas) and Phase 12 (full metric suite). What we ship now:

* take 1-20 scripted user messages
* run them through one conversation in order
* return per-turn replies + aggregate metrics (steps, tokens, cost,
  latency, tools used)

The trace and audit JSONL files for the customer are appended just like
any /chat call — so a dashboard built in Phase 13 can find the run by
scenario_id.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from clarion.config import CustomerConfigError, CustomerNotFoundError
from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    ErrorResponse,
    EvaluateMetrics,
    EvaluateRequest,
    EvaluateResponse,
    EvaluateTurn,
)

log = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    tags=["agent"],
    summary="Run a scripted scenario through the agent",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
def evaluate(request: Request, body: EvaluateRequest) -> EvaluateResponse:
    """Run ``messages`` sequentially against one conversation.

    Each entry advances the same agent transcript. The aggregate metrics
    sum across all turns; the per-turn breakdown lives in the JSONL
    trace files for the customer (visible via the Phase 13 dashboard).
    """
    sessions = request.app.state.sessions
    try:
        # Always allocate a fresh conversation_id so /evaluate runs are
        # isolated from any concurrent /chat session.
        conversation_id, agent = sessions.get_or_create_session(body.customer_id, None)
    except CustomerNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={"detail": str(e), "code": "customer_not_found"},
        ) from e
    except CustomerConfigError as e:
        raise HTTPException(
            status_code=400,
            detail={"detail": str(e), "code": "customer_config_invalid"},
        ) from e

    # Customer-level traces.jsonl path so we can read back this run's
    # spans for aggregation.
    customer = sessions.get_customer(body.customer_id)
    traces_path = customer.traces.path
    pre_existing_lines = (
        sum(1 for _ in traces_path.open("r", encoding="utf-8")) if traces_path.exists() else 0
    )

    transcript: list[EvaluateTurn] = []
    trace_ids: list[str] = []
    for user_msg in body.messages:
        reply = agent.chat(user_msg)
        transcript.append(EvaluateTurn(user_message=user_msg, agent_reply=reply))
        if agent.last_trace_id:
            trace_ids.append(agent.last_trace_id)

    metrics = _aggregate_metrics(
        traces_path=traces_path,
        from_line=pre_existing_lines,
        trace_ids=trace_ids,
        turn_count=len(body.messages),
    )
    return EvaluateResponse(
        customer_id=body.customer_id,
        scenario_id=body.scenario_id,
        conversation_id=conversation_id,
        trace_ids=trace_ids,
        transcript=transcript,
        metrics=metrics,
    )


# ---------- aggregation ----------


def _aggregate_metrics(
    *,
    traces_path: Path,
    from_line: int,
    trace_ids: list[str],
    turn_count: int,
) -> EvaluateMetrics:
    """Aggregate metrics from the JSONL trace file.

    We read only the new lines (``from_line:``) so concurrent /chat
    traffic on the same customer doesn't pollute the totals. Filter by
    trace_id as a belt-and-suspenders against ordering issues.
    """
    import json

    total_steps = 0
    in_tokens = 0
    out_tokens = 0
    cost = 0.0
    latency = 0.0
    tools_used: Counter[str] = Counter()

    if not traces_path.exists():
        return _empty_metrics(turn_count)

    wanted = set(trace_ids)
    lines = traces_path.read_text(encoding="utf-8").splitlines()
    for line in lines[from_line:]:
        try:
            trace = json.loads(line)
        except json.JSONDecodeError:
            continue
        if wanted and trace.get("trace_id") not in wanted:
            continue
        for span in trace.get("spans", []):
            name = span.get("name", "")
            attrs: dict[str, Any] = span.get("attributes") or {}
            if name == "agent.chat":
                latency += float(span.get("duration_ms") or 0)
            elif name == "react.step":
                total_steps += 1
            elif name == "llm.complete":
                in_tokens += int(attrs.get("input_tokens") or 0)
                out_tokens += int(attrs.get("output_tokens") or 0)
                cost += float(attrs.get("cost_usd") or 0)
            elif name.startswith("tool."):
                tool_name = name.removeprefix("tool.")
                tools_used[tool_name] += 1

    return EvaluateMetrics(
        turns=turn_count,
        total_steps=total_steps,
        total_input_tokens=in_tokens,
        total_output_tokens=out_tokens,
        total_cost_usd=cost,
        total_latency_ms=latency,
        tools_used=dict(tools_used),
    )


def _empty_metrics(turn_count: int) -> EvaluateMetrics:
    return EvaluateMetrics(
        turns=turn_count,
        total_steps=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_cost_usd=0.0,
        total_latency_ms=0.0,
        tools_used={},
    )
