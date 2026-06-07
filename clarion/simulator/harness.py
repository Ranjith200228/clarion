"""Run scenarios through the agent and score each against ground truth.

Two modes:

* **scripted** — wires a ``FakeLLM`` whose responses come from each
  scenario's ``llm_script``. Deterministic, zero cost. Used in CI.
* **live** — uses a real ``LLMClient`` (typically ``OpenAIClient``).
  Costs money, requires an API key, but exercises the real agent
  intelligence. Used in dev / staging.

Output is a ``HarnessReport`` (per-scenario ``HarnessResult`` + roll-up
counters). Phase 12's evaluation framework will compute richer metrics
from the same shape.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

from clarion.agents.agent import Agent
from clarion.agents.llm import FakeLLM, LLMClient, LLMResponse, LLMUsage, ToolCall
from clarion.config import CustomerConfig, Settings, load_customer
from clarion.observability import TraceWriter
from clarion.pipelines.structured import StructuredStore
from clarion.rag.builder import load_customer_retriever
from clarion.rag.retriever import Retriever
from clarion.schemas import (
    HarnessReport,
    HarnessResult,
    Outcome,
    Scenario,
)
from clarion.schemas.scenarios import LLMScriptStep
from clarion.sentinel import AuditLog
from clarion.tools.base import ToolContext

log = logging.getLogger(__name__)


def load_scenarios(path: Path) -> list[Scenario]:
    """Read scenarios from a ``data/personas/<customer>.json`` payload."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Scenario(**s) for s in raw["scenarios"]]


def run_scripted(
    scenarios: Iterable[Scenario],
    *,
    customer_config: CustomerConfig,
    structured: StructuredStore,
    retriever: Retriever | None,
    audit: AuditLog | None = None,
    traces: TraceWriter | None = None,
) -> HarnessReport:
    """Run scenarios with a FakeLLM driven by each scenario's llm_script."""
    return _run(
        scenarios,
        customer_config=customer_config,
        structured=structured,
        retriever=retriever,
        audit=audit,
        traces=traces,
        llm_factory=_make_scripted_llm_factory(),
    )


def run_live(
    scenarios: Iterable[Scenario],
    *,
    customer_config: CustomerConfig,
    structured: StructuredStore,
    retriever: Retriever | None,
    llm_client: LLMClient,
    audit: AuditLog | None = None,
    traces: TraceWriter | None = None,
) -> HarnessReport:
    """Run scenarios with a real LLM (e.g. ``OpenAIClient``)."""
    return _run(
        scenarios,
        customer_config=customer_config,
        structured=structured,
        retriever=retriever,
        audit=audit,
        traces=traces,
        llm_factory=lambda _scenario: llm_client,
    )


def run_for_customer(
    customer_id: str,
    *,
    settings: Settings,
    mode: str = "scripted",
    llm_client: LLMClient | None = None,
) -> HarnessReport:
    """Convenience: load everything from disk, run, return report.

    ``mode="scripted"`` ignores ``llm_client``. ``mode="live"`` requires it.
    """
    customer = load_customer(customer_id, settings=settings)
    structured = StructuredStore.for_customer(customer.customer_id, settings.data_dir)
    try:
        retriever: Retriever | None = load_customer_retriever(customer, data_dir=settings.data_dir)
    except FileNotFoundError:
        retriever = None
    personas_path = settings.data_dir / "personas" / f"{customer_id}.json"
    scenarios = load_scenarios(personas_path)

    if mode == "scripted":
        return run_scripted(
            scenarios,
            customer_config=customer,
            structured=structured,
            retriever=retriever,
        )
    if mode == "live":
        if llm_client is None:
            raise ValueError("mode='live' requires llm_client")
        return run_live(
            scenarios,
            customer_config=customer,
            structured=structured,
            retriever=retriever,
            llm_client=llm_client,
        )
    raise ValueError(f"unknown mode {mode!r}")


# ---------- internals ----------


LLMFactory = Callable[[Scenario], LLMClient]


def _make_scripted_llm_factory() -> LLMFactory:
    """Each scenario gets its own FakeLLM with its own scripted responses.

    Guardrail-short-circuit scenarios (emergency / clinical_advice) have
    empty scripts; the FakeLLM is still created but is never consulted.
    """

    def factory(scenario: Scenario) -> LLMClient:
        responses = [_script_step_to_response(step) for step in scenario.llm_script]
        return FakeLLM(responses=responses)

    return factory


def _script_step_to_response(step: LLMScriptStep) -> LLMResponse:
    tool_calls = tuple(
        ToolCall(
            id=str(tc.get("id") or ""),
            name=str(tc["name"]),
            arguments=dict(tc.get("arguments") or {}),
        )
        for tc in step.tool_calls
    )
    return LLMResponse(
        content=step.content,
        tool_calls=tool_calls,
        usage=LLMUsage(
            model=step.model,
            input_tokens=step.input_tokens,
            output_tokens=step.output_tokens,
        ),
    )


def _run(
    scenarios: Iterable[Scenario],
    *,
    customer_config: CustomerConfig,
    structured: StructuredStore,
    retriever: Retriever | None,
    audit: AuditLog | None,
    traces: TraceWriter | None,
    llm_factory: LLMFactory,
) -> HarnessReport:
    results: list[HarnessResult] = []
    for s in scenarios:
        result = _run_one(
            s,
            customer_config=customer_config,
            structured=structured,
            retriever=retriever,
            audit=audit,
            traces=traces,
            llm_factory=llm_factory,
        )
        results.append(result)
    return _build_report(customer_config.customer_id, results)


def _run_one(
    scenario: Scenario,
    *,
    customer_config: CustomerConfig,
    structured: StructuredStore,
    retriever: Retriever | None,
    audit: AuditLog | None,
    traces: TraceWriter | None,
    llm_factory: LLMFactory,
) -> HarnessResult:
    tasks_before = len(structured.list_open_tasks())

    agent = Agent(
        customer=customer_config,
        llm=llm_factory(scenario),
        ctx=ToolContext(customer=customer_config, structured=structured),
        retriever=retriever,
        audit=audit,
        traces=traces,
    )

    replies: list[str] = []
    trace_ids: list[str] = []
    actual_tools: list[str] = []

    for message in scenario.messages:
        reply = agent.chat(message)
        replies.append(reply)
        trace_ids.append(agent.last_trace_id)

    # Read the tool-call history out of the agent's transcript (role=tool
    # messages carry tool name) — this works for the scripted and live
    # paths uniformly.
    for msg in agent.transcript:
        if msg.role == "tool" and msg.name:
            actual_tools.append(msg.name)

    tasks_after = structured.list_open_tasks()
    new_tasks = tasks_after[tasks_before:]
    escalated = bool(new_tasks)

    actual_outcome = _classify_outcome(
        replies=replies,
        actual_tools=actual_tools,
    )

    failure_reasons: list[str] = []
    if actual_outcome != scenario.ground_truth.expected_outcome:
        failure_reasons.append(
            f"outcome={actual_outcome!r} expected={scenario.ground_truth.expected_outcome!r}"
        )
    expected_set = set(scenario.ground_truth.expected_tools)
    actual_set = set(actual_tools)
    if expected_set and not expected_set.issubset(actual_set):
        failure_reasons.append(f"missing tools: {sorted(expected_set - actual_set)}")

    return HarnessResult(
        scenario_id=scenario.scenario_id,
        customer_id=scenario.customer_id,
        difficulty=scenario.difficulty,
        intent=scenario.intent,
        actual_outcome=actual_outcome,
        actual_tools=actual_tools,
        escalated=escalated,
        agent_replies=replies,
        trace_ids=trace_ids,
        passed=not failure_reasons,
        failure_reasons=failure_reasons,
    )


def _classify_outcome(
    *,
    replies: list[str],
    actual_tools: list[str],
) -> Outcome:
    """Derive an Outcome label from the actual run.

    Order matters: an emergency phrase in the reply is the strongest
    signal (the canned 911 line), then clinical-advice refusal, then we
    fall back to the tool-call evidence (book / cancel / task).
    """
    # Emergency reply is the canned "call 911" line.
    if any("911" in r for r in replies):
        return "escalated_emergency"

    # Clinical-advice refusal is the canned "clinical question" line.
    if any("clinical" in r.lower() for r in replies):
        return "refused_clinical"

    if "book_appointment" in actual_tools:
        return "booked"
    if "cancel_appointment" in actual_tools:
        return "cancelled"
    if "create_pms_task" in actual_tools:
        return "task_created"

    # Default — agent answered (FAQ, clarification, eligibility info).
    return "info_provided"


def _build_report(customer_id: str, results: list[HarnessResult]) -> HarnessReport:
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    total = len(results)
    pass_rate = passed / total if total else 0.0

    by_difficulty: dict[str, dict[str, int]] = {}
    by_intent: dict[str, dict[str, int]] = {}
    diff_counter: Counter[str] = Counter()
    int_counter: Counter[str] = Counter()
    diff_pass: Counter[str] = Counter()
    int_pass: Counter[str] = Counter()
    for r in results:
        diff_counter[r.difficulty] += 1
        int_counter[r.intent] += 1
        if r.passed:
            diff_pass[r.difficulty] += 1
            int_pass[r.intent] += 1
    for d, t in diff_counter.items():
        by_difficulty[d] = {"total": t, "passed": diff_pass[d], "failed": t - diff_pass[d]}
    for i, t in int_counter.items():
        by_intent[i] = {"total": t, "passed": int_pass[i], "failed": t - int_pass[i]}

    return HarnessReport(
        customer_id=customer_id,
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=pass_rate,
        by_difficulty=by_difficulty,
        by_intent=by_intent,
        results=results,
    )


