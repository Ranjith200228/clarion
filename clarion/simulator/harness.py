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
    JudgeRequest,
    Outcome,
    Scenario,
)
from clarion.schemas.scenarios import LLMScriptStep
from clarion.sentinel import AuditLog, Judge
from clarion.sentinel.escalation import ConversationFacts, EscalationScorer
from clarion.tools.base import ToolContext

log = logging.getLogger(__name__)

# One process-wide scorer instance. Stateless — the call shapes the
# whole result. Cheap construction, no need to allocate per-scenario.
_escalation_scorer = EscalationScorer()


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
    judge: Judge | None = None,
) -> HarnessReport:
    """Run scenarios with a FakeLLM driven by each scenario's llm_script.

    When ``judge`` is provided, every scenario's result also carries a
    ``judge_verdict`` (booking correctness + hallucination + policy
    violations from the Sentinel LLM-as-judge).
    """
    return _run(
        scenarios,
        customer_config=customer_config,
        structured=structured,
        retriever=retriever,
        audit=audit,
        traces=traces,
        llm_factory=_make_scripted_llm_factory(),
        judge=judge,
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
    judge: Judge | None = None,
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
        judge=judge,
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
    judge: Judge | None = None,
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
            judge=judge,
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
    judge: Judge | None = None,
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

    verdict = None
    if judge is not None and replies:
        # Build a JudgeRequest from the completed turn. Tool calls come
        # from the agent's transcript (we already walked it for
        # actual_tools above) — reconstruct {name, arguments, ok} dicts.
        tool_call_dicts = _tool_calls_for_judge(agent.transcript)
        rag_context = _rag_context_for_judge(retriever, scenario.messages[-1])
        verdict = judge.judge(
            JudgeRequest(
                customer_id=scenario.customer_id,
                user_message=scenario.messages[-1],
                agent_reply=replies[-1],
                tool_calls=tool_call_dicts,
                rag_context=rag_context,
                expected_appointment_type=scenario.ground_truth.expected_appointment_type,
            )
        )

    # Phase 11: always run the escalation scorer. It works with or
    # without a judge — the judge just sharpens the low_confidence +
    # rule_conflict signals.
    expected_outcome_is_task = scenario.ground_truth.expected_outcome == "task_created"
    escalation_score = _escalation_scorer.score(
        ConversationFacts(
            user_messages=list(scenario.messages),
            agent_replies=replies,
            tools_called=actual_tools,
            judge=verdict,
            expected_outcome_is_task=expected_outcome_is_task,
        )
    )

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
        judge_verdict=verdict.model_dump() if verdict is not None else None,
        escalation=escalation_score.model_dump(),
    )


def _tool_calls_for_judge(transcript: list) -> list[dict[str, object]]:  # type: ignore[type-arg]
    """Reconstruct a {name, arguments, ok} per tool call from the agent
    transcript. The role='tool' message carries the JSON result; the
    preceding assistant message carries the tool_calls list with the
    arguments. We zip them in order."""
    import json as _json

    pending_calls: list[tuple[str, dict[str, object]]] = []
    pending_results: list[dict[str, object]] = []
    flat: list[dict[str, object]] = []
    for msg in transcript:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                pending_calls.append((tc.name, dict(tc.arguments)))
        elif msg.role == "tool":
            try:
                result = _json.loads(msg.content or "{}")
            except _json.JSONDecodeError:
                result = {}
            pending_results.append(result if isinstance(result, dict) else {})
    for (name, args), result in zip(pending_calls, pending_results, strict=False):
        flat.append(
            {
                "name": name,
                "arguments": args,
                "ok": bool(result.get("ok")),
                "error": result.get("error"),
            }
        )
    return flat


def _rag_context_for_judge(retriever: Retriever | None, query: str) -> list[str]:
    """Top-k chunk texts the agent would have seen for ``query``.
    Returns [] when no retriever — the judge prompt handles the empty case."""
    if retriever is None:
        return []
    hits = retriever.retrieve(query, k=4)
    return [h.chunk.text for h in hits]


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
