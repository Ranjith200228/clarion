"""End-to-end orchestration for one customer's evaluation run.

The Phase 13 spec splits the evaluation framework into three modules:

* ``runner.py``   — orchestration (load personas, run harness, gather traces)
* ``metrics.py``  — pure metric computation
* ``reporter.py`` — report assembly + JSON writing

This file is the runner. It loads everything from disk for one customer
and hands the resulting ``HarnessReport`` + scenario list to the
reporter, which is responsible for assembling the wire-shape
``EvaluationReport``.

Keeping this thin and ignorant of metric internals is what enforces the
spec's "LOCK THE REPORT SCHEMA" rule — the runner doesn't compute
numbers, it just orchestrates the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from clarion.config import Settings, load_customer
from clarion.pipelines.structured import StructuredStore
from clarion.rag.builder import load_customer_retriever
from clarion.rag.retriever import Retriever
from clarion.schemas import EvaluationReport, HarnessReport, Scenario
from clarion.simulator.harness import load_scenarios, run_scripted

log = logging.getLogger(__name__)


def run_evaluation(
    customer_id: str,
    *,
    settings: Settings,
    mode: str = "scripted",
) -> EvaluationReport:
    """Load personas + structured store + retriever, run the harness,
    return the ``EvaluationReport``.

    Pure scripted mode for now; live mode is reachable by calling
    ``clarion.simulator.harness.run_live`` then
    ``clarion.evaluation.reporter.build_report`` directly.
    """
    if mode != "scripted":
        raise NotImplementedError(
            f"mode={mode!r} not yet wired into run_evaluation; "
            f"the CLI exposes only scripted at the moment. Live mode "
            f"is available via clarion.simulator.harness.run_live() + "
            f"clarion.evaluation.reporter.build_report() directly."
        )

    scenarios, harness_report, traces_path = _execute_pipeline(customer_id, settings=settings)

    # Lazy import to break the runner -> reporter -> metrics circular
    # potential. The reporter imports metric helpers but doesn't depend
    # on the runner; importing it here keeps the dep graph one-way.
    from clarion.evaluation.reporter import build_report

    return build_report(
        customer_id,
        scenarios,
        harness_report,
        traces_path=traces_path if traces_path.exists() else None,
    )


def _execute_pipeline(
    customer_id: str, *, settings: Settings
) -> tuple[list[Scenario], HarnessReport, Path]:
    """Run the harness for one customer; return scenarios + report + traces path.

    Exposed as a private helper so callers that want to build a custom
    report (e.g. with a live LLM) can reuse the load / run path.
    """
    customer = load_customer(customer_id, settings=settings)
    structured = StructuredStore.for_customer(customer.customer_id, settings.data_dir)
    try:
        retriever: Retriever | None = load_customer_retriever(customer, data_dir=settings.data_dir)
    except FileNotFoundError:
        log.warning(
            "no prebuilt RAG index for %r — evaluation will run without retrieval",
            customer_id,
        )
        retriever = None

    personas_path = settings.data_dir / "personas" / f"{customer_id}.json"
    scenarios = load_scenarios(personas_path)

    harness_report = run_scripted(
        scenarios,
        customer_config=customer,
        structured=structured,
        retriever=retriever,
    )

    traces_path = settings.data_dir / customer_id / "traces.jsonl"
    return scenarios, harness_report, traces_path
