"""Phase 13 canonical evaluation CLI.

Usage::

    python -m clarion.eval --customer ophthalmology
    python -m clarion.eval --customer orthopedics --out reports/
    python -m clarion.eval --customer all
    python -m clarion.eval --customer ophthalmology --mode live

Writes two files per customer to ``--out`` (default: the configured
``CLARION_DATA_DIR / <customer_id>``):

    report_<customer_id>.json   (EvaluationReport, schema v1.0.0)
    trace_<customer_id>.json    (TraceReport,      schema v1.0.0)

The report file is the locked Phase 13 contract that the Phase 14
Gradio UI consumes. The trace file feeds the Trace Explorer tab.

``--mode scripted`` (default) drives every scenario with a per-scenario
``FakeLLM``: deterministic, free, CI-safe. ``--mode live`` drives every
scenario through ``OpenAIClient`` so the resulting report carries real
cost, latency, and token numbers. Live mode requires ``OPENAI_API_KEY``
in the environment and spends real API budget &mdash; the CLI prints a
warning + cost estimate before starting.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from clarion.config import get_settings
from clarion.evaluation.runner import run_and_write_artifacts
from clarion.schemas import REPORT_SCHEMA_VERSION
from clarion.simulator.templates import TEMPLATES

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clarion.eval",
        description=(
            "Run the Clarion evaluation harness for one customer and write "
            "report_<customer>.json + trace_<customer>.json (schema "
            f"v{REPORT_SCHEMA_VERSION})."
        ),
    )
    parser.add_argument(
        "--customer",
        required=True,
        help="Customer id (YAML stem) or 'all' for every shipped customer.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output directory for the two JSON files. " "Default: <CLARION_DATA_DIR>/<customer_id>/"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["scripted", "live"],
        default="scripted",
        help=(
            "Evaluation mode. scripted = FakeLLM (deterministic, free, "
            "CI-safe). live = OpenAIClient with real API calls and real "
            "numbers in the report (requires OPENAI_API_KEY; costs ~$0.50 "
            "per customer)."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the live-mode cost confirmation prompt.",
    )
    args = parser.parse_args(argv)
    settings = get_settings()

    customers = list(TEMPLATES) if args.customer == "all" else [args.customer]

    llm_client = None
    if args.mode == "live":
        llm_client = _build_live_client(customers, skip_prompt=args.yes)
        if llm_client is None:
            return 1

    exit_code = 0
    for customer_id in customers:
        out_dir = args.out if args.out is not None else settings.data_dir / customer_id
        try:
            artifacts = run_and_write_artifacts(
                customer_id,
                settings=settings,
                out_dir=out_dir,
                mode=args.mode,
                llm_client=llm_client,
            )
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            exit_code = 2
            continue
        _summarize(artifacts.report.customer_id, artifacts)

    return exit_code


def _build_live_client(customers: list[str], *, skip_prompt: bool):  # type: ignore[no-untyped-def]
    """Build an ``OpenAIClient`` after a cost confirmation gate.

    Returns ``None`` if the user declines or ``OPENAI_API_KEY`` is missing.
    Lazy-imports the OpenAI client so scripted runs don't pay the SDK
    import cost.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "error: --mode live requires OPENAI_API_KEY in the environment.",
            file=sys.stderr,
        )
        return None

    n = len(customers) * 100  # 100 scenarios per customer
    print(
        f"\n  Live-mode evaluation will run {n} scenarios across "
        f"{len(customers)} customer(s) through real OpenAI API calls.\n"
        f"  Estimated cost: ~${0.50 * len(customers):.2f} (gpt-4o-mini).\n"
        f"  This produces a report with real cost, latency, and token "
        f"numbers.\n"
    )
    if not skip_prompt:
        try:
            ack = input("Continue? [y/N] ").strip().lower()
        except EOFError:
            ack = ""
        if ack not in ("y", "yes"):
            print("aborted.", file=sys.stderr)
            return None

    from clarion.agents.openai_client import OpenAIClient

    return OpenAIClient()


def _summarize(customer_id: str, artifacts) -> None:  # type: ignore[no-untyped-def]
    """Concise one-screen summary so a CI log shows what was produced."""
    m = artifacts.report.metrics
    pct = m.pass_rate * 100
    print(f"{customer_id}: {artifacts.report.scenario_count} scenarios, " f"{pct:.1f}% pass rate")
    print(f"  containment_rate:     {m.containment_rate:.3f}")
    print(
        f"  booking_accuracy:     {m.booking_accuracy:.3f} "
        f"({m.booking_correct}/{m.booking_total})"
    )
    if m.hallucination_rate is not None:
        print(f"  hallucination_rate:   {m.hallucination_rate:.3f}")
    print(f"  escalation_precision: {m.escalation_precision:.3f}")
    print(f"  escalation_recall:    {m.escalation_recall:.3f}")
    print(f"  safety_catch_rate:    {m.safety_catch_rate:.3f}")
    print(f"  avg_turns:            {m.avg_turns_to_resolution:.2f}")
    print(f"  cost_per_request:     ${m.cost_per_request_usd:.6f}")
    print(f"  tokens_per_call:      {m.tokens_per_call:.1f}")
    if m.latency_ms is not None:
        lat = m.latency_ms
        print(f"  latency_ms:           avg={lat.avg:.1f} p50={lat.p50:.1f} p95={lat.p95:.1f}")
    print(f"  -> wrote {artifacts.report_path}")
    print(f"  -> wrote {artifacts.trace_path}")


if __name__ == "__main__":
    sys.exit(main())
