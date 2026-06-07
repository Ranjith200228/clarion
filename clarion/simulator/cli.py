"""Simulator CLI — generate persona files and run the harness.

Usage::

    # Regenerate scenarios (writes data/personas/<customer>.json)
    poetry run python -m clarion.simulator.cli generate ophthalmology
    poetry run python -m clarion.simulator.cli generate orthopedics
    poetry run python -m clarion.simulator.cli generate all

    # Run the harness against shipped personas
    poetry run python -m clarion.simulator.cli run ophthalmology
    poetry run python -m clarion.simulator.cli run orthopedics --mode live
    poetry run python -m clarion.simulator.cli run all --report-out reports/

The output of ``run`` is a HarnessReport JSON file per customer at
``<report_out>/<customer>.report.json``, plus a one-line summary to
stdout (``ophthalmology: 95/100 passed (95.0%)``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from clarion.config import Settings, get_settings
from clarion.schemas import HarnessReport
from clarion.simulator.generator import generate
from clarion.simulator.harness import run_for_customer
from clarion.simulator.templates import TEMPLATES

log = logging.getLogger(__name__)


def cmd_generate(customer_id: str, *, settings: Settings) -> int:
    cust = TEMPLATES.get(customer_id)
    if cust is None:
        print(
            f"unknown customer {customer_id!r}; known: {sorted(TEMPLATES)}",
            file=sys.stderr,
        )
        return 2
    scenarios = generate(cust)
    out = settings.data_dir / "personas" / f"{customer_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "customer_id": customer_id,
        "count": len(scenarios),
        "seed": 42,
        "scenarios": [s.model_dump() for s in scenarios],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"generated {len(scenarios)} scenarios -> {out}")
    return 0


def cmd_run(
    customer_id: str,
    *,
    settings: Settings,
    mode: str,
    report_dir: Path | None,
) -> int:
    llm_client = None
    if mode == "live":
        from clarion.agents.openai_client import OpenAIClient

        llm_client = OpenAIClient()
    report = run_for_customer(
        customer_id,
        settings=settings,
        mode=mode,
        llm_client=llm_client,
    )
    _summarize(report)
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        out = report_dir / f"{customer_id}.report.json"
        out.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
        print(f"  -> wrote {out}")
    return 0 if report.failed == 0 else 1


def _summarize(report: HarnessReport) -> None:
    pct = report.pass_rate * 100
    print(
        f"{report.customer_id}: {report.passed}/{report.total} passed "
        f"({pct:.1f}%); failed={report.failed}"
    )
    for difficulty, counts in sorted(report.by_difficulty.items()):
        print(f"  difficulty={difficulty:14s} {counts['passed']:3d}/{counts['total']:3d} passed")
    for intent, counts in sorted(report.by_intent.items()):
        print(f"  intent={intent:18s} {counts['passed']:3d}/{counts['total']:3d} passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clarion-sim")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="regenerate the persona JSON file")
    gen.add_argument("customer", help="customer id or 'all'")

    run = sub.add_parser("run", help="run the harness")
    run.add_argument("customer", help="customer id or 'all'")
    run.add_argument(
        "--mode",
        choices=["scripted", "live"],
        default="scripted",
    )
    run.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="dir to write per-customer report JSON files into",
    )

    args = parser.parse_args(argv)
    settings = get_settings()
    customers = list(TEMPLATES) if args.customer == "all" else [args.customer]

    exit_code = 0
    for cid in customers:
        if args.cmd == "generate":
            code = cmd_generate(cid, settings=settings)
        else:
            code = cmd_run(
                cid,
                settings=settings,
                mode=args.mode,
                report_dir=args.report_out,
            )
        exit_code = exit_code or code
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
