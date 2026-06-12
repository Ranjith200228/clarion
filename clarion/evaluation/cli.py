"""Evaluation CLI — run scripted harness, build report, write JSON.

Usage::

    # Per-customer
    poetry run python -m clarion.evaluation.cli run ophthalmology
    poetry run python -m clarion.evaluation.cli run orthopedics --out reports/

    # Both customers in one invocation
    poetry run python -m clarion.evaluation.cli run all

The default output path is ``<data_dir>/<customer>/evaluation_report.json``;
``--out DIR`` overrides to a custom directory and the filenames stay
``<customer>.evaluation_report.json`` so a single dir can hold both.

Exit code reflects pass rate vs ``--min-pass-rate`` (default 0.9). CI
can gate on it without parsing the JSON.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from clarion.config import get_settings
from clarion.evaluation.reporter import write_report
from clarion.evaluation.runner import run_evaluation
from clarion.simulator.templates import TEMPLATES

log = logging.getLogger(__name__)


def _print_summary(customer_id: str, report_path: Path, headline: dict[str, float]) -> None:
    print(f"\n[{customer_id}] wrote {report_path}")
    for key in (
        "containment_rate",
        "booking_accuracy",
        "hallucination_rate",
        "escalation_precision",
        "escalation_recall",
        "safety_catch_rate",
    ):
        val = headline.get(key, 0.0)
        print(f"  {key:24s} {val:.3f}")


def cmd_run(customer_id: str, *, out_dir: Path | None, min_pass_rate: float) -> int:
    settings = get_settings()
    report = run_evaluation(customer_id, settings=settings, mode="scripted")

    if out_dir is None:
        out_path = settings.data_dir / customer_id / "evaluation_report.json"
    else:
        out_path = out_dir / f"{customer_id}.evaluation_report.json"

    write_report(report, out_path)
    _print_summary(customer_id, out_path, report.headline)

    if report.pass_rate < min_pass_rate:
        print(
            f"  pass_rate {report.pass_rate:.3f} below --min-pass-rate "
            f"{min_pass_rate:.3f}; non-zero exit",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clarion-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the scripted evaluation harness")
    run.add_argument("customer", help="customer id or 'all'")
    run.add_argument(
        "--out",
        type=Path,
        default=None,
        help="dir to write reports into (default: per-customer data dir)",
    )
    run.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.9,
        help="exit non-zero if any customer's pass_rate falls below this",
    )

    args = parser.parse_args(argv)
    customers = list(TEMPLATES) if args.customer == "all" else [args.customer]

    exit_code = 0
    for cid in customers:
        if cid not in TEMPLATES:
            print(
                f"unknown customer {cid!r}; known: {sorted(TEMPLATES)}",
                file=sys.stderr,
            )
            exit_code = 2
            continue
        code = cmd_run(cid, out_dir=args.out, min_pass_rate=args.min_pass_rate)
        exit_code = exit_code or code
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
