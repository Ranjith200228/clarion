"""Ingest CLI — populate a customer's data stores from seeds and rules.

Usage::

    poetry run python -m clarion.pipelines.ingest structured ophthalmology
    poetry run python -m clarion.pipelines.ingest all orthopedics

The CLI is intentionally tiny. Heavy lifting lives in
``clarion.pipelines.structured.seed`` and (later) ``...unstructured.build``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clarion.config import Settings, get_settings, load_customer
from clarion.pipelines.structured.seed import seed_structured


def _run_structured(customer_id: str, settings: Settings) -> int:
    # Validate that the customer config exists; tooling shouldn't silently
    # seed for a customer whose YAML you forgot to add.
    cfg = load_customer(customer_id, settings=settings)
    summary = seed_structured(cfg.customer_id, data_dir=settings.data_dir)
    print(
        f"[structured] {cfg.customer_id}: "
        f"{summary.providers} providers, {summary.slots} slots, "
        f"{summary.eligibility} eligibility records"
    )
    return 0


def _run_unstructured(customer_id: str, settings: Settings) -> int:
    # Lazy import so the structured-only path doesn't pay the FAISS load.
    from clarion.rag.builder import build_customer_index

    cfg = load_customer(customer_id, settings=settings)
    n_chunks = build_customer_index(cfg, data_dir=settings.data_dir)
    print(f"[unstructured] {cfg.customer_id}: indexed {n_chunks} chunks")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clarion-ingest")
    parser.add_argument(
        "stage",
        choices=["structured", "unstructured", "all"],
        help="Which half of the dual pipeline to run.",
    )
    parser.add_argument(
        "customer",
        nargs="?",
        help="Customer id (YAML stem). Defaults to CLARION_CUSTOMER.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override data dir (default: CLARION_DATA_DIR or <repo>/data).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.data_dir is not None:
        settings = settings.model_copy(update={"data_dir": args.data_dir})
    customer_id = args.customer or settings.customer

    if args.stage in {"structured", "all"}:
        _run_structured(customer_id, settings)
    if args.stage in {"unstructured", "all"}:
        _run_unstructured(customer_id, settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
