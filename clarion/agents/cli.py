"""Interactive REPL for poking the agent locally.

Usage::

    OPENAI_API_KEY=sk-... poetry run python -m clarion.agents.cli ophthalmology
    OPENAI_API_KEY=sk-... poetry run python -m clarion.agents.cli orthopedics

You'll get a prompt; type messages, the agent replies. Type ``:quit`` to
exit. The ingest CLI must have been run for the customer first
(`python -m clarion.pipelines.ingest all <customer>`) so the FAISS
index and SQLite store exist.

This is a developer convenience, not a production interface — Phase 8
adds the FastAPI service.
"""

from __future__ import annotations

import argparse
import sys

from clarion.agents.agent import Agent
from clarion.agents.openai_client import OpenAIClient
from clarion.config import get_settings, load_customer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clarion-chat")
    parser.add_argument(
        "customer",
        nargs="?",
        help="Customer id (YAML stem). Defaults to CLARION_CUSTOMER.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override CLARION_MODEL (default: gpt-4o-mini).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    customer_id = args.customer or settings.customer
    customer = load_customer(customer_id, settings=settings)

    llm = OpenAIClient(model=args.model)
    agent = Agent.build(customer=customer, llm=llm, data_dir=settings.data_dir)

    print(f"Clarion ({customer.display_name}) — model={llm.model}. " "Type :quit to exit.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {":quit", ":q", "exit"}:
            return 0
        reply = agent.chat(line)
        print(f"clarion> {reply}\n")


if __name__ == "__main__":
    sys.exit(main())
