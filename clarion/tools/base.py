"""Foundation for Clarion tools.

A tool is a small Python class that:

* declares its ``name`` (matches the ``ToolName`` Literal in CustomerConfig)
* declares its Pydantic input/output models
* implements ``run(input, ctx)`` returning a populated output model

The agent (Phase 5) discovers tools through a registry (commit 9) and
respects the per-customer ``enabled_tools`` allowlist — no global tools.

Errors
------

Tools never raise to the agent. They catch known failure modes and return
``ok=False`` with a human-readable ``error`` string. ``run_with_retry``
wraps the SQLite call sites so transient ``OperationalError`` retries
once before the wrapping tool surfaces a structured error.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Generic, Protocol, TypeVar

from pydantic import BaseModel

from clarion.config import CustomerConfig
from clarion.pipelines.structured import StructuredStore
from clarion.schemas.tools import ToolOutput

log = logging.getLogger(__name__)

# Variance markers required because Protocol generics enforce them:
# inputs are contravariant, outputs are covariant.
InputT_contra = TypeVar("InputT_contra", bound=BaseModel, contravariant=True)
OutputT_co = TypeVar("OutputT_co", bound=ToolOutput, covariant=True)
# Plain typevar used by run_with_retry, which is just an internal helper.
RetryT = TypeVar("RetryT", bound=ToolOutput)


@dataclass(frozen=True)
class ToolContext:
    """Runtime context every tool receives.

    Holds the per-customer state (config + structured store). Phase 5 will
    pass the retriever in too for an FAQ-style tool; until then, tools that
    only need the structured store keep their interface minimal.
    """

    customer: CustomerConfig
    structured: StructuredStore


class ToolError(RuntimeError):
    """Raised inside a tool when no recovery is possible. The dispatcher
    converts it into ``ok=False`` on the way out — never reaches the agent.
    """


class Tool(Protocol, Generic[InputT_contra, OutputT_co]):
    """Protocol every tool implements.

    Subclasses also set ``name``, ``input_model``, ``output_model`` as
    class-level attributes so the registry can introspect them.
    """

    name: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[ToolOutput]]

    def run(self, input: InputT_contra, ctx: ToolContext) -> OutputT_co: ...


# ---------- retry helper ----------


_TRANSIENT_DB_EXCEPTIONS = (sqlite3.OperationalError,)


def run_with_retry(
    fn: Callable[[], RetryT],
    *,
    attempts: int = 2,
    backoff_seconds: float = 0.05,
) -> RetryT:
    """Retry ``fn`` once on transient SQLite errors (e.g. database is locked).

    ``attempts=2`` means at most one retry. Backoff is intentionally tiny;
    the goal is robustness against momentary contention, not a full retry
    framework — tools that need richer retry semantics can call this twice.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except _TRANSIENT_DB_EXCEPTIONS as e:
            last_exc = e
            log.warning("tool retry %d/%d after transient error: %s", i + 1, attempts, e)
            time.sleep(backoff_seconds)
    # The static checker can't see that the loop guarantees last_exc is set.
    assert last_exc is not None
    raise last_exc
