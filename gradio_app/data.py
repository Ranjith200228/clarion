"""Load + validate the Phase 13 JSON contracts for the Gradio UI.

The UI never reads raw dicts. Both files are validated through their
Pydantic models so the rest of ``gradio_app`` can rely on typed access
to every field. If the schema_version in a file doesn't match what the
UI was compiled against, we surface a clear error rather than render
garbage.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from clarion.schemas import (
    REPORT_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    EvaluationReport,
    TraceReport,
)

log = logging.getLogger(__name__)

# Customers the UI knows about. Matches the Phase 9 / Phase 13 demo set.
KNOWN_CUSTOMERS = ("ophthalmology", "orthopedics")

# Default location of the per-customer JSON files. Overridable via env
# var so HF Spaces deployments can point at a different mount.
DEFAULT_DATA_DIR = Path(os.environ.get("CLARION_DATA_DIR", "data"))


@dataclass(frozen=True)
class CustomerArtifacts:
    """The two JSON files for one customer, loaded + validated."""

    customer_id: str
    report: EvaluationReport
    trace_report: TraceReport


class SchemaVersionMismatchError(RuntimeError):
    """Raised when a file's schema_version doesn't match the UI's expected version."""


# ---------- discovery ----------


def report_path(customer_id: str, data_dir: Path | None = None) -> Path:
    base = data_dir if data_dir is not None else DEFAULT_DATA_DIR
    return base / customer_id / f"report_{customer_id}.json"


def trace_path(customer_id: str, data_dir: Path | None = None) -> Path:
    base = data_dir if data_dir is not None else DEFAULT_DATA_DIR
    return base / customer_id / f"trace_{customer_id}.json"


def available_customers(data_dir: Path | None = None) -> list[str]:
    """Return the customers whose ``report_*.json`` exist on disk.

    Falls back to KNOWN_CUSTOMERS when nothing has been generated yet so
    the UI still loads (it'll display empty-state messages instead).
    """
    found: list[str] = []
    for customer_id in KNOWN_CUSTOMERS:
        if report_path(customer_id, data_dir).is_file():
            found.append(customer_id)
    return found or list(KNOWN_CUSTOMERS)


# ---------- typed loaders ----------


def load_report(customer_id: str, data_dir: Path | None = None) -> EvaluationReport:
    """Read + validate ``report_<customer_id>.json``.

    Raises FileNotFoundError when the file is missing; the UI checks for
    this and shows an empty-state hint pointing at the eval CLI.
    """
    path = report_path(customer_id, data_dir)
    raw = _read_json(path)
    _check_schema_version(raw, expected=REPORT_SCHEMA_VERSION, path=path, kind="report")
    return EvaluationReport.model_validate(raw)


def load_trace_report(customer_id: str, data_dir: Path | None = None) -> TraceReport:
    """Read + validate ``trace_<customer_id>.json``."""
    path = trace_path(customer_id, data_dir)
    raw = _read_json(path)
    _check_schema_version(raw, expected=TRACE_SCHEMA_VERSION, path=path, kind="trace")
    return TraceReport.model_validate(raw)


def load_artifacts(customer_id: str, data_dir: Path | None = None) -> CustomerArtifacts:
    """Load both files for one customer."""
    return CustomerArtifacts(
        customer_id=customer_id,
        report=load_report(customer_id, data_dir),
        trace_report=load_trace_report(customer_id, data_dir),
    )


# ---------- helpers ----------


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Generate it with: "
            f"python -m clarion.eval --customer {path.parent.name}"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a JSON object at the top level")
    return raw


def _check_schema_version(raw: dict[str, object], *, expected: str, path: Path, kind: str) -> None:
    actual = raw.get("schema_version")
    if actual != expected:
        # The lock rule says additive changes keep the version stable.
        # Anything else is a breaking change and the UI must refuse to
        # render rather than silently misinterpret fields.
        raise SchemaVersionMismatchError(
            f"{kind} file at {path} declares schema_version={actual!r} "
            f"but this UI was built for {expected!r}. Regenerate the "
            f"file or upgrade the UI."
        )
