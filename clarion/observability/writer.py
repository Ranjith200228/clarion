"""JSONL trace writer.

One file per customer at ``<data_dir>/<customer_id>/traces.jsonl``. Each
line is one ``Trace.to_dict()`` payload — a full conversation turn with
nested spans. The dashboard (Phase 13) tails this file to render the
"trace explorer" panel.

Append-only. Thread-safe (one process-level lock) so the FastAPI service
in Phase 8 can share an instance across concurrent requests.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clarion.observability.tracer import Trace


@dataclass
class TraceWriter:
    """Append-only JSONL writer for ``Trace`` payloads."""

    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_customer(cls, customer_id: str, data_dir: Path) -> TraceWriter:
        return cls(path=data_dir / customer_id / "traces.jsonl")

    def write(self, trace: Trace) -> dict[str, Any]:
        """Append the trace as one JSON line. Returns the written record."""
        record = trace.to_dict()
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)
        return record
