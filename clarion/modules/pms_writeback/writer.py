"""PMS writeback writer.

Drains an ``ExtractionContext`` through an ``Extractor`` then writes
two JSON files to disk:

    <data_dir>/<customer_id>/pms_writeback/<conversation_id>/
        summary.json   (ConversationSummary)
        task.json      (PmsTaskWriteback)

Both files run through ``clarion.sentinel.phi.redact`` before
serialization so phones / emails / member ids never leak into the
written outputs. The Phase 6 PHI redaction layer is the load-bearing
contract here.

One ``WritebackOutcome`` is returned per call so the harness (commit 5)
can record which conversation ids produced output, and Phase M1
accuracy (commit 4) can read them back.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from clarion.modules.pms_writeback.extractor import (
    ExtractionContext,
    Extractor,
    HeuristicExtractor,
)
from clarion.schemas import (
    ConversationSummary,
    PmsTaskWriteback,
    WritebackTaskPriority,
)
from clarion.sentinel.phi import redact

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WritebackOutcome:
    """The two files that landed on disk for one conversation."""

    summary_path: Path
    task_path: Path
    summary: ConversationSummary
    task: PmsTaskWriteback


class PmsWritebackWriter:
    """Write ``summary.json`` + ``task.json`` per conversation.

    Defaults to the ``HeuristicExtractor``; a future ``LLMExtractor`` can
    be plugged in by passing ``extractor=`` to the constructor.
    """

    def __init__(self, *, extractor: Extractor | None = None) -> None:
        self._extractor: Extractor = extractor or HeuristicExtractor()

    def write(self, ctx: ExtractionContext, *, data_dir: Path) -> WritebackOutcome:
        """Extract + persist for one conversation.

        Returns the paths + the in-memory shapes that landed on disk
        so callers can chain (e.g. add the paths to a HarnessResult).
        """
        summary = self._extractor.extract(ctx)
        task = self._build_task(ctx, summary)

        out_dir = data_dir / ctx.customer_id / "pms_writeback" / ctx.conversation_id
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_path = out_dir / "summary.json"
        task_path = out_dir / "task.json"

        _write_json(summary_path, _redact_payload(summary.model_dump(mode="json")))
        _write_json(task_path, _redact_payload(task.model_dump(mode="json")))

        log.info(
            "pms_writeback: wrote %s and %s for %s/%s",
            summary_path,
            task_path,
            ctx.customer_id,
            ctx.conversation_id,
        )
        return WritebackOutcome(
            summary_path=summary_path,
            task_path=task_path,
            summary=summary,
            task=task,
        )

    # ---------- internals ----------

    def _build_task(self, ctx: ExtractionContext, summary: ConversationSummary) -> PmsTaskWriteback:
        """Build the matching PmsTaskWriteback for a summary.

        Subject + body are derived from the outcome; priority escalates
        for emergencies; assignee_group routes to ``triage`` for
        emergencies and ``front_desk`` otherwise.
        """
        outcome = summary.outcome
        priority: WritebackTaskPriority = "urgent" if outcome == "escalated_emergency" else "normal"
        assignee = "triage" if outcome == "escalated_emergency" else "front_desk"

        subject = self._build_subject(summary)
        body = self._build_body(summary)

        return PmsTaskWriteback(
            customer_id=ctx.customer_id,
            conversation_id=ctx.conversation_id,
            task_id=f"task_{uuid.uuid4().hex[:12]}",
            generated_at=datetime.now(UTC),
            subject=subject,
            body=body,
            priority=priority,
            status="open",
            patient_id=summary.patient_id,
            assignee_group=assignee,
            summary_ref="summary.json",
        )

    def _build_subject(self, summary: ConversationSummary) -> str:
        # Subject is the load-bearing field for front-desk queue
        # triage. Keep it terse + scannable.
        outcome = summary.outcome
        appt = summary.appointment_type or summary.intent or "general inquiry"
        prefixes: dict[str, str] = {
            "booked": "Confirm booking",
            "rescheduled": "Confirm reschedule",
            "cancelled": "Confirm cancellation",
            "task_created": "Follow up",
            "escalated_emergency": "URGENT: patient escalation",
            "refused_clinical": "Clinician callback",
            "info_provided": "FYI",
            "unresolved": "Follow up — unresolved",
        }
        prefix = prefixes.get(outcome, "Follow up")
        return f"{prefix} — {appt}"[:200]

    def _build_body(self, summary: ConversationSummary) -> str:
        lines: list[str] = []
        if summary.caller_name:
            lines.append(f"Caller: {summary.caller_name}")
        if summary.patient_id:
            lines.append(f"Patient: {summary.patient_id}")
        if summary.payer:
            lines.append(f"Payer: {summary.payer}")
        if summary.appointment_type:
            lines.append(f"Appointment type: {summary.appointment_type}")
        if summary.appointment_time:
            lines.append(f"Appointment time: {summary.appointment_time.isoformat()}")
        lines.append(f"Outcome: {summary.outcome}")
        if summary.escalated:
            lines.append("Escalation: yes")
        if summary.notes:
            lines.append("")
            lines.append("Notes:")
            lines.append(summary.notes)
        return ("\n".join(lines))[:4000]


# ---------- helpers ----------


def _redact_payload(payload: dict[str, object]) -> dict[str, object]:
    """PHI-redact every string value in a model_dump payload."""
    out: dict[str, object] = {}
    for k, v in payload.items():
        if isinstance(v, str):
            out[k] = redact(v)
        elif isinstance(v, dict):
            out[k] = _redact_payload(v)
        elif isinstance(v, list):
            out[k] = [redact(x) if isinstance(x, str) else x for x in v]
        else:
            out[k] = v
    return out


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
