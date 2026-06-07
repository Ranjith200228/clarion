"""Per-conversation audit log.

One JSONL file per customer at ``<data_dir>/<customer_id>/audit.jsonl``.
Every record is a single line of JSON with these fields::

    timestamp        ISO-8601 with offset
    conversation_id  uuid generated when the Agent boots
    customer_id      from CustomerConfig
    user_message     PHI-redacted
    agent_reply      PHI-redacted
    redactions       {"<TAG>": count, ...} so we can spot outliers
    guardrail        "safe" | "emergency" | "clinical_advice"
    tool_calls       list of {name, arguments_redacted, ok, error?}
    steps            int  (react loop steps consumed)

Stored append-only — never mutate prior lines. Rotation / retention is
out of scope for the MVP; Phase 14 deployment can add it.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clarion.sentinel.phi import redact_with_counts


def new_conversation_id() -> str:
    return f"conv_{uuid.uuid4().hex[:12]}"


@dataclass
class AuditTurn:
    """One turn's worth of audit data. Always PHI-redacted before write."""

    user_message: str
    agent_reply: str
    guardrail: str = "safe"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditLog:
    """Thread-safe append-only JSONL writer."""

    path: Path
    customer_id: str
    conversation_id: str = field(default_factory=new_conversation_id)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_customer(cls, customer_id: str, data_dir: Path) -> AuditLog:
        return cls(path=data_dir / customer_id / "audit.jsonl", customer_id=customer_id)

    def write(self, turn: AuditTurn) -> dict[str, Any]:
        """Append one turn to the audit log. Returns the written record."""
        user_red = redact_with_counts(turn.user_message)
        reply_red = redact_with_counts(turn.agent_reply)

        # Merge tag counts so the audit log surfaces totals per turn.
        redactions = dict(user_red.tag_counts)
        for tag, n in reply_red.tag_counts.items():
            redactions[tag] = redactions.get(tag, 0) + n

        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "conversation_id": self.conversation_id,
            "customer_id": self.customer_id,
            "user_message": user_red.text,
            "agent_reply": reply_red.text,
            "redactions": redactions,
            "guardrail": turn.guardrail,
            "tool_calls": [_redact_tool_call(tc) for tc in turn.tool_calls],
            "steps": turn.steps,
        }
        if turn.extra:
            record["extra"] = turn.extra

        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)
        return record


def _redact_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    """Redact a tool-call summary before it lands in the audit log.

    ``arguments`` is serialized through redact_with_counts so a free-text
    notes field never leaks PHI. The tag counts are not surfaced per
    call — the turn-level ``redactions`` field already captures totals.
    """
    raw_args = json.dumps(call.get("arguments", {}), ensure_ascii=False)
    redacted_args = redact_with_counts(raw_args).text
    return {
        "name": call.get("name"),
        "arguments": redacted_args,
        "ok": call.get("ok"),
        "error": call.get("error"),
    }
