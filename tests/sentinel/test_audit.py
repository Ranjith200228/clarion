"""Tests for the audit log writer."""

from __future__ import annotations

import json
from pathlib import Path

from clarion.sentinel.audit import AuditLog, AuditTurn, new_conversation_id


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writes_one_jsonl_record_per_turn(tmp_path: Path) -> None:
    log = AuditLog(path=tmp_path / "audit.jsonl", customer_id="demo")
    log.write(AuditTurn(user_message="hi", agent_reply="hello"))
    log.write(AuditTurn(user_message="thanks", agent_reply="you're welcome"))

    records = _read_jsonl(log.path)
    assert len(records) == 2
    assert records[0]["user_message"] == "hi"
    assert records[1]["agent_reply"] == "you're welcome"


def test_for_customer_path_convention(tmp_path: Path) -> None:
    log = AuditLog.for_customer("demo", tmp_path)
    assert log.path == tmp_path / "demo" / "audit.jsonl"
    log.write(AuditTurn(user_message="x", agent_reply="y"))
    assert log.path.exists()


def test_conversation_id_is_stable_across_writes(tmp_path: Path) -> None:
    log = AuditLog(path=tmp_path / "audit.jsonl", customer_id="demo")
    log.write(AuditTurn(user_message="a", agent_reply="b"))
    log.write(AuditTurn(user_message="c", agent_reply="d"))

    records = _read_jsonl(log.path)
    ids = {r["conversation_id"] for r in records}
    assert len(ids) == 1
    assert log.conversation_id in ids


def test_redacts_phi_in_user_message_and_reply(tmp_path: Path) -> None:
    log = AuditLog(path=tmp_path / "audit.jsonl", customer_id="demo")
    log.write(
        AuditTurn(
            user_message=(
                "I'm calling for pat_001, member AET-9981, "
                "phone 555-555-1234, email me@example.com"
            ),
            agent_reply="Sure, calling pat_001 at 555-555-1234.",
        )
    )
    records = _read_jsonl(log.path)
    r = records[0]
    text = json.dumps(r)
    assert "pat_001" not in text
    assert "AET-9981" not in text
    assert "555-555-1234" not in text
    assert "me@example.com" not in text
    # Tag counts surface so we can spot outliers.
    redactions = r["redactions"]
    assert redactions["<PATIENT_ID>"] >= 1
    assert redactions["<PHONE>"] >= 1
    assert redactions["<EMAIL>"] == 1
    assert redactions["<MEMBER_ID>"] == 1


def test_records_guardrail_decision(tmp_path: Path) -> None:
    log = AuditLog(path=tmp_path / "audit.jsonl", customer_id="demo")
    log.write(
        AuditTurn(
            user_message="should I take my drops?",
            agent_reply="That's a clinical question — I'll have a nurse call back.",
            guardrail="clinical_advice",
        )
    )
    records = _read_jsonl(log.path)
    assert records[0]["guardrail"] == "clinical_advice"


def test_redacts_phi_inside_tool_call_arguments(tmp_path: Path) -> None:
    log = AuditLog(path=tmp_path / "audit.jsonl", customer_id="demo")
    log.write(
        AuditTurn(
            user_message="anything",
            agent_reply="anything",
            tool_calls=[
                {
                    "name": "book_appointment",
                    "arguments": {
                        "slot_id": "slot_demo_1",
                        "patient_id": "pat_001",
                        "notes": "Callback 555-555-1234.",
                    },
                    "ok": True,
                }
            ],
        )
    )
    records = _read_jsonl(log.path)
    tool = records[0]["tool_calls"][0]
    # patient_id and phone are redacted inside the serialized args.
    args = tool["arguments"]
    assert "pat_001" not in args
    assert "555-555-1234" not in args
    assert "<PATIENT_ID>" in args
    assert "<PHONE>" in args
    # The other fields survive.
    assert tool["name"] == "book_appointment"
    assert tool["ok"] is True


def test_clean_text_yields_empty_redactions_dict(tmp_path: Path) -> None:
    log = AuditLog(path=tmp_path / "audit.jsonl", customer_id="demo")
    log.write(
        AuditTurn(user_message="hi", agent_reply="hello, how can I help?"),
    )
    records = _read_jsonl(log.path)
    assert records[0]["redactions"] == {}


def test_new_conversation_id_is_unique_and_prefixed() -> None:
    a = new_conversation_id()
    b = new_conversation_id()
    assert a != b
    assert a.startswith("conv_")
    assert b.startswith("conv_")
