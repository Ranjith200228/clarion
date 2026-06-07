"""Tests for the JSONL trace writer."""

from __future__ import annotations

import json
from pathlib import Path

from clarion.observability.tracer import Tracer
from clarion.observability.writer import TraceWriter


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _build_trace(label: str) -> Tracer:
    tracer = Tracer(customer_id="demo", conversation_id="conv_1")
    with tracer.span("agent.chat", label=label) as root:
        root.set("user_chars", 42)
        with tracer.span("retrieval", k=4, hit_count=2):
            pass
    return tracer


def test_writes_one_jsonl_record_per_trace(tmp_path: Path) -> None:
    writer = TraceWriter(path=tmp_path / "traces.jsonl")
    writer.write(_build_trace("a").emit())
    writer.write(_build_trace("b").emit())

    records = _read_jsonl(writer.path)
    assert len(records) == 2
    assert records[0]["customer_id"] == "demo"
    assert records[1]["customer_id"] == "demo"


def test_for_customer_path_convention(tmp_path: Path) -> None:
    writer = TraceWriter.for_customer("demo", tmp_path)
    assert writer.path == tmp_path / "demo" / "traces.jsonl"
    writer.write(_build_trace("a").emit())
    assert writer.path.exists()


def test_record_carries_trace_id_and_spans(tmp_path: Path) -> None:
    writer = TraceWriter(path=tmp_path / "traces.jsonl")
    trace = _build_trace("a").emit()
    record = writer.write(trace)
    assert record["trace_id"] == trace.trace_id
    assert len(record["spans"]) == 2
    assert record["spans"][0]["name"] == "agent.chat"
    assert record["spans"][1]["name"] == "retrieval"


def test_appends_rather_than_overwrites(tmp_path: Path) -> None:
    writer = TraceWriter(path=tmp_path / "traces.jsonl")
    writer.write(_build_trace("first").emit())
    # Reload by constructing a NEW writer at the same path.
    writer2 = TraceWriter(path=tmp_path / "traces.jsonl")
    writer2.write(_build_trace("second").emit())
    assert len(_read_jsonl(writer.path)) == 2
