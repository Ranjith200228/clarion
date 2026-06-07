"""End-to-end observability tests on real customer configs.

Phase 7 acceptance: *every interaction generates traces*. We assert:

- Every Agent.chat call emits one trace with a unique trace_id and the
  expected span hierarchy.
- The trace persists to JSONL when a TraceWriter is attached.
- llm.complete spans carry model + token + cost attributes.
- tool.<name> spans carry the tool's ok flag.
- The retrieval span surfaces hit_count and top_source.
- Guardrail short-circuits still emit a trace (with a guardrails.check
  span fired=True), proving the "every interaction" promise.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from clarion.agents import Agent, FakeLLM, LLMResponse, LLMUsage, ToolCall
from clarion.config import CustomerConfig
from clarion.observability import TraceWriter
from clarion.pipelines.structured import StructuredStore
from clarion.pipelines.unstructured import chunk_rules_dir
from clarion.rag.embeddings import TfidfEmbedder
from clarion.rag.retriever import Retriever


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _build_retriever(customer: CustomerConfig, out_dir: Path) -> Retriever:
    chunks = chunk_rules_dir(customer.rules_path)
    return Retriever.build_and_save(chunks, embedder=TfidfEmbedder(), out_dir=out_dir)


def test_booking_turn_trace_has_full_span_hierarchy(
    ophthalmology_config: CustomerConfig,
    store_with_ophthalmology_seed: StructuredStore,
    tmp_path: Path,
) -> None:
    retriever = _build_retriever(ophthalmology_config, tmp_path / "rag")
    traces = TraceWriter(path=tmp_path / "traces.jsonl")
    fake = FakeLLM(
        responses=[
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="search_slots",
                        arguments={
                            "appointment_type": "Cataract Pre-Op Consult",
                            "on_or_after": date(2026, 6, 1).isoformat(),
                        },
                    ),
                ),
                usage=LLMUsage(model="gpt-4o-mini", input_tokens=300, output_tokens=50),
            ),
            LLMResponse(
                content="Found one slot June 15 9 AM.",
                usage=LLMUsage(model="gpt-4o-mini", input_tokens=420, output_tokens=15),
            ),
        ]
    )

    agent = Agent.from_customer(
        customer=ophthalmology_config,
        llm=fake,
        structured=store_with_ophthalmology_seed,
        retriever=retriever,
    )
    agent.traces = traces
    reply = agent.chat("Hi, I'd like a cataract pre-op consult after June 1.")
    assert "June 15" in reply

    # One trace per chat() call.
    records = _read_jsonl(traces.path)
    assert len(records) == 1
    trace = records[0]
    assert trace["customer_id"] == "ophthalmology"
    assert isinstance(trace["trace_id"], str) and trace["trace_id"].startswith("trace_")

    span_names = [s["name"] for s in trace["spans"]]
    # Order is "open order": parent before children.
    assert span_names[0] == "agent.chat"
    assert "guardrails.check" in span_names
    assert "retrieval" in span_names
    assert span_names.count("react.step") == 2  # tool step + final reply step
    assert span_names.count("llm.complete") == 2
    assert "tool.search_slots" in span_names

    # llm.complete spans have model + tokens + cost populated.
    llm_spans = [s for s in trace["spans"] if s["name"] == "llm.complete"]
    for s in llm_spans:
        attrs = s["attributes"]
        assert attrs["model"] == "gpt-4o-mini"
        assert attrs["input_tokens"] > 0
        assert attrs["output_tokens"] > 0
        # gpt-4o-mini pricing applied (non-zero, positive).
        assert attrs["cost_usd"] > 0
        assert attrs["tool_calls_count"] >= 0

    # tool.search_slots span has ok=True.
    tool_span = next(s for s in trace["spans"] if s["name"] == "tool.search_slots")
    assert tool_span["attributes"]["ok"] is True

    # retrieval span has hit_count + top_source.
    r_span = next(s for s in trace["spans"] if s["name"] == "retrieval")
    assert r_span["attributes"]["hit_count"] >= 1
    assert r_span["attributes"]["top_source"]  # non-empty string


def test_emergency_short_circuit_still_emits_trace(
    ophthalmology_config: CustomerConfig,
    store_with_ophthalmology_seed: StructuredStore,
    tmp_path: Path,
) -> None:
    """The 'every interaction generates traces' promise covers guardrail
    short-circuits too."""
    fake = FakeLLM(responses=[])  # LLM should NOT be consulted.
    traces = TraceWriter(path=tmp_path / "traces.jsonl")
    agent = Agent.from_customer(
        customer=ophthalmology_config,
        llm=fake,
        structured=store_with_ophthalmology_seed,
        retriever=None,
    )
    agent.traces = traces

    reply = agent.chat("I suddenly lost my sight!")
    assert "911" in reply

    records = _read_jsonl(traces.path)
    assert len(records) == 1
    span_names = [s["name"] for s in records[0]["spans"]]
    assert span_names == ["agent.chat", "guardrails.check"]
    g_span = next(s for s in records[0]["spans"] if s["name"] == "guardrails.check")
    assert g_span["attributes"]["fired"] is True


def test_trace_attached_audit_carries_trace_id(
    ophthalmology_config: CustomerConfig,
    store_with_ophthalmology_seed: StructuredStore,
    tmp_path: Path,
) -> None:
    """When both writers are attached, the audit row carries the matching
    trace_id so the dashboard can pivot from one to the other."""
    from clarion.sentinel.audit import AuditLog

    retriever = _build_retriever(ophthalmology_config, tmp_path / "rag")
    traces = TraceWriter(path=tmp_path / "traces.jsonl")
    audit = AuditLog(path=tmp_path / "audit.jsonl", customer_id="ophthalmology")
    fake = FakeLLM(
        responses=[
            LLMResponse(
                content="ack",
                usage=LLMUsage(model="gpt-4o-mini", input_tokens=10, output_tokens=2),
            )
        ]
    )
    agent = Agent.from_customer(
        customer=ophthalmology_config,
        llm=fake,
        structured=store_with_ophthalmology_seed,
        retriever=retriever,
    )
    agent.audit = audit
    agent.traces = traces

    agent.chat("hi")

    trace_records = _read_jsonl(traces.path)
    audit_records = _read_jsonl(audit.path)
    assert audit_records[0]["extra"]["trace_id"] == trace_records[0]["trace_id"]
