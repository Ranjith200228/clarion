"""Unit tests for the span tracer."""

from __future__ import annotations

import time

from clarion.observability.tracer import Tracer, new_trace_id


def test_single_span_records_duration() -> None:
    tracer = Tracer()
    with tracer.span("work"):
        time.sleep(0.005)
    trace = tracer.emit()
    assert len(trace.spans) == 1
    span = trace.spans[0]
    assert span.name == "work"
    assert span.parent_id is None
    assert span.duration_ms is not None
    assert span.duration_ms >= 4  # allow some scheduling slop


def test_nested_spans_record_parent_ids() -> None:
    tracer = Tracer()
    with tracer.span("agent.chat"):
        with tracer.span("retrieval"):
            pass
        with tracer.span("react.step"):
            with tracer.span("llm.complete"):
                pass
            with tracer.span("tool.search_slots"):
                pass

    spans = tracer.emit().spans
    names = [s.name for s in spans]
    # Order is "open order": parent before children.
    assert names == [
        "agent.chat",
        "retrieval",
        "react.step",
        "llm.complete",
        "tool.search_slots",
    ]

    # Parent ids form the tree.
    by_name = {s.name: s for s in spans}
    assert by_name["agent.chat"].parent_id is None
    assert by_name["retrieval"].parent_id == by_name["agent.chat"].span_id
    assert by_name["react.step"].parent_id == by_name["agent.chat"].span_id
    assert by_name["llm.complete"].parent_id == by_name["react.step"].span_id
    assert by_name["tool.search_slots"].parent_id == by_name["react.step"].span_id


def test_attributes_captured() -> None:
    tracer = Tracer()
    with tracer.span("llm.complete", model="gpt-4o-mini") as span:
        span.set("input_tokens", 120)
        span.update({"output_tokens": 30, "cost_usd": 0.001})

    s = tracer.emit().spans[0]
    assert s.attributes == {
        "model": "gpt-4o-mini",
        "input_tokens": 120,
        "output_tokens": 30,
        "cost_usd": 0.001,
    }


def test_current_returns_innermost_open_span() -> None:
    tracer = Tracer()
    assert tracer.current is None
    with tracer.span("outer") as outer:
        assert tracer.current is outer
        with tracer.span("inner") as inner:
            assert tracer.current is inner
        assert tracer.current is outer
    assert tracer.current is None


def test_emit_carries_identity_fields() -> None:
    tracer = Tracer(customer_id="ophthalmology", conversation_id="conv_abc123")
    with tracer.span("agent.chat"):
        pass
    trace = tracer.emit()
    assert trace.customer_id == "ophthalmology"
    assert trace.conversation_id == "conv_abc123"
    assert trace.trace_id == tracer.trace_id
    assert trace.trace_id.startswith("trace_")


def test_to_dict_serializes_datetimes_as_iso() -> None:
    tracer = Tracer()
    with tracer.span("work", k=4):
        pass
    payload = tracer.emit().to_dict()
    span = payload["spans"][0]
    assert isinstance(span["started_at"], str)
    assert isinstance(span["ended_at"], str)
    assert span["attributes"] == {"k": 4}


def test_new_trace_id_is_unique_and_prefixed() -> None:
    a = new_trace_id()
    b = new_trace_id()
    assert a != b
    assert a.startswith("trace_")
