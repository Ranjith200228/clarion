"""Tests for the Sentinel Operations Center: data_sources rollup +
view HTML.

Strategy mirrors the Phase B test files:

- ``build_sentinel_ops`` reads tmp on-disk artifacts; we write
  schema-valid TraceEntry payloads with realistic escalation
  reasons + judge fields and assert the rollup math.
- The view tests run on in-memory ``SentinelOpsSnapshot`` instances
  and assert structural facts (right ``clarion-*`` classes, right
  KPI labels, empty-state path).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gradio_app import data_sources
from gradio_app.data_sources import (
    AuditTailItem,
    JudgeAgreement,
    SentinelOpsSnapshot,
    SignalContribution,
)
from gradio_app.views import sentinel_ops as view

# ---------- helpers (mirror those in test_data_sources) ----------


def _write_report(base: Path, *, customer_id: str, precision: float, recall: float) -> None:
    customer_dir = base / customer_id
    customer_dir.mkdir(parents=True, exist_ok=True)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    payload = {
        "schema_version": "1.0.0",
        "customer_id": customer_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario_count": 10,
        "pass_rate": 1.0,
        "metrics": {
            "scenario_count": 10,
            "pass_rate": 1.0,
            "containment_rate": 0.7,
            "booking_accuracy": 1.0,
            "booking_total": 10,
            "booking_correct": 10,
            "hallucination_rate": 0.0,
            "hallucination_with_judge": 0,
            "escalation_precision": precision,
            "escalation_recall": recall,
            "escalation_f1": f1,
            "escalation_accuracy": 0.9,
            "safety_catch_rate": 1.0,
            "safety_total": 5,
            "safety_caught": 5,
            "avg_turns_to_resolution": 1.5,
            "cost_per_request_usd": 0.001,
        },
        "by_difficulty": {},
        "by_intent": {},
        "headline": {},
    }
    (customer_dir / f"report_{customer_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_trace(base: Path, *, customer_id: str, entries: list[dict]) -> None:
    customer_dir = base / customer_id
    customer_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "customer_id": customer_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "entries": entries,
    }
    (customer_dir / f"trace_{customer_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _entry(
    *,
    scenario_id: str,
    customer_id: str = "ophthalmology",
    escalation_score: float | None = None,
    escalation_reasons: list[str] | None = None,
    judge_hallucination: float | None = None,
    judge_booking_correct: float | None = None,
    judge_violations: list[str] | None = None,
    actual_outcome: str = "booked",
) -> dict:
    return {
        "scenario_id": scenario_id,
        "customer_id": customer_id,
        "trace_id": f"trace_{scenario_id}",
        "difficulty": "clear",
        "intent": "book",
        "agent_replies": ["confirmed."],
        "tools_called": [],
        "actual_outcome": actual_outcome,
        "passed": True,
        "escalation_score": escalation_score,
        "escalation_reasons": escalation_reasons or [],
        "judge_hallucination": judge_hallucination,
        "judge_booking_correct": judge_booking_correct,
        "judge_violations": judge_violations or [],
        "duration_ms": 500.0,
        "cost_usd": 0.001,
        "input_tokens": 200,
        "output_tokens": 60,
        "step_count": 1,
    }


@pytest.fixture
def sentinel_data_dir(tmp_path: Path) -> Path:
    """One tenant with rich, multi-signal trace data."""
    _write_report(
        tmp_path,
        customer_id="ophthalmology",
        precision=0.62,
        recall=1.0,
    )
    _write_trace(
        tmp_path,
        customer_id="ophthalmology",
        entries=[
            _entry(
                scenario_id="s_001",
                escalation_score=0.15,
                escalation_reasons=[],
                judge_hallucination=0.0,
                judge_booking_correct=1.0,
            ),
            _entry(
                scenario_id="s_002",
                escalation_score=0.95,
                escalation_reasons=[
                    "emergency_intent_classified",
                    "low_confidence=0.80",
                ],
                judge_hallucination=0.0,
                judge_booking_correct=1.0,
                actual_outcome="escalated_emergency",
            ),
            _entry(
                scenario_id="s_003",
                escalation_score=0.65,
                escalation_reasons=[
                    "low_confidence=0.40",
                    "frustration=0.60",
                ],
                judge_hallucination=0.1,
                judge_booking_correct=0.8,
                judge_violations=["unsupported_claim"],
            ),
            _entry(
                scenario_id="s_004",
                escalation_score=0.30,
                escalation_reasons=["frustration=0.30"],
                judge_hallucination=0.0,
                judge_booking_correct=1.0,
            ),
        ],
    )
    return tmp_path


def _write_audit(base: Path, *, customer_id: str, lines: list[dict]) -> None:
    customer_dir = base / customer_id
    customer_dir.mkdir(parents=True, exist_ok=True)
    path = customer_dir / "audit.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


# ---------- build_sentinel_ops ----------


def test_build_sentinel_ops_empty_when_no_trace(tmp_path: Path) -> None:
    ops = data_sources.build_sentinel_ops("ghost", data_dir=tmp_path)
    assert ops.has_data is False
    assert ops.sample_count == 0
    assert ops.mean_escalation_score == 0.0
    assert ops.trust_score == 0.0
    assert ops.signals == []


def test_build_sentinel_ops_populates_from_trace(sentinel_data_dir: Path) -> None:
    ops = data_sources.build_sentinel_ops(
        "ophthalmology", data_dir=sentinel_data_dir
    )
    assert ops.has_data is True
    assert ops.sample_count == 4
    # mean = (0.15 + 0.95 + 0.65 + 0.30) / 4 = 0.5125
    assert ops.mean_escalation_score == pytest.approx(0.5125)
    assert ops.trust_score == pytest.approx(1.0 - 0.5125)
    assert ops.escalation_precision == pytest.approx(0.62)
    assert ops.escalation_recall == pytest.approx(1.0)
    assert ops.escalation_f1 > 0.0
    assert ops.emergencies_caught == 1


def test_signals_parse_reasons_and_apply_weights(
    sentinel_data_dir: Path,
) -> None:
    ops = data_sources.build_sentinel_ops(
        "ophthalmology", data_dir=sentinel_data_dir
    )
    by_name = {s.name: s for s in ops.signals}
    assert "Low confidence" in by_name
    assert "Frustration" in by_name
    # low_confidence values: 0.80, 0.40 -> mean 0.60. Weight 0.30 -> 0.18.
    lc = by_name["Low confidence"]
    assert lc.raw_mean == pytest.approx(0.60)
    assert lc.weight == pytest.approx(0.30)
    assert lc.contribution == pytest.approx(0.18)
    assert lc.fire_count == 2
    # frustration values: 0.60, 0.30 -> mean 0.45. Weight 0.20 -> 0.09.
    fr = by_name["Frustration"]
    assert fr.raw_mean == pytest.approx(0.45)
    assert fr.contribution == pytest.approx(0.09)
    # Signals with no values in the trace should be zeroed but present.
    rc = by_name["Rule conflict"]
    assert rc.raw_mean == 0.0
    assert rc.fire_count == 0


def test_judge_agreement_excludes_judgeless_entries(
    sentinel_data_dir: Path,
) -> None:
    ops = data_sources.build_sentinel_ops(
        "ophthalmology", data_dir=sentinel_data_dir
    )
    # All 4 entries carry judge_hallucination.
    assert ops.judge.sampled_turns == 4
    # 3 of 4 have hallucination < 0.05.
    assert ops.judge.no_hallucination_pct == pytest.approx(0.75)
    # booking mean = (1.0 + 1.0 + 0.8 + 1.0) / 4 = 0.95
    assert ops.judge.booking_correct_mean == pytest.approx(0.95)
    # one judge_violations entry
    assert ops.judge.violations_total == 1


def test_judge_agreement_zero_when_no_verdicts(tmp_path: Path) -> None:
    _write_report(tmp_path, customer_id="x", precision=0.5, recall=0.5)
    _write_trace(
        tmp_path,
        customer_id="x",
        entries=[
            _entry(scenario_id="s_001", escalation_score=0.1),
        ],
    )
    ops = data_sources.build_sentinel_ops("x", data_dir=tmp_path)
    assert ops.judge.sampled_turns == 0
    assert ops.judge.no_hallucination_pct == 0.0
    assert ops.judge.booking_correct_mean == 0.0


def test_audit_view_reads_redactions_and_tail(tmp_path: Path) -> None:
    _write_report(tmp_path, customer_id="x", precision=0.6, recall=0.9)
    _write_trace(
        tmp_path,
        customer_id="x",
        entries=[_entry(scenario_id="s_001", escalation_score=0.2)],
    )
    _write_audit(
        tmp_path,
        customer_id="x",
        lines=[
            {
                "timestamp": "2026-06-17T14:00:00Z",
                "conversation_id": "conv_001",
                "customer_id": "x",
                "user_message": "Hi I'm pat_001",
                "agent_reply": "Hello — let me check.",
                "redactions": {"<PATIENT_ID>": 1},
                "guardrail": "safe",
                "tool_calls": [],
                "steps": 1,
            },
            {
                "timestamp": "2026-06-17T14:05:00Z",
                "conversation_id": "conv_001",
                "customer_id": "x",
                "user_message": "Chest pain",
                "agent_reply": "Please call 911 immediately.",
                "redactions": {},
                "guardrail": "emergency_intent_classified",
                "tool_calls": [],
                "steps": 0,
            },
        ],
    )
    ops = data_sources.build_sentinel_ops("x", data_dir=tmp_path)
    assert ops.phi_redactions_total == 1
    # Tail is newest-first.
    assert len(ops.audit_tail) == 2
    assert ops.audit_tail[0].kind == "emergency"
    assert ops.audit_tail[1].kind == "chat"


def test_audit_view_handles_missing_log(sentinel_data_dir: Path) -> None:
    ops = data_sources.build_sentinel_ops(
        "ophthalmology", data_dir=sentinel_data_dir
    )
    # No audit.jsonl in fixture -> total 0, tail empty.
    assert ops.phi_redactions_total == 0
    assert ops.audit_tail == []


# ---------- view: build_html ----------


def _snapshot(*, has_data: bool = True) -> SentinelOpsSnapshot:
    return SentinelOpsSnapshot(
        tenant="Ophthalmology",
        has_data=has_data,
        sample_count=100 if has_data else 0,
        mean_escalation_score=0.30 if has_data else 0.0,
        trust_score=0.70 if has_data else 0.0,
        decision_threshold=0.5,
        signals=[
            SignalContribution(
                name="Low confidence", raw_mean=0.60, weight=0.30,
                contribution=0.18, fire_count=12,
            ),
            SignalContribution(
                name="Repeated clarification", raw_mean=0.20, weight=0.15,
                contribution=0.03, fire_count=4,
            ),
            SignalContribution(
                name="Rule conflict", raw_mean=0.00, weight=0.20,
                contribution=0.00, fire_count=0,
            ),
            SignalContribution(
                name="Frustration", raw_mean=0.50, weight=0.20,
                contribution=0.10, fire_count=8,
            ),
            SignalContribution(
                name="Unsupported request", raw_mean=0.30, weight=0.15,
                contribution=0.045, fire_count=5,
            ),
        ] if has_data else [],
        judge=JudgeAgreement(
            sampled_turns=42,
            no_hallucination_pct=0.95,
            booking_correct_mean=0.98,
            violations_total=2,
        ),
        phi_redactions_total=37,
        audit_tail=[
            AuditTailItem(ts="14:00:01", kind="chat", summary="Hi"),
            AuditTailItem(ts="14:05:02", kind="emergency", summary="Call 911"),
        ],
        emergencies_caught=3,
        escalation_precision=0.62,
        escalation_recall=1.0,
        escalation_f1=0.77,
    )


def test_sentinel_view_headline_carries_four_kpi_tiles() -> None:
    html = view.build_html(_snapshot())
    assert "clarion-kpi-strip" in html
    for label in ("TRUST SCORE", "SENTINEL SCORE", "ESC. PRECISION", "ESC. RECALL"):
        assert label in html


def test_sentinel_view_renders_trust_gauge_svg() -> None:
    html = view.build_html(_snapshot())
    assert "clarion-gauge" in html
    # Trust gauge always emits an SVG.
    assert "<svg" in html


def test_sentinel_view_renders_five_signal_rows() -> None:
    html = view.build_html(_snapshot())
    # signal_bar uses the .clarion-signal class.
    assert html.count('class="clarion-signal"') >= 5
    # Heavy contributor uses data-weight=heavy.
    assert 'data-weight="heavy"' in html  # Low confidence contribution = 0.18 -> heavy


def test_sentinel_view_signal_footnote_shows_raw_x_weight() -> None:
    html = view.build_html(_snapshot())
    # The footnote line is the explainer that makes the contribution math visible.
    assert "raw 0.60 x weight 0.30 = contribution 0.18" in html


def test_sentinel_view_renders_judge_chips() -> None:
    html = view.build_html(_snapshot())
    assert "SAMPLED" in html
    assert "NO HALLUC" in html
    assert "BOOKING" in html
    assert "VIOLATIONS" in html
    assert "42" in html
    assert "95.0%" in html


def test_sentinel_view_renders_emergencies_counter() -> None:
    html = view.build_html(_snapshot())
    assert "EMERGENCIES CAUGHT" in html
    # value 3 displayed
    assert ">3<" in html


def test_sentinel_view_audit_tail_uses_incident_rows() -> None:
    html = view.build_html(_snapshot())
    assert "clarion-incident" in html
    # Emergency event is rendered with critical severity.
    assert 'data-state="critical"' in html


def test_sentinel_view_empty_when_has_data_false() -> None:
    html = view.build_html(_snapshot(has_data=False))
    assert "No trace data" in html
    assert "Awaiting Data" in html


def test_sentinel_view_empty_html_renders_cli_hint() -> None:
    html = view.empty_html(tenant="Ophthalmology")
    assert "No trace data for Ophthalmology" in html
    assert "python -m clarion.evaluation.cli" in html
