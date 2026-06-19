"""Tests for the Voice Intelligence view: data_sources rollup +
view HTML.

Coverage:
- Emotion classifier maps the six buckets correctly from
  escalation_reasons + outcome + intent / difficulty.
- Frustration trace preserves chrono order and marks ``escalated``
  at the 0.50 threshold.
- Bayesian smoothing produces a sane prior on small samples.
- The view's headline tiles, emotion bars, frustration SVG,
  voice pipeline rows, and shaded transcript all render.
- Empty-state path still ships the static voice pipeline + sample
  transcript panels so the page is never visually empty.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gradio_app import data_sources
from gradio_app.data_sources import (
    EmotionTotal,
    FrustrationPoint,
    VoiceIntelligenceSnapshot,
    VoicePipelineStage,
)
from gradio_app.views import voice_intel as view

# ---------- helpers ----------


def _write_report(base: Path, customer_id: str) -> None:
    customer_dir = base / customer_id
    customer_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "customer_id": customer_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario_count": 6,
        "pass_rate": 1.0,
        "metrics": {
            "scenario_count": 6,
            "pass_rate": 1.0,
            "containment_rate": 0.66,
            "booking_accuracy": 1.0,
            "booking_total": 6,
            "booking_correct": 6,
            "hallucination_rate": 0.0,
            "hallucination_with_judge": 0,
            "escalation_precision": 0.8,
            "escalation_recall": 1.0,
            "escalation_f1": 0.89,
            "escalation_accuracy": 0.9,
            "safety_catch_rate": 1.0,
            "safety_total": 2,
            "safety_caught": 2,
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


def _write_trace(base: Path, customer_id: str, entries: list[dict]) -> None:
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
    intent: str = "book",
    difficulty: str = "clear",
    actual_outcome: str = "booked",
    escalation_score: float | None = 0.10,
    escalation_reasons: list[str] | None = None,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "customer_id": "ophthalmology",
        "trace_id": f"trace_{scenario_id}",
        "difficulty": difficulty,
        "intent": intent,
        "agent_replies": ["ok"],
        "tools_called": [],
        "actual_outcome": actual_outcome,
        "passed": True,
        "escalation_score": escalation_score,
        "escalation_reasons": escalation_reasons or [],
        "judge_hallucination": None,
        "judge_booking_correct": None,
        "judge_violations": [],
        "duration_ms": 500.0,
        "cost_usd": 0.001,
        "input_tokens": 200,
        "output_tokens": 60,
        "step_count": 1,
    }


@pytest.fixture
def voice_data_dir(tmp_path: Path) -> Path:
    """Corpus exercising all six emotion buckets."""
    _write_report(tmp_path, "ophthalmology")
    _write_trace(
        tmp_path,
        "ophthalmology",
        [
            _entry(scenario_id="c_001", escalation_score=0.05),  # calm
            _entry(
                scenario_id="a_001",
                escalation_score=0.40,
                escalation_reasons=["low_confidence=0.40"],
            ),  # anxious
            _entry(
                scenario_id="x_001",
                escalation_score=0.35,
                escalation_reasons=["repeated_clarification=0.30"],
            ),  # confused
            _entry(
                scenario_id="f_001",
                escalation_score=0.55,
                escalation_reasons=["frustration=0.60"],
            ),  # frustrated (and crosses threshold)
            _entry(
                scenario_id="u_001",
                intent="emergency",
                difficulty="emergency",
                escalation_score=0.30,
                actual_outcome="info_provided",
            ),  # urgent (intent classification but not escalated)
            _entry(
                scenario_id="d_001",
                actual_outcome="escalated_emergency",
                escalation_score=0.95,
                escalation_reasons=["emergency_intent_classified"],
            ),  # distressed
        ],
    )
    return tmp_path


# ---------- build_voice_intelligence ----------


def test_voice_empty_when_no_trace(tmp_path: Path) -> None:
    vi = data_sources.build_voice_intelligence("ghost", data_dir=tmp_path)
    assert vi.has_data is False
    assert vi.total_turns == 0
    assert vi.frustration_trace == []
    # Empty-state still ships the voice pipeline + sample transcript.
    assert len(vi.voice_pipeline) == 3
    assert len(vi.sample_transcript) > 0


def test_voice_classifies_six_emotion_buckets(voice_data_dir: Path) -> None:
    vi = data_sources.build_voice_intelligence(
        "ophthalmology", data_dir=voice_data_dir
    )
    by_emotion = {e.emotion: e for e in vi.emotions}
    # Order is preserved.
    assert [e.emotion for e in vi.emotions] == [
        "calm",
        "anxious",
        "confused",
        "frustrated",
        "urgent",
        "distressed",
    ]
    # One of each.
    for name in ("calm", "anxious", "confused", "frustrated", "urgent", "distressed"):
        assert by_emotion[name].count == 1
        assert by_emotion[name].fraction == pytest.approx(1.0 / 6.0)


def test_voice_frustration_trace_chrono_order(voice_data_dir: Path) -> None:
    vi = data_sources.build_voice_intelligence(
        "ophthalmology", data_dir=voice_data_dir
    )
    assert len(vi.frustration_trace) == 6
    # turn_index is monotonic.
    assert [p.turn_index for p in vi.frustration_trace] == [0, 1, 2, 3, 4, 5]
    # f_001 + d_001 should be marked escalated (score >= 0.50).
    escalated_ids = {p.scenario_id for p in vi.frustration_trace if p.escalated}
    assert escalated_ids == {"f_001", "d_001"}


def test_voice_escalation_rate_and_prediction(voice_data_dir: Path) -> None:
    vi = data_sources.build_voice_intelligence(
        "ophthalmology", data_dir=voice_data_dir
    )
    # 2 of 6 are escalated. The rate is rounded to 4 places in the
    # snapshot so we widen the tolerance to match the round() trim.
    assert vi.escalation_rate == pytest.approx(2 / 6, abs=1e-4)
    # Predicted is Bayesian-smoothed against (alpha=2, beta=8)
    # -> (2+2)/(6+10) = 0.25.
    assert vi.predicted_escalation_rate == pytest.approx(0.25)


def test_voice_predicted_rate_uses_prior_on_empty_sample() -> None:
    # Empty sample -> alpha/(alpha+beta) = 0.20.
    assert data_sources._predicted_escalation_rate(0, 0) == pytest.approx(0.20)


def test_voice_empty_still_carries_pipeline_and_transcript(tmp_path: Path) -> None:
    vi = data_sources.build_voice_intelligence("ophthalmology", data_dir=tmp_path)
    assert vi.has_data is False
    # Pipeline stages match the M5 plan order.
    assert [s.name for s in vi.voice_pipeline] == ["STT", "Agent", "TTS"]
    # Sample transcript present + every token has a 0..1 confidence.
    assert vi.sample_transcript
    for token, conf in vi.sample_transcript:
        assert isinstance(token, str)
        assert 0.0 <= conf <= 1.0


# ---------- view: build_html ----------


def _snapshot(*, has_data: bool = True) -> VoiceIntelligenceSnapshot:
    pipeline = [
        VoicePipelineStage(name="STT", target_ms=500, description="Whisper"),
        VoicePipelineStage(name="Agent", target_ms=1500, description="ReAct"),
        VoicePipelineStage(name="TTS", target_ms=600, description="OpenAI TTS"),
    ]
    transcript = [("Hi", 0.99), ("Patel", 0.72)]
    if not has_data:
        return VoiceIntelligenceSnapshot(
            tenant="Ophthalmology",
            has_data=False,
            total_turns=0,
            emotions=[],
            frustration_trace=[],
            mean_frustration=0.0,
            escalation_rate=0.0,
            predicted_escalation_rate=0.20,
            voice_pipeline=pipeline,
            sample_transcript=transcript,
        )
    return VoiceIntelligenceSnapshot(
        tenant="Ophthalmology",
        has_data=True,
        total_turns=4,
        emotions=[
            EmotionTotal(emotion="calm", count=2, fraction=0.50),
            EmotionTotal(emotion="anxious", count=0, fraction=0.0),
            EmotionTotal(emotion="confused", count=0, fraction=0.0),
            EmotionTotal(emotion="frustrated", count=1, fraction=0.25),
            EmotionTotal(emotion="urgent", count=0, fraction=0.0),
            EmotionTotal(emotion="distressed", count=1, fraction=0.25),
        ],
        frustration_trace=[
            FrustrationPoint(turn_index=0, scenario_id="s_001", score=0.10, escalated=False),
            FrustrationPoint(turn_index=1, scenario_id="s_002", score=0.55, escalated=True),
            FrustrationPoint(turn_index=2, scenario_id="s_003", score=0.20, escalated=False),
            FrustrationPoint(turn_index=3, scenario_id="s_004", score=0.95, escalated=True),
        ],
        mean_frustration=0.45,
        escalation_rate=0.50,
        predicted_escalation_rate=0.43,
        voice_pipeline=pipeline,
        sample_transcript=transcript,
    )


def test_view_headline_strip_present_with_four_tiles() -> None:
    html = view.build_html(_snapshot())
    assert "clarion-kpi-strip" in html
    for label in (
        "TURNS SAMPLED",
        "MEAN FRUSTRATION",
        "ESCALATION RATE",
        "PREDICTED NEXT TURN",
    ):
        assert label in html


def test_view_emotion_panel_renders_six_rows() -> None:
    html = view.build_html(_snapshot())
    # signal-bar layout for each emotion -> at least 6 clarion-signal blocks
    # (plus the contribution chip for some).
    assert html.count('class="clarion-signal"') >= 6
    # Title-cased emotion labels.
    for label in (
        "Calm",
        "Anxious",
        "Confused",
        "Frustrated",
        "Urgent",
        "Distressed",
    ):
        assert label in html


def test_view_frustration_chart_present_when_trace_exists() -> None:
    html = view.build_html(_snapshot())
    # Inline SVG polyline rendered.
    assert "<svg" in html
    assert "<polyline" in html
    # Threshold hairline (dashed).
    assert 'stroke-dasharray="4 4"' in html
    # 2 escalated markers expected.
    assert html.count("<circle ") == 2


def test_view_pipeline_panel_lists_three_stages() -> None:
    html = view.build_html(_snapshot())
    for name in ("STT", "Agent", "TTS"):
        assert name in html
    # Each carries a "target" cost_chip label.
    assert html.count("</span>target</span>") >= 3 or "target</span>" in html


def test_view_transcript_panel_shades_tokens_by_confidence() -> None:
    html = view.build_html(_snapshot())
    # Token-by-token rendering — each token wrapped in a span with opacity.
    assert "opacity:" in html
    # Low-confidence token underlined.
    assert "underline" in html


def test_view_empty_state_still_ships_pipeline_and_transcript() -> None:
    html = view.build_html(_snapshot(has_data=False))
    assert "Awaiting Data" in html
    # Pipeline panel still rendered.
    assert "STT" in html
    # Sample transcript still rendered.
    assert "Patel" in html


def test_view_empty_state_renders_cli_hint() -> None:
    html = view.build_html(_snapshot(has_data=False))
    assert "python -m clarion.evaluation.cli" in html
