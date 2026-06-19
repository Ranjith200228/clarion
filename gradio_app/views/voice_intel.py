"""Voice Intelligence — the third hero view.

Per the vision brief: *demonstrate how emotion affects escalation
decisions*. This view makes the causal link visible — emotions
classified across the corpus on the left, escalation rate +
prediction on the right, with the M5 voice pipeline as a static
reference at the bottom.

Layout top-to-bottom:

1. **Headline strip** — 4 tiles: TOTAL TURNS, MEAN FRUSTRATION,
   ESCALATION RATE, PREDICTED ESCALATION (Bayesian-smoothed).
2. **Emotion distribution** — six bars sized by per-emotion
   fraction. Each bar is colour-banded so a viewer reads the
   intensity at a glance.
3. **Frustration trace** — an inline-SVG line chart of the
   per-turn escalation score across the corpus. A dashed
   horizontal hairline marks the 0.50 decision threshold so
   escalated turns are visually obvious.
4. **Voice pipeline** — three stages (STT / Agent / TTS) with
   their target latencies, styled like the M5 round-trip
   diagram (no live mic data needed — it's a reference budget
   chart).
5. **Sample transcript** — token-by-token shading where each
   token's opacity is proportional to its STT confidence. The
   panel header labels this as an illustration, not a recorded
   turn (we don't persist live STT output).
"""

from __future__ import annotations

from gradio_app import components as c
from gradio_app.data_sources import (
    EmotionTotal,
    FrustrationPoint,
    VoiceIntelligenceSnapshot,
    VoicePipelineStage,
)
from gradio_app.theme import PALETTE

# ---------- public ----------


def build_html(vi: VoiceIntelligenceSnapshot) -> str:
    if not vi.has_data:
        return _empty_with_pipeline(vi)
    return (
        '<div class="clarion-stack" style="gap: 24px;">'
        + _section_title(
            title="Voice Intelligence",
            subtitle=(
                f"Emotion -> escalation signal for {_esc(vi.tenant)} "
                f"({vi.total_turns} turns)"
            ),
        )
        + _headline_strip(vi)
        + '<div class="clarion-row" style="align-items: stretch; gap: 16px;">'
        + _emotion_panel(vi.emotions)
        + _frustration_panel(vi.frustration_trace)
        + "</div>"
        + '<div class="clarion-row" style="align-items: stretch; gap: 16px;">'
        + _pipeline_panel(vi.voice_pipeline)
        + _transcript_panel(vi.sample_transcript)
        + "</div>"
        + "</div>"
    )


def _empty_with_pipeline(vi: VoiceIntelligenceSnapshot) -> str:
    """Even on a fresh deploy with zero scored turns, ship the
    voice pipeline + sample transcript panels — they are static
    and tell the rest of the story without needing data."""
    body = (
        '<div class="clarion-stack" style="align-items: center; gap: 12px;">'
        '<div style="font-size: var(--fs-lg); color: var(--c-text);">'
        f"No turns yet for {_esc(vi.tenant)}."
        "</div>"
        '<div style="font-size: var(--fs-sm); color: var(--c-text-muted);">'
        "Run the harness to populate the emotion + frustration analytics:"
        "</div>"
        '<div class="clarion-mono" style="padding: 12px 16px; '
        'background: var(--c-bg-subtle); border-radius: var(--r-md);">'
        f"python -m clarion.evaluation.cli run {_esc(vi.tenant).lower()}"
        "</div>"
        "</div>"
    )
    return (
        '<div class="clarion-stack" style="gap: 24px;">'
        + _section_title(
            title="Voice Intelligence",
            subtitle="Emotion analytics, frustration trace, escalation prediction",
        )
        + c.panel(title="Awaiting Data", body_html=body)
        + '<div class="clarion-row" style="align-items: stretch; gap: 16px;">'
        + _pipeline_panel(vi.voice_pipeline)
        + _transcript_panel(vi.sample_transcript)
        + "</div>"
        + "</div>"
    )


# ---------- headline strip ----------


def _headline_strip(vi: VoiceIntelligenceSnapshot) -> str:
    return c.kpi_strip(
        [
            c.kpi_tile(
                label="TURNS SAMPLED",
                value=str(vi.total_turns),
                status="info",
            ),
            c.kpi_tile(
                label="MEAN FRUSTRATION",
                value=f"{vi.mean_frustration:.2f}",
                # Lower is better.
                status=_band_inverse(vi.mean_frustration, healthy=0.30, warning=0.50),
            ),
            c.kpi_tile(
                label="ESCALATION RATE",
                value=_pct(vi.escalation_rate),
                status=_band_inverse(vi.escalation_rate, healthy=0.10, warning=0.30),
            ),
            c.kpi_tile(
                label="PREDICTED NEXT TURN",
                value=_pct(vi.predicted_escalation_rate),
                status=_band_inverse(
                    vi.predicted_escalation_rate, healthy=0.10, warning=0.30
                ),
            ),
        ]
    )


# ---------- emotion panel ----------


# Pairing between emotion name and the brand palette key used to
# colour its bar. Calm + anxious read as healthy/warning (lower
# severity); frustrated + urgent + distressed read as worse.
_EMOTION_COLOR: dict[str, str] = {
    "calm":       PALETTE["healthy"],
    "anxious":    PALETTE["info"],
    "confused":   PALETTE["accent"],
    "frustrated": PALETTE["warning"],
    "urgent":     PALETTE["warning"],
    "distressed": PALETTE["critical"],
}


def _emotion_panel(emotions: list[EmotionTotal]) -> str:
    if all(e.count == 0 for e in emotions):
        body = _empty_message("No emotion samples in this corpus yet.")
    else:
        rows = "".join(_emotion_row(e) for e in emotions)
        body = '<div class="clarion-stack" style="gap: 8px;">' + rows + "</div>"
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title="Emotion Distribution", body_html=body)
        + "</div>"
    )


def _emotion_row(emo: EmotionTotal) -> str:
    """One horizontal bar per emotion. Bar uses the palette colour
    for that emotion + a tabular count + percentage at the right."""
    pct = emo.fraction * 100.0
    fill_color = _EMOTION_COLOR.get(emo.emotion, PALETTE["accent"])
    return (
        '<div class="clarion-signal">'
        f'<div class="clarion-signal-name">{_esc(emo.emotion.title())}</div>'
        '<div class="clarion-signal-track">'
        f'<div class="clarion-signal-fill" style="width: {pct:.1f}%; '
        f'background: {fill_color};"></div>'
        "</div>"
        f'<div class="clarion-signal-value">{emo.count} ({pct:.0f}%)</div>'
        "</div>"
    )


# ---------- frustration trace ----------


def _frustration_panel(trace: list[FrustrationPoint]) -> str:
    if not trace:
        body = _empty_message("No frustration samples in this corpus yet.")
    else:
        chart = _frustration_chart(trace)
        legend = (
            '<div class="clarion-row" style="gap: 12px; flex-wrap: wrap; '
            'margin-top: 8px;">'
            + c.status_badge("healthy", label="below threshold")
            + c.status_badge("critical", label="escalated")
            + '<span style="font-size: var(--fs-xs); color: var(--c-text-muted);">'
            "dashed line = 0.50 decision boundary"
            "</span>"
            "</div>"
        )
        body = (
            '<div class="clarion-stack" style="gap: 4px;">'
            + chart
            + legend
            + "</div>"
        )
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title="Frustration Trace", body_html=body)
        + "</div>"
    )


def _frustration_chart(trace: list[FrustrationPoint]) -> str:
    """Inline SVG line chart of per-turn escalation score.

    X axis: turn index (no axis labels — we want the picture to be
    a glanceable mood rather than a measurement instrument).
    Y axis: 0 at the top of the chart, 1 at the bottom (so a
    "spike" goes UP visually).
    """
    if not trace:
        return ""
    width, height = 480, 120
    pad_x, pad_y = 16, 12
    inner_w = width - 2 * pad_x
    inner_h = height - 2 * pad_y
    n = len(trace)
    # Avoid div-by-zero on a single-point series.
    step = inner_w / max(1, n - 1)

    def x_for(idx: int) -> float:
        return pad_x + idx * step

    def y_for(score: float) -> float:
        # Score 0 at bottom, 1 at top -> invert relative to SVG y.
        clamped = max(0.0, min(1.0, score))
        return pad_y + (1.0 - clamped) * inner_h

    points = " ".join(
        f"{x_for(i):.1f},{y_for(p.score):.1f}" for i, p in enumerate(trace)
    )
    # Threshold hairline at score=0.50.
    threshold_y = y_for(0.5)
    # Markers at each escalated turn so they pop visually.
    markers = "".join(
        (
            f'<circle cx="{x_for(i):.1f}" cy="{y_for(p.score):.1f}" '
            f'r="3" fill="{PALETTE["critical"]}"/>'
        )
        for i, p in enumerate(trace)
        if p.escalated
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        'preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" '
        'aria-hidden="true">'
        # Threshold hairline.
        f'<line x1="{pad_x}" y1="{threshold_y:.1f}" x2="{width - pad_x}" '
        f'y2="{threshold_y:.1f}" stroke="{PALETTE["text_muted"]}" '
        'stroke-width="1" stroke-dasharray="4 4"/>'
        # Polyline.
        f'<polyline points="{points}" fill="none" stroke="{PALETTE["accent"]}" '
        'stroke-width="2" stroke-linejoin="round"/>'
        f"{markers}"
        "</svg>"
    )


# ---------- voice pipeline ----------


def _pipeline_panel(stages: list[VoicePipelineStage]) -> str:
    rows = "".join(_pipeline_row(s) for s in stages)
    body = (
        '<div class="clarion-stack" style="gap: 8px;">'
        + rows
        + '<div style="font-size: var(--fs-xs); color: var(--c-text-muted); '
        'margin-top: 8px;">'
        "Targets from the M5 Voice Layer plan. Live mic turns flow "
        "through this pipeline; static analytics above derive from "
        "the chat trace as a proxy."
        "</div>"
        "</div>"
    )
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title="Voice Round-Trip Targets", body_html=body)
        + "</div>"
    )


def _pipeline_row(stage: VoicePipelineStage) -> str:
    """One stage row — name + target ms badge + description."""
    return (
        '<div class="clarion-kpi-tile" data-status="info" '
        'style="min-width: unset; height: auto; padding: 12px 16px;">'
        '<div class="clarion-row" style="justify-content: space-between;">'
        f'<div style="font-size: var(--fs-md); font-weight: var(--fw-semibold); '
        f'color: var(--c-text-strong);">{_esc(stage.name)}</div>'
        f'<span class="clarion-cost-chip"><span style="color: var(--c-mono-fg);">'
        f"{stage.target_ms} ms</span>"
        '<span class="clarion-cost-chip-period">target</span></span>'
        "</div>"
        f'<div style="font-size: var(--fs-xs); color: var(--c-text-muted); '
        f'margin-top: 4px;">{_esc(stage.description)}</div>'
        "</div>"
    )


# ---------- transcript panel ----------


def _transcript_panel(tokens: list[tuple[str, float]]) -> str:
    """Sample transcript with per-token confidence shading.

    Confidence drives the rendered opacity (high confidence = full
    colour, low confidence = greyer) so a viewer can see at a glance
    which tokens an STT engine would flag for review. The panel
    header labels this as an illustration so nobody mistakes the
    static text for a recorded turn.
    """
    if not tokens:
        body = _empty_message("No transcript sample on file.")
    else:
        spans = " ".join(_transcript_token(tok, conf) for tok, conf in tokens)
        body = (
            '<div style="font-size: var(--fs-md); line-height: 1.8; '
            'color: var(--c-text-strong);">'
            + spans
            + "</div>"
            '<div style="font-size: var(--fs-xs); color: var(--c-text-muted); '
            'margin-top: 12px;">'
            "Illustrative transcript. Shading proportional to per-token "
            "confidence — dimmer = lower confidence the STT engine would "
            "surface for review."
            "</div>"
        )
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title="Live Transcript (sample)", body_html=body)
        + "</div>"
    )


def _transcript_token(token: str, confidence: float) -> str:
    """One shaded token. Below 0.80 confidence we underline as well
    so the lower-confidence words are obvious without colour."""
    clamped = max(0.0, min(1.0, confidence))
    # Opacity range: 0.45 .. 1.0 so even very-low-conf tokens
    # remain legible.
    opacity = 0.45 + clamped * 0.55
    decoration = "underline" if clamped < 0.80 else "none"
    return (
        f'<span style="opacity: {opacity:.2f}; text-decoration: {decoration}; '
        f'text-decoration-color: var(--c-warning);">{_esc(token)}</span>'
    )


# ---------- helpers ----------


def _band_inverse(value: float, *, healthy: float, warning: float) -> c.Status:
    if value <= healthy:
        return "healthy"
    if value <= warning:
        return "warning"
    return "critical"


def _pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _section_title(*, title: str, subtitle: str) -> str:
    return (
        '<div class="clarion-stack" style="gap: 4px;">'
        f'<div style="font-size: var(--fs-2xl); font-weight: var(--fw-bold); '
        f'color: var(--c-text-strong); line-height: 1.1;">{title}</div>'
        f'<div style="font-size: var(--fs-sm); color: var(--c-text-muted);">{subtitle}</div>'
        "</div>"
    )


def _empty_message(text: str) -> str:
    return (
        f'<div style="padding: 16px; color: var(--c-text-muted); '
        f'font-size: var(--fs-sm);">{_esc(text)}</div>'
    )


def _esc(text: str) -> str:
    import html as _html

    return _html.escape(text, quote=True)


__all__ = ["build_html"]
