"""Sentinel Operations Center — the primary hero view.

Per the Phase 0 plan: *the trust engine is the strongest piece of
engineering in the repo and currently has zero UI*. This view
closes that gap.

Four panels:

1. **Headline strip** — Trust Score (the single 0..1 number), the
   mean escalation score it inverts, the decision threshold, and
   the scoreboard tile (precision / recall / F1 against the
   scenario corpus).
2. **Trust Gauge** — the load-bearing visual. The
   ``components.trust_gauge`` semicircle interprets the value
   it's given as an escalation score (higher = more risky); the
   panel labels it "Sentinel Score" so the legend matches the
   gauge's colour bands without confusing the reader.
3. **Signal Breakdown** — five rows, one per
   :class:`EscalationSignals` field. Each row carries the raw mean
   value (bar length), the weight (label), and the contribution
   (raw x weight) the signal made to the composite. The
   ``components.signal_bar`` weight bucket colours the fill so a
   "heavy contributor" pops red.
4. **Judge + Safety + Audit** — three sub-panels:
   * Judge Confidence — sampled turns, no-hallucination %, mean
     booking-correctness, total policy violations.
   * Safety — total emergencies caught for this tenant.
   * Audit Tail — last 10 audit-log events (already
     PHI-redacted at write time) + total redactions across the
     log.

Same rules as every Phase B+ view: typed rollup in, HTML string
out, no business logic.
"""

from __future__ import annotations

from gradio_app import components as c
from gradio_app.data_sources import (
    AuditTailItem,
    JudgeAgreement,
    SentinelOpsSnapshot,
    SignalContribution,
)

# ---------- public ----------


def build_html(ops: SentinelOpsSnapshot) -> str:
    if not ops.has_data:
        return empty_html(tenant=ops.tenant)

    return (
        '<div class="clarion-stack" style="gap: 24px;">'
        + c.page_intro(
            title="Sentinel Operations",
            what=(
                "Per-trace verdicts from the trust engine — "
                "what passed, what was caught, why."
            ),
            quote="Every decision shows its work.",
        )
        + _headline_strip(ops)
        + '<div class="clarion-row" style="align-items: stretch; gap: 16px;">'
        + _gauge_panel(ops)
        + _signal_panel(ops.signals)
        + "</div>"
        + '<div class="clarion-row" style="align-items: stretch; gap: 16px;">'
        + _judge_panel(ops.judge, ops.emergencies_caught)
        + _audit_panel(ops.phi_redactions_total, ops.audit_tail)
        + "</div>"
        + "</div>"
    )


def empty_html(*, tenant: str = "this customer") -> str:
    body = (
        '<div class="clarion-stack" style="align-items: center; gap: 12px;">'
        '<div style="font-size: var(--fs-lg); color: var(--c-text);">'
        f"No trace data for {_esc(tenant)} on disk."
        "</div>"
        '<div style="font-size: var(--fs-sm); color: var(--c-text-muted);">'
        "Sentinel rollups draw from the per-tenant TraceReport + audit log."
        "</div>"
        '<div class="clarion-mono" style="padding: 12px 16px; '
        'background: var(--c-bg-subtle); border-radius: var(--r-md);">'
        f"python -m clarion.evaluation.cli run {_esc(tenant).lower()}"
        "</div>"
        "</div>"
    )
    return (
        '<div class="clarion-stack" style="gap: 24px;">'
        + c.page_intro(
            title="Sentinel Operations",
            what=(
                "Per-trace verdicts from the trust engine — "
                "what passed, what was caught, why."
            ),
            quote="Every decision shows its work.",
        )
        + c.panel(title="Awaiting Data", body_html=body)
        + "</div>"
    )


# ---------- panels ----------


def _headline_strip(ops: SentinelOpsSnapshot) -> str:
    """Four KPI tiles. Trust is the marquee value — first tile, big."""
    tiles = [
        c.kpi_tile(
            label="TRUST SCORE",
            value=f"{ops.trust_score:.2f}",
            status=_band_trust(ops.trust_score),
        ),
        c.kpi_tile(
            label="SENTINEL SCORE",
            value=f"{ops.mean_escalation_score:.2f}",
            # Higher = worse, so flip the band test.
            status=_band_inverse(ops.mean_escalation_score, healthy=0.20, warning=0.50),
        ),
        c.kpi_tile(
            label="ESC. PRECISION",
            value=_pct(ops.escalation_precision),
            status=_band(ops.escalation_precision, healthy=0.80, warning=0.60),
        ),
        c.kpi_tile(
            label="ESC. RECALL",
            value=_pct(ops.escalation_recall),
            status=_band(ops.escalation_recall, healthy=0.90, warning=0.70),
        ),
    ]
    return c.kpi_strip(tiles)


def _gauge_panel(ops: SentinelOpsSnapshot) -> str:
    """Big inline-SVG gauge + the literal threshold value for trust."""
    gauge_html = c.trust_gauge(
        score=ops.mean_escalation_score,
        threshold=ops.decision_threshold,
        label="SENTINEL",
    )
    legend = (
        '<div class="clarion-stack" style="gap: 8px;">'
        '<div class="clarion-row" style="gap: 8px; flex-wrap: wrap;">'
        + c.status_badge("healthy", label="below threshold/2")
        + c.status_badge("info", label="below threshold")
        + c.status_badge("warning", label="approaching")
        + c.status_badge("critical", label="escalation")
        + "</div>"
        '<div style="font-size: var(--fs-xs); color: var(--c-text-muted); '
        'line-height: 1.6;">'
        f"Threshold {ops.decision_threshold:.2f}. "
        "Mean escalation score across the last "
        f"{ops.sample_count} turns. Lower is healthier — the gauge"
        " shows the same value the agent uses when deciding to "
        "escalate."
        "</div>"
        "</div>"
    )
    body = (
        '<div class="clarion-stack" style="align-items: center; gap: 16px;">'
        + gauge_html
        + legend
        + "</div>"
    )
    return (
        '<div style="flex: 0 0 320px;">'
        + c.panel(title="Composite Trust", body_html=body)
        + "</div>"
    )


def _signal_panel(signals: list[SignalContribution]) -> str:
    if not signals:
        body = _empty_message("No signal contributions on file.")
    else:
        rows = "".join(_signal_row(s) for s in signals)
        body = (
            '<div class="clarion-stack" style="gap: 4px;">'
            + rows
            + "</div>"
        )
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title="Signal Breakdown", body_html=body)
        + "</div>"
    )


def _signal_row(signal: SignalContribution) -> str:
    """One weighted-signal row with the bar + footnote line below."""
    weight_bucket = _signal_weight_bucket(signal.contribution)
    bar = c.signal_bar(
        name=signal.name, value=signal.raw_mean, weight=weight_bucket
    )
    footnote = (
        f'<div style="margin-left: 152px; margin-top: -4px; '
        f'font-size: var(--fs-xs); color: var(--c-text-muted); '
        f'font-family: var(--font-mono); font-variant-numeric: tabular-nums;">'
        f"raw {signal.raw_mean:.2f} x weight {signal.weight:.2f} = "
        f"contribution {signal.contribution:.2f}"
        f"  ·  fired in {signal.fire_count} turn{'s' if signal.fire_count != 1 else ''}"
        "</div>"
    )
    return (
        '<div class="clarion-stack" style="gap: 2px;">'
        + bar
        + footnote
        + "</div>"
    )


def _judge_panel(judge: JudgeAgreement, emergencies: int) -> str:
    """Judge KPIs + the safety counter side by side."""
    if judge.sampled_turns == 0:
        judge_body = _empty_message("No judge verdicts in this run.")
    else:
        chips = (
            '<div class="clarion-row" style="gap: 16px; flex-wrap: wrap;">'
            + _mini("SAMPLED", str(judge.sampled_turns))
            + _mini("NO HALLUC", _pct(judge.no_hallucination_pct))
            + _mini("BOOKING ✓", _pct(judge.booking_correct_mean))
            + _mini("VIOLATIONS", str(judge.violations_total))
            + "</div>"
        )
        judge_body = chips

    emergency_body = (
        '<div class="clarion-stack" style="gap: 4px;">'
        + _mini("EMERGENCIES CAUGHT", str(emergencies))
        + '<div style="font-size: var(--fs-xs); color: var(--c-text-muted);">'
        "Guardrail-fired or judged emergency outcomes across all sampled turns."
        "</div>"
        "</div>"
    )

    inner = (
        '<div class="clarion-stack" style="gap: 16px;">'
        + judge_body
        + '<div style="height: 1px; background: var(--c-border);"></div>'
        + emergency_body
        + "</div>"
    )

    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title="LLM Judge + Safety", body_html=inner)
        + "</div>"
    )


def _audit_panel(redactions_total: int, tail: list[AuditTailItem]) -> str:
    """Total PHI redactions + the last 10 audit-log rows."""
    redactions_chip = (
        '<div class="clarion-row" style="justify-content: space-between; '
        'align-items: center;">'
        + _mini("PHI REDACTIONS", f"{redactions_total:,}")
        + '<div style="font-size: var(--fs-xs); color: var(--c-text-muted);">'
        "Counted by the PHI redactor on every audit-log write."
        "</div>"
        "</div>"
    )

    if not tail:
        tail_html = _empty_message(
            "Audit log empty — no chat turns recorded for this tenant yet."
        )
    else:
        rows = "".join(_audit_row(item) for item in tail)
        tail_html = (
            '<div class="clarion-stack" style="gap: 0;">' + rows + "</div>"
        )

    inner = (
        '<div class="clarion-stack" style="gap: 12px;">'
        + redactions_chip
        + '<div style="height: 1px; background: var(--c-border);"></div>'
        + tail_html
        + "</div>"
    )
    return (
        '<div style="flex: 1 1 0; min-width: 0;">'
        + c.panel(title="Audit Trail", body_html=inner)
        + "</div>"
    )


def _audit_row(item: AuditTailItem) -> str:
    """One audit-tail row — visually mirrors the incident_row layout
    so the audit panel reads like the Mission Control streams."""
    sev = _audit_severity(item.kind)
    return c.incident_row(
        ts=item.ts,
        severity=sev,
        tenant=item.kind,
        summary=item.summary,
    )


# ---------- helpers ----------


def _signal_weight_bucket(contribution: float) -> c.SignalWeight:
    """Map the contribution to one of the three colour buckets used
    by ``components.signal_bar``. The thresholds line up with what a
    human eye reads as "this one's a problem":

      * contribution >= 0.10  → heavy (red)
      * contribution >= 0.05  → medium (amber)
      * else                  → light (cyan)
    """
    if contribution >= 0.10:
        return "heavy"
    if contribution >= 0.05:
        return "medium"
    return "light"


def _audit_severity(kind: str) -> c.Status:
    if kind.startswith("emergency"):
        return "critical"
    if kind.startswith("clinical"):
        return "warning"
    return "info"


def _band(value: float, *, healthy: float, warning: float) -> c.Status:
    if value >= healthy:
        return "healthy"
    if value >= warning:
        return "warning"
    return "critical"


def _band_inverse(value: float, *, healthy: float, warning: float) -> c.Status:
    if value <= healthy:
        return "healthy"
    if value <= warning:
        return "warning"
    return "critical"


def _band_trust(trust: float) -> c.Status:
    """Trust = 1 - mean escalation. Bands inverted from the score."""
    if trust >= 0.80:
        return "healthy"
    if trust >= 0.50:
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


def _mini(label: str, value: str) -> str:
    return (
        '<div class="clarion-stack" style="gap: 2px;">'
        f'<div class="clarion-kpi-label">{_esc(label)}</div>'
        f'<div style="font-size: var(--fs-lg); font-weight: var(--fw-semibold); '
        f'color: var(--c-text-strong); font-variant-numeric: tabular-nums;">'
        f"{_esc(value)}"
        "</div>"
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


__all__ = ["build_html", "empty_html"]
