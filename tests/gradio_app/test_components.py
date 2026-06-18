"""Snapshot tests for ``gradio_app.components``.

Every render helper has at least one test asserting:

- output is a non-empty string
- output references the right ``clarion-*`` class
- HTML escaping fires on user-supplied text (no XSS surface)
- key data attributes (``data-status``, ``data-state``,
  ``data-trend``, ``data-weight``) carry the right value
- numeric clamping holds at edges (gauge with score < 0 or > 1,
  latency over budget)

The tests deliberately do NOT pin the full byte-exact string —
that would force us to rewrite assertions every time we adjust a
class name. We assert the load-bearing structural facts and let
the rest be cosmetic.
"""

from __future__ import annotations

from gradio_app import components as c

# ---------- kpi_tile ----------


def test_kpi_tile_renders_label_value_and_status() -> None:
    out = c.kpi_tile(label="PASS RATE", value="100%", status="healthy")
    assert 'class="clarion-kpi-tile"' in out
    assert 'data-status="healthy"' in out
    assert "PASS RATE" in out
    assert "100%" in out


def test_kpi_tile_delta_appears_only_when_provided() -> None:
    without = c.kpi_tile(label="X", value="1", status="info")
    assert "clarion-kpi-delta" not in without
    with_delta = c.kpi_tile(label="X", value="1", delta="+2.1%", trend="up", status="info")
    assert "clarion-kpi-delta" in with_delta
    assert 'data-trend="up"' in with_delta
    assert "+2.1%" in with_delta


def test_kpi_tile_escapes_label_and_value() -> None:
    out = c.kpi_tile(label="<script>", value="<x>", status="info")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "&lt;x&gt;" in out


# ---------- status_badge ----------


def test_status_badge_uppercases_state_label_by_default() -> None:
    out = c.status_badge("healthy")
    assert 'data-state="healthy"' in out
    assert "HEALTHY" in out


def test_status_badge_uses_custom_label_when_provided() -> None:
    out = c.status_badge("critical", label="ON FIRE")
    assert "ON FIRE" in out
    assert "CRITICAL" not in out
    assert 'data-state="critical"' in out


def test_status_badge_escapes_custom_label() -> None:
    out = c.status_badge("info", label="<x>")
    assert "<x>" not in out
    assert "&lt;x&gt;" in out


# ---------- trust_gauge ----------


def test_trust_gauge_clamps_score_to_unit_range() -> None:
    low = c.trust_gauge(score=-0.2)
    high = c.trust_gauge(score=1.5)
    # Both should render without raising; the displayed value
    # should sit in [0.00, 1.00].
    assert ">0.00<" in low
    assert ">1.00<" in high


def test_trust_gauge_shows_threshold_hairline() -> None:
    out = c.trust_gauge(score=0.5, threshold=0.6)
    # Hairline is the dashed <line> we emit.
    assert "stroke-dasharray=\"3 3\"" in out


def test_trust_gauge_band_colors_track_score() -> None:
    # 0.0 -> healthy band (score < threshold/2 = 0.25)
    healthy_out = c.trust_gauge(score=0.1, threshold=0.5)
    # 0.4 -> info band (between threshold/2 and threshold)
    info_out = c.trust_gauge(score=0.4, threshold=0.5)
    # 0.6 -> warning band (threshold to threshold + 0.2)
    warning_out = c.trust_gauge(score=0.6, threshold=0.5)
    # 0.9 -> critical band (above)
    critical_out = c.trust_gauge(score=0.9, threshold=0.5)
    # Pull out the brand colors so we don't repeat hex codes here.
    from gradio_app.theme import PALETTE
    assert PALETTE["healthy"] in healthy_out
    assert PALETTE["accent"] in info_out
    assert PALETTE["warning"] in warning_out
    assert PALETTE["critical"] in critical_out


# ---------- signal_bar ----------


def test_signal_bar_width_reflects_value() -> None:
    out = c.signal_bar(name="frustration", value=0.42, weight="heavy")
    assert 'data-weight="heavy"' in out
    assert "width: 42.0%" in out
    assert "0.42" in out


def test_signal_bar_clamps_extremes() -> None:
    low = c.signal_bar(name="x", value=-0.5)
    high = c.signal_bar(name="x", value=1.7)
    assert "width: 0.0%" in low
    assert "width: 100.0%" in high


# ---------- latency_ring ----------


def test_latency_ring_color_bands() -> None:
    from gradio_app.theme import PALETTE
    under = c.latency_ring(stage="STT", ms=100, target_ms=200)
    near = c.latency_ring(stage="STT", ms=180, target_ms=200)
    over = c.latency_ring(stage="STT", ms=300, target_ms=200)
    assert PALETTE["healthy"] in under
    assert PALETTE["warning"] in near
    assert PALETTE["critical"] in over


def test_latency_ring_caps_fill_at_100pct() -> None:
    # When ms is much larger than target, the fill arc must still be
    # bounded by the circumference (no overflow).
    out = c.latency_ring(stage="STT", ms=10_000, target_ms=200)
    # Pull dasharray fill length out; ensure it doesn't exceed
    # circumference. circumference = 2*pi*22 ≈ 138.23.
    import re
    match = re.search(r'stroke-dasharray="([\d.]+) ([\d.]+)"', out)
    assert match
    fill = float(match.group(1))
    circ = float(match.group(2))
    assert fill <= circ + 0.01


# ---------- tenant_card ----------


def test_tenant_card_active_flag_flips_data_attribute() -> None:
    inactive = c.tenant_card(
        customer_id="x",
        display_name="X",
        health="healthy",
        last_run_at="1m ago",
        is_active=False,
    )
    active = c.tenant_card(
        customer_id="x",
        display_name="X",
        health="healthy",
        last_run_at="1m ago",
        is_active=True,
    )
    assert 'data-active="false"' in inactive
    assert 'data-active="true"' in active


def test_tenant_card_embeds_status_badge() -> None:
    out = c.tenant_card(
        customer_id="x",
        display_name="X",
        health="critical",
        last_run_at="now",
    )
    assert 'data-state="critical"' in out


# ---------- incident_row ----------


def test_incident_row_renders_all_four_columns() -> None:
    out = c.incident_row(
        ts="2026-06-17 14:00",
        severity="warning",
        tenant="ophthalmology",
        summary="emergency phrase fired",
    )
    assert "2026-06-17 14:00" in out
    assert 'data-state="warning"' in out
    assert "ophthalmology" in out
    assert "emergency phrase fired" in out


def test_incident_row_escapes_summary() -> None:
    out = c.incident_row(
        ts="now",
        severity="info",
        tenant="t",
        summary="<script>alert(1)</script>",
    )
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


# ---------- agent_node ----------


def test_agent_node_minimum_render() -> None:
    out = c.agent_node(name="Router", state="active")
    assert "clarion-agent-node" in out
    assert 'data-state="active"' in out
    assert "Router" in out
    # No ms / no cost -> no meta block.
    assert "clarion-agent-node-meta" not in out


def test_agent_node_meta_when_provided() -> None:
    out = c.agent_node(name="LLM", state="done", ms=142, cost_usd=0.0023)
    assert "clarion-agent-node-meta" in out
    assert "142 ms" in out
    assert "$0.0023" in out


# ---------- cost_chip ----------


def test_cost_chip_default_period() -> None:
    out = c.cost_chip(usd=0.0123)
    assert "$0.0123" in out
    assert "/ turn" in out


def test_cost_chip_custom_period() -> None:
    out = c.cost_chip(usd=1.50, period="month")
    assert "$1.5000" in out
    assert "/ month" in out


# ---------- mono ----------


def test_mono_wraps_in_class() -> None:
    out = c.mono("trace_abc123")
    assert 'class="clarion-mono"' in out
    assert "trace_abc123" in out


def test_mono_escapes_input() -> None:
    out = c.mono("<x>")
    assert "<x>" not in out
    assert "&lt;x&gt;" in out


# ---------- panel + kpi_strip layouts ----------


def test_panel_wraps_body_with_title() -> None:
    body = "<p>hi</p>"
    out = c.panel(title="Trust", body_html=body)
    assert "clarion-panel" in out
    assert "Trust" in out
    assert body in out  # body is trusted HTML, NOT escaped


def test_kpi_strip_concatenates_tiles() -> None:
    tile1 = c.kpi_tile(label="A", value="1", status="info")
    tile2 = c.kpi_tile(label="B", value="2", status="healthy")
    out = c.kpi_strip([tile1, tile2])
    assert "clarion-kpi-strip" in out
    assert tile1 in out
    assert tile2 in out
