"""Tests for the cost calculator."""

from __future__ import annotations

import pytest
from clarion.observability.cost import cost_usd, known_models, pricing_for


def test_known_models_lists_gpt4o_mini() -> None:
    assert "gpt-4o-mini" in known_models()


def test_pricing_for_known_model_returns_struct() -> None:
    p = pricing_for("gpt-4o-mini")
    assert p is not None
    assert p.input_per_1k > 0
    assert p.output_per_1k > 0


def test_pricing_for_unknown_model_is_none() -> None:
    assert pricing_for("not-a-model") is None


@pytest.mark.parametrize(
    "model, in_t, out_t, expected",
    [
        # gpt-4o-mini: $0.00015 / 1k in, $0.0006 / 1k out
        ("gpt-4o-mini", 1000, 0, 0.00015),
        ("gpt-4o-mini", 0, 1000, 0.0006),
        ("gpt-4o-mini", 1000, 1000, 0.00015 + 0.0006),
        # Half rates.
        ("gpt-4o-mini", 500, 500, (0.00015 + 0.0006) / 2),
        # Zero tokens.
        ("gpt-4o-mini", 0, 0, 0.0),
    ],
)
def test_cost_usd_computes_expected(model: str, in_t: int, out_t: int, expected: float) -> None:
    assert cost_usd(model, input_tokens=in_t, output_tokens=out_t) == pytest.approx(
        expected, rel=1e-9
    )


def test_cost_usd_unknown_model_returns_zero() -> None:
    assert cost_usd("ghost-model", input_tokens=10_000, output_tokens=10_000) == 0.0
