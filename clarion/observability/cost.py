"""Per-model token pricing and the cost calculator.

Prices are USD per 1k tokens as published by OpenAI at the time of
writing. They drift; the table lives here so updating one number flows
into every span the agent emits. An unknown model returns ``0.0`` so
local FakeLLM runs (no real billing) cleanly report cost=0 without
guessing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Per-1k-token rates."""

    input_per_1k: float
    output_per_1k: float


# Snapshot of public OpenAI pricing for the models the agent might touch.
# If you add a model in clarion/agents/openai_client.py, add a row here too.
_PRICING: dict[str, ModelPricing] = {
    "gpt-4o-mini": ModelPricing(input_per_1k=0.00015, output_per_1k=0.0006),
    "gpt-4o": ModelPricing(input_per_1k=0.0025, output_per_1k=0.01),
    "gpt-4-turbo": ModelPricing(input_per_1k=0.01, output_per_1k=0.03),
}


def known_models() -> list[str]:
    return sorted(_PRICING.keys())


def pricing_for(model: str) -> ModelPricing | None:
    return _PRICING.get(model)


def cost_usd(model: str, *, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost for one LLM call.

    Returns 0.0 for unknown models — the observability layer prefers
    "unknown cost reported as zero" over guessing. The presence of the
    span with ``cost_usd=0`` and ``model=<unknown>`` is enough for an
    operator to spot the gap and add a pricing row.
    """
    p = _PRICING.get(model)
    if p is None:
        return 0.0
    return (input_tokens / 1000.0) * p.input_per_1k + (output_tokens / 1000.0) * p.output_per_1k
