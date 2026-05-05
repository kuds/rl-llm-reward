"""USD cost estimator for Anthropic API usage.

These prices are baked in for the demo's convenience. They may lag
actual published pricing — Anthropic publishes the authoritative table
at https://www.anthropic.com/pricing. ``summary.json`` always logs the
raw token counts alongside the estimate, so the dollar figure can be
re-derived later if pricing has shifted.
"""

from __future__ import annotations

# (input_per_mtok_usd, output_per_mtok_usd). As of 2026-04.
PRICING_PER_MTOK_USD: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    # Google Gemini (pay-as-you-go list prices).
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}


def _lookup(model: str) -> tuple[float, float]:
    if model in PRICING_PER_MTOK_USD:
        return PRICING_PER_MTOK_USD[model]
    # Strip any "[1m]" context-tier suffix (e.g. "claude-opus-4-7[1m]").
    base = model.split("[")[0]
    if base in PRICING_PER_MTOK_USD:
        return PRICING_PER_MTOK_USD[base]
    # Conservative fallback: charge as Opus to avoid under-reporting spend.
    return PRICING_PER_MTOK_USD["claude-opus-4-7"]


def estimate_cost_usd(model: str, usage: dict[str, int]) -> float:
    """Estimate USD cost from input/output token counts."""
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    input_price, output_price = _lookup(model)
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
