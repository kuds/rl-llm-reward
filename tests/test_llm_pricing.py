"""Tests for the USD cost estimator."""

from __future__ import annotations

from prompt_to_policy.llm import estimate_cost_usd
from prompt_to_policy.llm.pricing import PRICING_PER_MTOK_USD


def test_zero_usage_is_zero_cost():
    assert estimate_cost_usd("claude-opus-4-7", {"input_tokens": 0, "output_tokens": 0}) == 0.0


def test_known_model_uses_published_rate():
    in_rate, out_rate = PRICING_PER_MTOK_USD["claude-opus-4-7"]
    cost = estimate_cost_usd(
        "claude-opus-4-7", {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    )
    assert cost == in_rate + out_rate


def test_context_tier_suffix_strips():
    """A model id like 'claude-opus-4-7[1m]' falls back to the base rate."""
    base = estimate_cost_usd("claude-opus-4-7", {"input_tokens": 1_000_000, "output_tokens": 0})
    tier = estimate_cost_usd(
        "claude-opus-4-7[1m]", {"input_tokens": 1_000_000, "output_tokens": 0}
    )
    assert tier == base


def test_unknown_model_falls_back_to_opus_pricing():
    in_rate, _ = PRICING_PER_MTOK_USD["claude-opus-4-7"]
    cost = estimate_cost_usd(
        "claude-some-future-model", {"input_tokens": 1_000_000, "output_tokens": 0}
    )
    assert cost == in_rate


def test_partial_token_counts_scale_linearly():
    cost_full = estimate_cost_usd(
        "claude-sonnet-4-6", {"input_tokens": 1_000_000, "output_tokens": 0}
    )
    cost_half = estimate_cost_usd(
        "claude-sonnet-4-6", {"input_tokens": 500_000, "output_tokens": 0}
    )
    assert cost_half == cost_full / 2
