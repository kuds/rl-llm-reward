"""Smoke test: ask the LLM for three reward specs.

Per CLAUDE.md step 7: validates that the prompt template produces
sensible reward specs for representative prompts. Run with an
``ANTHROPIC_API_KEY`` environment variable; the resulting cached
responses land in ``examples/cached_responses/`` and let subsequent
runs of the demo (and other examples) work offline.

Usage:
    export ANTHROPIC_API_KEY=...
    python examples/smoke_llm_three_prompts.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from prompt_to_policy.envs import halfcheetah as hc
from prompt_to_policy.llm import LLMRewardClient
from prompt_to_policy.reward import build_reward_fn, smoke_test_reward_fn

CACHE_DIR = Path(__file__).parent / "cached_responses"

PROMPTS = [
    "make the cheetah run forward as fast as possible",
    "make the cheetah run backward as fast as possible",
    "make the cheetah stand still and stay upright",
]


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. Aborting.", file=sys.stderr)
        return 1

    client = LLMRewardClient(feature_docs=hc.FEATURE_DOCS, cache_dir=CACHE_DIR)
    smoke_env = hc.make_env()

    try:
        for i, prompt in enumerate(PROMPTS, start=1):
            print(f"\n[{i}/{len(PROMPTS)}] prompt: {prompt!r}")
            result = client.generate(prompt)
            print(f"  cached      : {result.cached}")
            print(f"  cache_key   : {result.cache_key}")
            print(f"  spec        : {result.spec.model_dump_json()}")
            print(f"  usage       : {result.usage}")
            print(f"  cost_usd    : {result.estimated_cost_usd:.4f}")

            # Validate the reward actually runs against the env.
            reward_fn = build_reward_fn(result.spec, hc.FEATURES)
            smoke_test_reward_fn(reward_fn, smoke_env, n_steps=8, seed=i)
            print("  smoke       : OK")
    finally:
        smoke_env.close()

    print(f"\nCache populated at: {CACHE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
