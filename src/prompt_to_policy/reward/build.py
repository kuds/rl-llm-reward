"""Build a callable reward function from a RewardSpec and a feature registry.

The builder resolves feature names to callables once and closes over the
list, so the per-step path is a tight loop. A separate smoke-test helper
exercises the built reward on a few real env steps to catch NaN/inf and
exceptions before training begins.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .spec import RewardSpec

RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, dict], float]
FeatureFn = Callable[[np.ndarray, np.ndarray, np.ndarray, dict], float]


class UnknownFeatureError(ValueError):
    """A spec referenced a feature name not present in the registry."""


class RewardSmokeError(RuntimeError):
    """A built reward produced a non-finite value or raised on a real env step."""


def build_reward_fn(spec: RewardSpec, registry: Mapping[str, FeatureFn]) -> RewardFn:
    """Compile a RewardSpec against a feature registry into a RewardFn.

    Raises ``UnknownFeatureError`` if any component references a feature
    not in ``registry``.
    """
    unknown = [c.feature for c in spec.components if c.feature not in registry]
    if unknown:
        raise UnknownFeatureError(f"Unknown features: {unknown}. Available: {sorted(registry)}")

    components = [(registry[c.feature], float(c.weight)) for c in spec.components]
    bias = float(spec.bias)

    def reward_fn(obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, info: dict) -> float:
        total = bias
        for fn, w in components:
            total += w * fn(obs, action, next_obs, info)
        return float(total)

    return reward_fn


def smoke_test_reward_fn(
    reward_fn: RewardFn,
    env: Any,
    n_steps: int = 16,
    seed: int = 0,
) -> None:
    """Run ``reward_fn`` for ``n_steps`` random actions in ``env``.

    Raises ``RewardSmokeError`` on the first non-finite reward or
    exception. Intended to be called on every LLM-produced reward before
    handing it to the training harness. The caller owns ``env``'s
    lifetime and is responsible for closing it.
    """
    obs, info = env.reset(seed=seed)
    for step_idx in range(n_steps):
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, info = env.step(action)
        try:
            r = reward_fn(obs, action, next_obs, info)
        except Exception as e:  # noqa: BLE001 — re-raised as a typed error
            raise RewardSmokeError(
                f"reward raised {type(e).__name__} at step {step_idx}: {e}"
            ) from e
        if not math.isfinite(r):
            raise RewardSmokeError(f"reward produced non-finite value {r!r} at step {step_idx}")
        if terminated or truncated:
            obs, info = env.reset(seed=seed)
        else:
            obs = next_obs
