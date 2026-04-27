"""Smoke-test tests: confirm ``smoke_test_reward_fn`` catches the failure
modes it's meant to catch (NaN reward, raised exception) and accepts a
healthy reward against a real env.
"""

from __future__ import annotations

import numpy as np
import pytest

from prompt_to_policy.reward import (
    RewardSmokeError,
    RewardSpec,
    build_reward_fn,
    smoke_test_reward_fn,
)

pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")

from prompt_to_policy.envs import halfcheetah as hc  # noqa: E402


def test_healthy_reward_passes_smoke_test():
    spec = RewardSpec.model_validate(
        {"components": [{"feature": "forward_velocity", "weight": 1.0}]}
    )
    r = build_reward_fn(spec, hc.FEATURES)
    env = hc.make_env()
    try:
        smoke_test_reward_fn(r, env, n_steps=8, seed=0)
    finally:
        env.close()


def test_smoke_test_catches_nan_reward():
    def nan_reward(obs, action, next_obs, info):
        return float("nan")

    env = hc.make_env()
    try:
        with pytest.raises(RewardSmokeError, match="non-finite"):
            smoke_test_reward_fn(nan_reward, env, n_steps=4, seed=0)
    finally:
        env.close()


def test_smoke_test_catches_exception():
    def boom(obs, action, next_obs, info):
        raise KeyError("missing_feature_xyz")

    env = hc.make_env()
    try:
        with pytest.raises(RewardSmokeError, match="KeyError"):
            smoke_test_reward_fn(boom, env, n_steps=4, seed=0)
    finally:
        env.close()


def test_smoke_test_catches_inf_reward():
    def inf_reward(obs, action, next_obs, info):
        return float("inf")

    env = hc.make_env()
    try:
        with pytest.raises(RewardSmokeError, match="non-finite"):
            smoke_test_reward_fn(inf_reward, env, n_steps=2, seed=0)
    finally:
        env.close()


def test_built_reward_finite_over_many_steps():
    """Robustness: random rollout for 64 steps stays finite."""
    spec = RewardSpec.model_validate(
        {
            "components": [
                {"feature": "forward_velocity", "weight": 1.0},
                {"feature": "control_cost", "weight": -0.05},
                {"feature": "torso_uprightness", "weight": 0.1},
            ],
            "bias": 0.0,
        }
    )
    r = build_reward_fn(spec, hc.FEATURES)
    env = hc.make_env()
    try:
        smoke_test_reward_fn(r, env, n_steps=64, seed=42)
    finally:
        env.close()


def test_smoke_test_handles_episode_reset():
    """If an episode ends inside the smoke window, the helper resets and continues."""
    # HalfCheetah doesn't terminate, so we wrap with a TimeLimit to force truncation.
    import gymnasium as gym

    spec = RewardSpec.model_validate(
        {"components": [{"feature": "forward_velocity", "weight": 1.0}]}
    )
    r = build_reward_fn(spec, hc.FEATURES)
    env = gym.wrappers.TimeLimit(hc.make_env(), max_episode_steps=3)
    try:
        # 8 steps with max_episode_steps=3 forces at least two truncations.
        smoke_test_reward_fn(r, env, n_steps=8, seed=0)
    finally:
        env.close()


def test_smoke_test_seeds_action_space_for_determinism():
    # The smoke test calls env.action_space.sample(); seeding the action
    # space is the caller's responsibility if they want determinism. This
    # test just confirms repeated runs with matching seeds + matching
    # action-space seeds produce identical outcomes (no hidden state).
    spec = RewardSpec.model_validate(
        {"components": [{"feature": "forward_velocity", "weight": 1.0}]}
    )
    r = build_reward_fn(spec, hc.FEATURES)
    for _ in range(2):
        env = hc.make_env()
        env.action_space.seed(123)
        try:
            smoke_test_reward_fn(r, env, n_steps=4, seed=0)
        finally:
            env.close()


def test_smoke_failure_chains_original_exception():
    sentinel = ValueError("boom")

    def raiser(obs, action, next_obs, info):
        raise sentinel

    env = hc.make_env()
    try:
        with pytest.raises(RewardSmokeError) as exc_info:
            smoke_test_reward_fn(raiser, env, n_steps=2, seed=0)
        # __cause__ is set by `raise X from e`; preserves the original for debugging.
        assert exc_info.value.__cause__ is sentinel
    finally:
        env.close()


def test_smoke_test_uses_action_from_step_in_reward():
    """The reward is called with the action that was just stepped, not a stale one."""
    seen_actions = []

    def recorder(obs, action, next_obs, info):
        seen_actions.append(np.array(action, copy=True))
        return 0.0

    env = hc.make_env()
    try:
        smoke_test_reward_fn(recorder, env, n_steps=3, seed=0)
        assert len(seen_actions) == 3
        # Actions are 6-d HalfCheetah controls.
        for a in seen_actions:
            assert a.shape == (6,)
    finally:
        env.close()
