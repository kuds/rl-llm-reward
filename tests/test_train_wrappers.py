"""Unit tests for the DSL reward wrapper.

The wrapper must call the reward fn with the *pre-step* obs as ``obs``
and the *post-step* obs as ``next_obs``. These tests use a stub env so
no MuJoCo install is needed.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from prompt_to_policy.train.wrappers import DSLRewardWrapper


class _StubEnv(gym.Env):
    """Minimal env: obs is a counter; action is ignored; reward is 999.0 (to confirm overwrite)."""

    metadata = {"render_modes": []}
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def __init__(self):
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.array([0.0], dtype=np.float32), {"step": 0}

    def step(self, action):
        self._t += 1
        obs = np.array([float(self._t)], dtype=np.float32)
        return obs, 999.0, False, self._t >= 5, {"step": self._t}


def test_wrapper_replaces_native_reward():
    calls: list[tuple] = []

    def reward(obs, action, next_obs, info):
        calls.append((float(obs[0]), float(next_obs[0]), info["step"]))
        return 7.0

    env = DSLRewardWrapper(_StubEnv(), reward)
    obs, _ = env.reset()
    assert obs.tolist() == [0.0]

    next_obs, r, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
    assert r == 7.0  # native 999.0 was discarded
    assert next_obs.tolist() == [1.0]
    assert calls == [(0.0, 1.0, 1)]  # pre-step obs was 0.0, post-step is 1.0


def test_wrapper_passes_pre_and_post_step_obs():
    seen: list[tuple[float, float]] = []

    def reward(obs, action, next_obs, info):
        seen.append((float(obs[0]), float(next_obs[0])))
        return 0.0

    env = DSLRewardWrapper(_StubEnv(), reward)
    env.reset()
    for _ in range(3):
        env.step(np.zeros(2, dtype=np.float32))

    assert seen == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]


def test_wrapper_resets_last_obs_between_episodes():
    seen: list[tuple[float, float]] = []

    def reward(obs, action, next_obs, info):
        seen.append((float(obs[0]), float(next_obs[0])))
        return 0.0

    env = DSLRewardWrapper(_StubEnv(), reward)
    env.reset()
    for _ in range(2):
        env.step(np.zeros(2, dtype=np.float32))
    env.reset()
    env.step(np.zeros(2, dtype=np.float32))

    assert seen[-1] == (0.0, 1.0)  # post-reset, the pre-step obs is 0.0 again


def test_wrapper_returns_python_float_reward():
    def reward(obs, action, next_obs, info):
        return np.float64(3.14)

    env = DSLRewardWrapper(_StubEnv(), reward)
    env.reset()
    _, r, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert isinstance(r, float) and not isinstance(r, np.floating)


@pytest.mark.parametrize("n_steps", [1, 3, 5])
def test_wrapper_propagates_truncation(n_steps: int):
    env = DSLRewardWrapper(_StubEnv(), lambda *_: 0.0)
    env.reset()
    truncs = []
    for _ in range(n_steps):
        _, _, term, trunc, _ = env.step(np.zeros(2, dtype=np.float32))
        truncs.append(trunc)
    # _StubEnv truncates at step 5
    assert truncs[-1] == (n_steps >= 5)
