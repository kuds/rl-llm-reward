"""Env wrappers used in training.

The DSLRewardWrapper replaces the env's native reward with a built
``RewardFn``. It owns the ``last_obs`` so the reward signature
``(obs, action, next_obs, info)`` is honored faithfully — i.e. ``obs``
is the observation that produced ``action``, not the post-step obs.
"""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np

RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, dict], float]


class DSLRewardWrapper(gym.Wrapper):
    """Replaces ``env.step`` reward with ``reward_fn(obs, action, next_obs, info)``."""

    def __init__(self, env: gym.Env, reward_fn: RewardFn):
        super().__init__(env)
        self._reward_fn = reward_fn
        self._last_obs: np.ndarray | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        next_obs, _native_reward, terminated, truncated, info = self.env.step(action)
        reward = float(self._reward_fn(self._last_obs, action, next_obs, info))
        self._last_obs = next_obs
        return next_obs, reward, terminated, truncated, info
