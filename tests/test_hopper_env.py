"""Smoke tests for the HopperFeatureEnv wrapper.

Requires ``gymnasium[mujoco]``. Confirms the wrapper populates the keys
the feature registry expects, on both reset and step, and that the
values are finite.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")

from prompt_to_policy.envs import hopper as hp  # noqa: E402

REQUIRED_INFO_KEYS = (
    "x_velocity",
    "z_velocity",
    "z_position",
    "pitch_angle",
    "pitch_velocity",
)


def _finite(x: float) -> bool:
    return math.isfinite(x)


def test_make_env_returns_wrapped_env():
    env = hp.make_env()
    try:
        assert isinstance(env, hp.HopperFeatureEnv)
        assert env.observation_space.shape == (11,)
        assert env.action_space.shape == (3,)
    finally:
        env.close()


def test_wrapper_populates_info_on_reset():
    env = hp.make_env()
    try:
        _, info = env.reset(seed=0)
        for key in REQUIRED_INFO_KEYS:
            assert key in info, f"missing key {key!r} after reset"
            assert _finite(info[key]), f"info[{key!r}] not finite: {info[key]!r}"
    finally:
        env.close()


def test_wrapper_populates_info_on_step():
    env = hp.make_env()
    try:
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        for key in REQUIRED_INFO_KEYS:
            assert key in info, f"missing key {key!r} after step"
            assert _finite(info[key]), f"info[{key!r}] not finite: {info[key]!r}"
    finally:
        env.close()


def test_features_evaluate_on_real_step_without_error():
    env = hp.make_env()
    try:
        obs, _ = env.reset(seed=0)
        action = env.action_space.sample()
        next_obs, _, _, _, info = env.step(action)
        for name, fn in hp.FEATURES.items():
            v = fn(obs, action, next_obs, info)
            assert isinstance(v, float), f"{name} returned non-float {type(v).__name__}"
            assert _finite(v), f"{name} returned non-finite {v!r}"
    finally:
        env.close()
