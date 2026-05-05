"""Smoke tests for the AntFeatureEnv wrapper. Requires gymnasium[mujoco]."""

from __future__ import annotations

import math

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")

from prompt_to_policy.envs import ant as ant_mod  # noqa: E402

REQUIRED_INFO_KEYS = (
    "x_velocity",
    "y_velocity",
    "z_velocity",
    "z_position",
    "upright_projection",
)


def _finite(x: float) -> bool:
    return math.isfinite(x)


def test_make_env_returns_wrapped_env():
    env = ant_mod.make_env()
    try:
        assert isinstance(env, ant_mod.AntFeatureEnv)
        assert env.action_space.shape == (8,)
    finally:
        env.close()


def test_wrapper_populates_info_on_reset():
    env = ant_mod.make_env()
    try:
        _, info = env.reset(seed=0)
        for key in REQUIRED_INFO_KEYS:
            assert key in info, f"missing key {key!r} after reset"
            assert _finite(info[key])
    finally:
        env.close()


def test_wrapper_populates_info_on_step():
    env = ant_mod.make_env()
    try:
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        for key in REQUIRED_INFO_KEYS:
            assert key in info
            assert _finite(info[key])
    finally:
        env.close()


def test_features_evaluate_on_real_step_without_error():
    env = ant_mod.make_env()
    try:
        obs, _ = env.reset(seed=0)
        action = env.action_space.sample()
        next_obs, _, _, _, info = env.step(action)
        for _name, fn in ant_mod.FEATURES.items():
            v = fn(obs, action, next_obs, info)
            assert isinstance(v, float)
            assert _finite(v)
    finally:
        env.close()


def test_uprightness_is_near_one_at_rest():
    """Ant resets in an upright pose; the projection should be ~+1."""
    env = ant_mod.make_env()
    try:
        _, info = env.reset(seed=0)
        # Ant's default starting orientation is approximately upright.
        assert info["upright_projection"] > 0.9
    finally:
        env.close()
