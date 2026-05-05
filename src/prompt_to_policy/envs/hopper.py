"""Hopper-v5 env wrapper and feature registry.

Hopper is a planar 2D one-legged robot. Unlike HalfCheetah it actually
*terminates* on falls, so the alive_bonus feature has real teeth — it
rewards whatever keeps the agent in the healthy pose distribution.

The wrapper augments ``info`` with named scalars so feature functions
read from ``info`` exclusively, isolating qpos/qvel indexing to one
place.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import gymnasium as gym
import numpy as np

ENV_ID = "Hopper-v5"

# qpos layout for Hopper-v5: [rootx, rootz, rooty, thigh, leg, foot]. qvel
# uses the same ordering.
QPOS_ROOTZ = 1
QPOS_ROOTY = 2
QVEL_ROOTX = 0
QVEL_ROOTZ = 1
QVEL_PITCH = 2


class HopperFeatureEnv(gym.Wrapper):
    """Wraps Hopper-v5 and adds named scalars to ``info``.

    Adds:
        x_velocity         signed forward velocity of the torso (m/s)
        z_velocity         signed vertical velocity of the torso (m/s)
        z_position         torso height (m)
        pitch_angle        rotation of the torso about the y-axis (rad)
        pitch_velocity     angular velocity about the y-axis (rad/s)
    """

    def _augment(self, info: dict) -> dict:
        data = self.unwrapped.data
        out = dict(info)
        out["z_position"] = float(data.qpos[QPOS_ROOTZ])
        out["pitch_angle"] = float(data.qpos[QPOS_ROOTY])
        out["z_velocity"] = float(data.qvel[QVEL_ROOTZ])
        out["pitch_velocity"] = float(data.qvel[QVEL_PITCH])
        out.setdefault("x_velocity", float(data.qvel[QVEL_ROOTX]))
        return out

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._augment(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._augment(info)


def make_env(render_mode: str | None = None) -> gym.Env:
    """Construct a Hopper-v5 env with the feature-augmenting wrapper."""
    base = gym.make(ENV_ID, render_mode=render_mode)
    return HopperFeatureEnv(base)


# --- Feature registry -------------------------------------------------------

FeatureFn = Callable[[np.ndarray, np.ndarray, np.ndarray, dict], float]


def _forward_velocity(obs, action, next_obs, info) -> float:
    return float(info["x_velocity"])


def _speed_magnitude(obs, action, next_obs, info) -> float:
    return abs(float(info["x_velocity"]))


def _vertical_velocity(obs, action, next_obs, info) -> float:
    return float(info["z_velocity"])


def _height(obs, action, next_obs, info) -> float:
    return float(info["z_position"])


def _torso_uprightness(obs, action, next_obs, info) -> float:
    return math.cos(float(info["pitch_angle"]))


def _pitch_velocity(obs, action, next_obs, info) -> float:
    return float(info["pitch_velocity"])


def _control_cost(obs, action, next_obs, info) -> float:
    return float(np.sum(np.square(np.asarray(action, dtype=np.float64))))


def _alive_bonus(obs, action, next_obs, info) -> float:
    return 1.0


FEATURES: dict[str, FeatureFn] = {
    "forward_velocity": _forward_velocity,
    "speed_magnitude": _speed_magnitude,
    "vertical_velocity": _vertical_velocity,
    "height": _height,
    "torso_uprightness": _torso_uprightness,
    "pitch_velocity": _pitch_velocity,
    "control_cost": _control_cost,
    "alive_bonus": _alive_bonus,
}

FEATURE_DOCS: dict[str, str] = {
    "forward_velocity": "Signed x-velocity of the torso (m/s). Positive = forward.",
    "speed_magnitude": "|x-velocity| of the torso (m/s). Always >= 0.",
    "vertical_velocity": (
        "Signed z-velocity of the torso (m/s). Positive when rising; negative when falling."
    ),
    "height": (
        "z-position of the torso (m). Default starting height ~1.25m. "
        "Hopper terminates if z drops below ~0.7m."
    ),
    "torso_uprightness": (
        "cos(pitch_angle). +1 when fully upright, 0 at 90 deg lean, -1 when inverted."
    ),
    "pitch_velocity": (
        "Angular velocity of the torso about the y-axis (rad/s). "
        "Useful for penalizing wobble or rewarding flips."
    ),
    "control_cost": (
        "Sum of squared joint actions. Always >= 0; use a negative weight to penalize energy."
    ),
    "alive_bonus": (
        "Constant 1.0. Because Hopper terminates on fall, weighting alive_bonus encourages "
        "the agent to stay in the healthy state distribution."
    ),
}


from prompt_to_policy.llm.templates.hopper import (  # noqa: E402
    build_system_prompt as _build_system_prompt,
)

from .registry import EnvSpec  # noqa: E402

SPEC = EnvSpec(
    name="hopper",
    env_id=ENV_ID,
    make_env=make_env,
    features=FEATURES,
    feature_docs=FEATURE_DOCS,
    build_system_prompt=_build_system_prompt,
)
