"""HalfCheetah-v5 env wrapper and feature registry.

The wrapper augments the env's ``info`` dict with named scalar quantities
(torso height, pitch angle, z-velocity, x-velocity) so that feature
functions read from ``info`` exclusively. This isolates the qpos/qvel
indexing convention to one place; if Gymnasium changes the obs layout
between versions, only this file needs to update.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import gymnasium as gym
import numpy as np

ENV_ID = "HalfCheetah-v5"

# qpos layout for HalfCheetah-v5: [rootx, rootz, rooty, bthigh, bshin, bfoot,
# fthigh, fshin, ffoot]. qvel uses the same ordering.
QPOS_ROOTZ = 1
QPOS_ROOTY = 2
QVEL_ROOTX = 0
QVEL_ROOTZ = 1


class HalfCheetahFeatureEnv(gym.Wrapper):
    """Wraps HalfCheetah-v5 and adds named scalars to ``info``.

    Adds the following keys (always present, including on reset):
        x_velocity         signed forward velocity of the torso (m/s)
        z_velocity         signed vertical velocity of the torso (m/s)
        z_position         torso height (m)
        pitch_angle        rotation of the torso about the y-axis (rad)
    """

    def _augment(self, info: dict) -> dict:
        data = self.unwrapped.data
        out = dict(info)
        out["z_position"] = float(data.qpos[QPOS_ROOTZ])
        out["pitch_angle"] = float(data.qpos[QPOS_ROOTY])
        out["z_velocity"] = float(data.qvel[QVEL_ROOTZ])
        out.setdefault("x_velocity", float(data.qvel[QVEL_ROOTX]))
        return out

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._augment(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._augment(info)


def make_env(render_mode: str | None = None) -> gym.Env:
    """Construct a HalfCheetah-v5 env with the feature-augmenting wrapper."""
    base = gym.make(ENV_ID, render_mode=render_mode)
    return HalfCheetahFeatureEnv(base)


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
    "control_cost": _control_cost,
    "alive_bonus": _alive_bonus,
}

FEATURE_DOCS: dict[str, str] = {
    "forward_velocity": "Signed x-velocity of the torso (m/s). Positive = forward.",
    "speed_magnitude": "|x-velocity| of the torso (m/s). Always >= 0.",
    "vertical_velocity": "Signed z-velocity of the torso (m/s). Positive = upward.",
    "height": "z-position of the torso (m). ~0 at default pose; rises when the body lifts.",
    "torso_uprightness": ("cos(pitch_angle). +1 when level, 0 at 90 deg, -1 when fully inverted."),
    "control_cost": (
        "Sum of squared joint actions. Always >= 0; use a negative weight to penalize energy."
    ),
    "alive_bonus": "Constant 1.0. HalfCheetah does not terminate; useful as a bias term.",
}
