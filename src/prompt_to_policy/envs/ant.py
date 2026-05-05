"""Ant-v5 env wrapper and feature registry.

Ant is a 3D quadruped. Unlike HalfCheetah and Hopper it has a free
torso quaternion, so uprightness is computed by projecting the body's
local +z axis onto the world +z axis. Locomotion is two-dimensional in
the (x, y) plane, which is why a separate ``lateral_velocity`` feature
is exposed in addition to ``forward_velocity``.

The wrapper augments ``info`` with named scalars so feature functions
read from ``info`` exclusively, isolating qpos/qvel indexing to one
place.
"""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np

ENV_ID = "Ant-v5"

# qpos layout for Ant-v5: [rootx, rooty, rootz, qw, qx, qy, qz, ...joints].
# qvel layout: [vx, vy, vz, wx, wy, wz, ...joint_vels].
QPOS_ROOTZ = 2
QPOS_QUAT = (3, 4, 5, 6)  # (w, x, y, z)
QVEL_ROOTX = 0
QVEL_ROOTY = 1
QVEL_ROOTZ = 2


def _world_z_of_body(qw: float, qx: float, qy: float, qz: float) -> float:
    """Return the z-component of the body's +z axis expressed in world frame.

    +1 if the body is upright, 0 if lying on its side, -1 if upside down.
    Derived from rotating ``[0, 0, 1]`` by the unit quaternion (w, x, y, z).
    """
    return 1.0 - 2.0 * (qx * qx + qy * qy)


class AntFeatureEnv(gym.Wrapper):
    """Wraps Ant-v5 and adds named scalars to ``info``.

    Adds:
        x_velocity         signed forward velocity of the torso (m/s)
        y_velocity         signed lateral velocity of the torso (m/s)
        z_velocity         signed vertical velocity of the torso (m/s)
        z_position         torso height (m)
        upright_projection cos(angle between torso +z and world +z)
    """

    def _augment(self, info: dict) -> dict:
        data = self.unwrapped.data
        qpos = data.qpos
        qvel = data.qvel
        qw = float(qpos[QPOS_QUAT[0]])
        qx = float(qpos[QPOS_QUAT[1]])
        qy = float(qpos[QPOS_QUAT[2]])
        qz = float(qpos[QPOS_QUAT[3]])
        out = dict(info)
        out["z_position"] = float(qpos[QPOS_ROOTZ])
        out["z_velocity"] = float(qvel[QVEL_ROOTZ])
        out["upright_projection"] = _world_z_of_body(qw, qx, qy, qz)
        out.setdefault("x_velocity", float(qvel[QVEL_ROOTX]))
        out.setdefault("y_velocity", float(qvel[QVEL_ROOTY]))
        return out

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._augment(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._augment(info)


def make_env(render_mode: str | None = None) -> gym.Env:
    """Construct an Ant-v5 env with the feature-augmenting wrapper."""
    base = gym.make(ENV_ID, render_mode=render_mode)
    return AntFeatureEnv(base)


# --- Feature registry -------------------------------------------------------

FeatureFn = Callable[[np.ndarray, np.ndarray, np.ndarray, dict], float]


def _forward_velocity(obs, action, next_obs, info) -> float:
    return float(info["x_velocity"])


def _lateral_velocity(obs, action, next_obs, info) -> float:
    return float(info["y_velocity"])


def _planar_speed(obs, action, next_obs, info) -> float:
    vx = float(info["x_velocity"])
    vy = float(info["y_velocity"])
    return float(np.hypot(vx, vy))


def _vertical_velocity(obs, action, next_obs, info) -> float:
    return float(info["z_velocity"])


def _height(obs, action, next_obs, info) -> float:
    return float(info["z_position"])


def _torso_uprightness(obs, action, next_obs, info) -> float:
    return float(info["upright_projection"])


def _control_cost(obs, action, next_obs, info) -> float:
    return float(np.sum(np.square(np.asarray(action, dtype=np.float64))))


def _alive_bonus(obs, action, next_obs, info) -> float:
    return 1.0


FEATURES: dict[str, FeatureFn] = {
    "forward_velocity": _forward_velocity,
    "lateral_velocity": _lateral_velocity,
    "planar_speed": _planar_speed,
    "vertical_velocity": _vertical_velocity,
    "height": _height,
    "torso_uprightness": _torso_uprightness,
    "control_cost": _control_cost,
    "alive_bonus": _alive_bonus,
}

FEATURE_DOCS: dict[str, str] = {
    "forward_velocity": "Signed x-velocity of the torso (m/s). Positive = forward.",
    "lateral_velocity": (
        "Signed y-velocity of the torso (m/s). Positive = leftward (env's +y direction)."
    ),
    "planar_speed": (
        "Magnitude of the (x, y) velocity vector (m/s). Always >= 0; ignores direction."
    ),
    "vertical_velocity": "Signed z-velocity of the torso (m/s). Positive = rising.",
    "height": (
        "z-position of the torso (m). Default starting height ~0.75m. "
        "Ant terminates if z leaves the healthy range [0.2, 1.0]."
    ),
    "torso_uprightness": (
        "Projection of the body's +z axis onto the world +z axis. "
        "+1 fully upright, 0 lying on side, -1 upside down."
    ),
    "control_cost": (
        "Sum of squared joint actions. Always >= 0; use a negative weight to penalize energy."
    ),
    "alive_bonus": (
        "Constant 1.0. Because Ant terminates on fall, weighting alive_bonus encourages "
        "the agent to stay in the healthy state distribution."
    ),
}


from prompt_to_policy.llm.templates.ant import (  # noqa: E402
    build_system_prompt as _build_system_prompt,
)

from .registry import EnvSpec  # noqa: E402

SPEC = EnvSpec(
    name="ant",
    env_id=ENV_ID,
    make_env=make_env,
    features=FEATURES,
    feature_docs=FEATURE_DOCS,
    build_system_prompt=_build_system_prompt,
)
