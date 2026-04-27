"""Builder + hand-written spec tests for the reward DSL.

These tests exercise the canonical v0 prompt set as JSON specs against
the HalfCheetah feature registry, with hand-built ``info`` dicts. No
MuJoCo install is required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from prompt_to_policy.envs import halfcheetah as hc
from prompt_to_policy.reward import (
    RewardSpec,
    UnknownFeatureError,
    build_reward_fn,
)

OBS = np.zeros(17, dtype=np.float32)
NEXT_OBS = np.zeros(17, dtype=np.float32)
ZERO_ACTION = np.zeros(6, dtype=np.float32)


def info(**overrides) -> dict:
    base = {
        "x_velocity": 0.0,
        "z_velocity": 0.0,
        "z_position": 0.0,
        "pitch_angle": 0.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Hand-written specs: the v0 prompt set on HalfCheetah.
# ---------------------------------------------------------------------------


def test_forward_locomotion_spec():
    """Run forward as fast as possible, with a small energy penalty."""
    spec = RewardSpec.model_validate(
        {
            "components": [
                {"feature": "forward_velocity", "weight": 1.0},
                {"feature": "control_cost", "weight": -0.05},
            ]
        }
    )
    r = build_reward_fn(spec, hc.FEATURES)
    a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # x_vel = 3.0, ctrl_cost = 1.0 -> 1.0 * 3.0 + (-0.05) * 1.0 = 2.95
    assert r(OBS, a, NEXT_OBS, info(x_velocity=3.0)) == pytest.approx(2.95)
    # Standing still gives 0 forward + 0 cost = 0
    assert r(OBS, ZERO_ACTION, NEXT_OBS, info(x_velocity=0.0)) == pytest.approx(0.0)
    # Going backward is penalized by the forward_velocity term
    assert r(OBS, ZERO_ACTION, NEXT_OBS, info(x_velocity=-2.0)) == pytest.approx(-2.0)


def test_backward_locomotion_spec():
    """Run backward: same shape as forward, weight flipped."""
    spec = RewardSpec.model_validate(
        {
            "components": [
                {"feature": "forward_velocity", "weight": -1.0},
                {"feature": "control_cost", "weight": -0.05},
            ]
        }
    )
    r = build_reward_fn(spec, hc.FEATURES)
    assert r(OBS, ZERO_ACTION, NEXT_OBS, info(x_velocity=-3.0)) == pytest.approx(3.0)
    assert r(OBS, ZERO_ACTION, NEXT_OBS, info(x_velocity=3.0)) == pytest.approx(-3.0)


def test_stand_still_spec():
    """Stand still: penalize any motion, reward staying upright."""
    spec = RewardSpec.model_validate(
        {
            "components": [
                {"feature": "speed_magnitude", "weight": -1.0},
                {"feature": "torso_uprightness", "weight": 0.5},
            ],
            "bias": 0.0,
        }
    )
    r = build_reward_fn(spec, hc.FEATURES)
    # Perfectly still and upright: -1 * 0 + 0.5 * cos(0) = 0.5
    assert r(OBS, ZERO_ACTION, NEXT_OBS, info()) == pytest.approx(0.5)
    # Moving (in either direction): penalized by speed_magnitude
    assert r(OBS, ZERO_ACTION, NEXT_OBS, info(x_velocity=2.0)) == pytest.approx(-1.5)
    assert r(OBS, ZERO_ACTION, NEXT_OBS, info(x_velocity=-2.0)) == pytest.approx(-1.5)
    # Tipped 90 deg, no motion: -1*0 + 0.5*cos(pi/2) = 0
    assert r(OBS, ZERO_ACTION, NEXT_OBS, info(pitch_angle=math.pi / 2)) == pytest.approx(
        0.0, abs=1e-9
    )


def test_hop_spec():
    """Hop: reward upward velocity and height; small energy penalty."""
    spec = RewardSpec.model_validate(
        {
            "components": [
                {"feature": "vertical_velocity", "weight": 1.0},
                {"feature": "height", "weight": 0.5},
                {"feature": "control_cost", "weight": -0.01},
            ]
        }
    )
    r = build_reward_fn(spec, hc.FEATURES)
    a = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # z_vel=2, height=0.4, ctrl_cost=2 -> 1*2 + 0.5*0.4 + -0.01*2 = 2.18
    assert r(OBS, a, NEXT_OBS, info(z_velocity=2.0, z_position=0.4)) == pytest.approx(2.18)


def test_bias_is_added_once():
    spec = RewardSpec.model_validate(
        {
            "components": [{"feature": "forward_velocity", "weight": 1.0}],
            "bias": 0.25,
        }
    )
    r = build_reward_fn(spec, hc.FEATURES)
    assert r(OBS, ZERO_ACTION, NEXT_OBS, info(x_velocity=1.0)) == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# Builder validation: unknown features.
# ---------------------------------------------------------------------------


def test_unknown_feature_rejected_by_builder():
    spec = RewardSpec.model_validate(
        {"components": [{"feature": "totally_made_up", "weight": 1.0}]}
    )
    with pytest.raises(UnknownFeatureError) as exc:
        build_reward_fn(spec, hc.FEATURES)
    msg = str(exc.value)
    assert "totally_made_up" in msg
    # Error names the available registry so an LLM caller can correct itself.
    assert "forward_velocity" in msg


def test_partial_unknown_feature_rejected():
    spec = RewardSpec.model_validate(
        {
            "components": [
                {"feature": "forward_velocity", "weight": 1.0},
                {"feature": "ghost", "weight": 0.5},
            ]
        }
    )
    with pytest.raises(UnknownFeatureError):
        build_reward_fn(spec, hc.FEATURES)


# ---------------------------------------------------------------------------
# Builder return-type contract.
# ---------------------------------------------------------------------------


def test_built_reward_returns_python_float():
    spec = RewardSpec.model_validate(
        {"components": [{"feature": "forward_velocity", "weight": 1.0}]}
    )
    r = build_reward_fn(spec, hc.FEATURES)
    out = r(OBS, ZERO_ACTION, NEXT_OBS, info(x_velocity=1.5))
    assert isinstance(out, float)
