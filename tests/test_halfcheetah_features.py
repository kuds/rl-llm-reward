"""Unit tests for the HalfCheetah feature registry.

Features are pure functions of ``(obs, action, next_obs, info)``. These
tests exercise each feature with hand-built info dicts and synthetic
actions; no MuJoCo install is required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from prompt_to_policy.envs import halfcheetah as hc

DUMMY_OBS = np.zeros(17, dtype=np.float32)
DUMMY_NEXT_OBS = np.zeros(17, dtype=np.float32)
DUMMY_ACTION = np.zeros(6, dtype=np.float32)


def info(**overrides) -> dict:
    base = {
        "x_velocity": 0.0,
        "z_velocity": 0.0,
        "z_position": 0.0,
        "pitch_angle": 0.0,
    }
    base.update(overrides)
    return base


def call(name: str, *, action=DUMMY_ACTION, **info_kwargs) -> float:
    fn = hc.FEATURES[name]
    return fn(DUMMY_OBS, action, DUMMY_NEXT_OBS, info(**info_kwargs))


def test_registry_contains_expected_feature_names():
    expected = {
        "forward_velocity",
        "speed_magnitude",
        "vertical_velocity",
        "height",
        "torso_uprightness",
        "control_cost",
        "alive_bonus",
    }
    assert set(hc.FEATURES) == expected


def test_every_feature_has_a_docstring_entry():
    assert set(hc.FEATURE_DOCS) == set(hc.FEATURES)
    for name, doc in hc.FEATURE_DOCS.items():
        assert doc.strip(), f"feature {name!r} has empty doc"


def test_forward_velocity_is_signed():
    assert call("forward_velocity", x_velocity=2.5) == pytest.approx(2.5)
    assert call("forward_velocity", x_velocity=-1.25) == pytest.approx(-1.25)


def test_speed_magnitude_is_absolute_value():
    assert call("speed_magnitude", x_velocity=2.5) == pytest.approx(2.5)
    assert call("speed_magnitude", x_velocity=-1.25) == pytest.approx(1.25)
    assert call("speed_magnitude", x_velocity=0.0) == 0.0


def test_vertical_velocity_is_signed():
    assert call("vertical_velocity", z_velocity=0.7) == pytest.approx(0.7)
    assert call("vertical_velocity", z_velocity=-0.3) == pytest.approx(-0.3)


def test_height_passes_through_z_position():
    assert call("height", z_position=0.42) == pytest.approx(0.42)


def test_torso_uprightness_is_cos_pitch():
    assert call("torso_uprightness", pitch_angle=0.0) == pytest.approx(1.0)
    assert call("torso_uprightness", pitch_angle=math.pi / 2) == pytest.approx(0.0, abs=1e-9)
    assert call("torso_uprightness", pitch_angle=math.pi) == pytest.approx(-1.0)
    assert call("torso_uprightness", pitch_angle=-math.pi / 3) == pytest.approx(0.5, abs=1e-9)


def test_control_cost_is_l2_squared_and_nonnegative():
    assert call("control_cost", action=np.zeros(6, dtype=np.float32)) == 0.0
    a = np.array([1.0, -1.0, 0.5, 0.0, -0.5, 0.0], dtype=np.float32)
    expected = float(np.sum(a * a))
    assert call("control_cost", action=a) == pytest.approx(expected)
    # Cost is always non-negative regardless of sign.
    a_neg = -a
    assert call("control_cost", action=a_neg) == pytest.approx(expected)


def test_alive_bonus_is_constant_one():
    assert call("alive_bonus") == 1.0
    assert call("alive_bonus", x_velocity=99.0, z_position=-5.0) == 1.0


def test_features_return_python_floats():
    # Reward layer downstream will sum these; numpy scalars sneak through
    # silently and break json serialization of summary.json. Pin the type.
    for name, fn in hc.FEATURES.items():
        v = fn(DUMMY_OBS, DUMMY_ACTION, DUMMY_NEXT_OBS, info())
        assert isinstance(v, float), f"feature {name!r} returned {type(v).__name__}"
