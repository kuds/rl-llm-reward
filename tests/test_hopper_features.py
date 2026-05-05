"""Unit tests for the Hopper feature registry.

Features are pure functions of ``(obs, action, next_obs, info)``. These
tests exercise each feature with hand-built info dicts; no MuJoCo
install is required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from prompt_to_policy.envs import hopper as hp

DUMMY_OBS = np.zeros(11, dtype=np.float32)
DUMMY_NEXT_OBS = np.zeros(11, dtype=np.float32)
DUMMY_ACTION = np.zeros(3, dtype=np.float32)


def info(**overrides) -> dict:
    base = {
        "x_velocity": 0.0,
        "z_velocity": 0.0,
        "z_position": 0.0,
        "pitch_angle": 0.0,
        "pitch_velocity": 0.0,
    }
    base.update(overrides)
    return base


def call(name: str, *, action=DUMMY_ACTION, **info_kwargs) -> float:
    fn = hp.FEATURES[name]
    return fn(DUMMY_OBS, action, DUMMY_NEXT_OBS, info(**info_kwargs))


def test_registry_contains_expected_feature_names():
    expected = {
        "forward_velocity",
        "speed_magnitude",
        "vertical_velocity",
        "height",
        "torso_uprightness",
        "pitch_velocity",
        "control_cost",
        "alive_bonus",
    }
    assert set(hp.FEATURES) == expected


def test_every_feature_has_a_docstring_entry():
    assert set(hp.FEATURE_DOCS) == set(hp.FEATURES)
    for name, doc in hp.FEATURE_DOCS.items():
        assert doc.strip(), f"feature {name!r} has empty doc"


def test_forward_velocity_is_signed():
    assert call("forward_velocity", x_velocity=2.5) == pytest.approx(2.5)
    assert call("forward_velocity", x_velocity=-1.25) == pytest.approx(-1.25)


def test_speed_magnitude_is_absolute_value():
    assert call("speed_magnitude", x_velocity=2.5) == pytest.approx(2.5)
    assert call("speed_magnitude", x_velocity=-1.25) == pytest.approx(1.25)


def test_height_passes_through_z_position():
    assert call("height", z_position=1.25) == pytest.approx(1.25)


def test_torso_uprightness_is_cos_pitch():
    assert call("torso_uprightness", pitch_angle=0.0) == pytest.approx(1.0)
    assert call("torso_uprightness", pitch_angle=math.pi / 2) == pytest.approx(0.0, abs=1e-9)
    assert call("torso_uprightness", pitch_angle=math.pi) == pytest.approx(-1.0)


def test_pitch_velocity_is_signed():
    assert call("pitch_velocity", pitch_velocity=0.7) == pytest.approx(0.7)
    assert call("pitch_velocity", pitch_velocity=-0.3) == pytest.approx(-0.3)


def test_control_cost_is_l2_squared_and_nonnegative():
    assert call("control_cost", action=np.zeros(3, dtype=np.float32)) == 0.0
    a = np.array([1.0, -0.5, 0.25], dtype=np.float32)
    assert call("control_cost", action=a) == pytest.approx(float(np.sum(a * a)))


def test_alive_bonus_is_constant_one():
    assert call("alive_bonus") == 1.0


def test_features_return_python_floats():
    for name, fn in hp.FEATURES.items():
        v = fn(DUMMY_OBS, DUMMY_ACTION, DUMMY_NEXT_OBS, info())
        assert isinstance(v, float), f"feature {name!r} returned {type(v).__name__}"


def test_spec_is_registered():
    from prompt_to_policy import envs

    spec = envs.get("hopper")
    assert spec.env_id == "Hopper-v5"
    assert spec.features is hp.FEATURES
