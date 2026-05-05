"""Unit tests for the Ant feature registry."""

from __future__ import annotations

import numpy as np
import pytest

from prompt_to_policy.envs import ant as ant_mod
from prompt_to_policy.envs.ant import _world_z_of_body

DUMMY_OBS = np.zeros(27, dtype=np.float32)
DUMMY_NEXT_OBS = np.zeros(27, dtype=np.float32)
DUMMY_ACTION = np.zeros(8, dtype=np.float32)


def info(**overrides) -> dict:
    base = {
        "x_velocity": 0.0,
        "y_velocity": 0.0,
        "z_velocity": 0.0,
        "z_position": 0.75,
        "upright_projection": 1.0,
    }
    base.update(overrides)
    return base


def call(name: str, *, action=DUMMY_ACTION, **info_kwargs) -> float:
    fn = ant_mod.FEATURES[name]
    return fn(DUMMY_OBS, action, DUMMY_NEXT_OBS, info(**info_kwargs))


def test_registry_contains_expected_feature_names():
    expected = {
        "forward_velocity",
        "lateral_velocity",
        "planar_speed",
        "vertical_velocity",
        "height",
        "torso_uprightness",
        "control_cost",
        "alive_bonus",
    }
    assert set(ant_mod.FEATURES) == expected


def test_every_feature_has_a_docstring_entry():
    assert set(ant_mod.FEATURE_DOCS) == set(ant_mod.FEATURES)


def test_forward_velocity_is_signed():
    assert call("forward_velocity", x_velocity=2.5) == pytest.approx(2.5)
    assert call("forward_velocity", x_velocity=-1.25) == pytest.approx(-1.25)


def test_lateral_velocity_is_signed():
    assert call("lateral_velocity", y_velocity=0.7) == pytest.approx(0.7)
    assert call("lateral_velocity", y_velocity=-0.3) == pytest.approx(-0.3)


def test_planar_speed_combines_x_and_y():
    assert call("planar_speed", x_velocity=3.0, y_velocity=4.0) == pytest.approx(5.0)
    assert call("planar_speed", x_velocity=-1.0, y_velocity=0.0) == pytest.approx(1.0)
    assert call("planar_speed") == 0.0


def test_height_passes_through_z_position():
    assert call("height", z_position=0.42) == pytest.approx(0.42)


def test_torso_uprightness_passes_through_projection():
    assert call("torso_uprightness", upright_projection=1.0) == pytest.approx(1.0)
    assert call("torso_uprightness", upright_projection=-1.0) == pytest.approx(-1.0)


def test_control_cost_is_l2_squared():
    a = np.array([1.0, -1.0, 0.5, 0.0, -0.5, 0.0, 0.25, -0.25], dtype=np.float32)
    assert call("control_cost", action=a) == pytest.approx(float(np.sum(a * a)))


def test_alive_bonus_is_constant_one():
    assert call("alive_bonus") == 1.0


def test_features_return_python_floats():
    for name, fn in ant_mod.FEATURES.items():
        v = fn(DUMMY_OBS, DUMMY_ACTION, DUMMY_NEXT_OBS, info())
        assert isinstance(v, float), f"feature {name!r} returned {type(v).__name__}"


def test_world_z_of_body_identity_quat_is_one():
    """Identity quaternion (w=1) means body is upright."""
    assert _world_z_of_body(1.0, 0.0, 0.0, 0.0) == pytest.approx(1.0)


def test_world_z_of_body_180_about_x_is_minus_one():
    """180-degree rotation about x-axis means upside-down."""
    # Quat (w,x,y,z) = (0, 1, 0, 0) corresponds to a 180-deg rotation about x.
    assert _world_z_of_body(0.0, 1.0, 0.0, 0.0) == pytest.approx(-1.0)


def test_world_z_of_body_90_about_x_is_zero():
    """90-deg about x rolls the body onto its side."""
    # Quat for 90 deg about x: (cos(45), sin(45), 0, 0)
    import math

    s = math.sin(math.pi / 4)
    c = math.cos(math.pi / 4)
    assert _world_z_of_body(c, s, 0.0, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_spec_is_registered():
    from prompt_to_policy import envs

    spec = envs.get("ant")
    assert spec.env_id == "Ant-v5"
    assert spec.features is ant_mod.FEATURES
