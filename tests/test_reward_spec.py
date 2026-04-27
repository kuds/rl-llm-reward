"""Schema-level tests for RewardSpec / RewardComponent."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from prompt_to_policy.reward import RewardComponent, RewardSpec


def test_minimal_spec_parses():
    spec = RewardSpec(components=[RewardComponent(feature="forward_velocity", weight=1.0)])
    assert spec.bias == 0.0
    assert len(spec.components) == 1
    assert spec.components[0].feature == "forward_velocity"
    assert spec.components[0].weight == 1.0


def test_spec_parses_from_json():
    s = """
    {
      "components": [
        {"feature": "forward_velocity", "weight": 1.0},
        {"feature": "control_cost", "weight": -0.05}
      ],
      "bias": 0.5
    }
    """
    spec = RewardSpec.model_validate_json(s)
    assert spec.bias == 0.5
    assert [c.feature for c in spec.components] == ["forward_velocity", "control_cost"]
    assert spec.components[1].weight == -0.05


def test_empty_components_rejected():
    with pytest.raises(ValidationError):
        RewardSpec(components=[])


def test_unknown_top_level_field_rejected():
    with pytest.raises(ValidationError):
        RewardSpec.model_validate(
            {
                "components": [{"feature": "forward_velocity", "weight": 1.0}],
                "scale": 2.0,
            }
        )


def test_unknown_component_field_rejected():
    with pytest.raises(ValidationError):
        RewardSpec.model_validate(
            {"components": [{"feature": "forward_velocity", "weight": 1.0, "transform": "abs"}]}
        )


def test_blank_feature_name_rejected():
    with pytest.raises(ValidationError):
        RewardComponent(feature="", weight=1.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_weight_rejected(bad: float):
    with pytest.raises(ValidationError):
        RewardComponent(feature="forward_velocity", weight=bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_bias_rejected(bad: float):
    with pytest.raises(ValidationError):
        RewardSpec(
            components=[RewardComponent(feature="forward_velocity", weight=1.0)],
            bias=bad,
        )


def test_negative_weight_allowed():
    # Reward hacking story #1: LLM picks the right feature with the wrong sign.
    # The DSL must allow it; semantics are the LLM's problem, not the schema's.
    spec = RewardSpec(components=[RewardComponent(feature="forward_velocity", weight=-1.0)])
    assert spec.components[0].weight == -1.0
