"""Reward DSL: schema, builder, smoke test.

See ``docs/decisions/0001-reward-representation.md`` for the design.
"""

from .build import (
    FeatureFn,
    RewardFn,
    RewardSmokeError,
    UnknownFeatureError,
    build_reward_fn,
    smoke_test_reward_fn,
)
from .spec import RewardComponent, RewardSpec

__all__ = [
    "FeatureFn",
    "RewardComponent",
    "RewardFn",
    "RewardSmokeError",
    "RewardSpec",
    "UnknownFeatureError",
    "build_reward_fn",
    "smoke_test_reward_fn",
]
