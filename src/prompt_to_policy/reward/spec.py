"""Pydantic schema for the reward DSL.

Per ADR 0001, an LLM-emitted reward is a weighted sum over a per-env
feature registry, plus an optional bias. The schema is intentionally
narrow: no per-component transforms, no nesting. If a prompt seems to
need a transformed feature ("stay near height 1.0"), define it as a
named feature in the env, not as a DSL operator.

Example
-------
>>> spec = RewardSpec.model_validate_json('''
... {
...   "components": [
...     {"feature": "forward_velocity", "weight": 1.0},
...     {"feature": "control_cost", "weight": -0.05}
...   ],
...   "bias": 0.0
... }
... ''')
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RewardComponent(BaseModel):
    """One term in the weighted sum: ``weight * feature(obs, action, next_obs, info)``."""

    model_config = ConfigDict(extra="forbid")

    feature: str = Field(min_length=1)
    weight: float

    @field_validator("weight")
    @classmethod
    def _weight_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"weight must be finite, got {v!r}")
        return v


class RewardSpec(BaseModel):
    """A reward as a weighted sum over named features plus a bias."""

    model_config = ConfigDict(extra="forbid")

    components: list[RewardComponent] = Field(min_length=1)
    bias: float = 0.0

    @field_validator("bias")
    @classmethod
    def _bias_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"bias must be finite, got {v!r}")
        return v
