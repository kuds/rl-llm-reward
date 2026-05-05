"""Prompt template for Ant-v5 reward generation."""

from __future__ import annotations

from ._base import PROMPT_VERSION, render_prompt

__all__ = ["PROMPT_VERSION", "build_system_prompt"]


_ENV_BLURB = """\
Ant-v5 specifics:
- 3D quadruped with eight legs joints. Default torso height ~0.75m.
- Locomotion is two-dimensional: "forward" is +x, "lateral" is +y.
- Ant TERMINATES the episode if z_position leaves the healthy range [0.2, 1.0] (the body
  has fallen or jumped). alive_bonus and torso_uprightness keep the policy in the healthy
  state distribution.
- torso_uprightness is the projection of the body's local +z axis onto world +z, computed
  from the torso quaternion. +1 fully upright, 0 on its side, -1 upside down.
"""


_EXAMPLES = """\
User: walk forward steadily without falling over
Output:
{"components": [{"feature": "forward_velocity", "weight": 1.0}, {"feature": "alive_bonus", "weight": 1.0}, {"feature": "torso_uprightness", "weight": 0.5}, {"feature": "control_cost", "weight": -0.05}]}

User: spin clockwise in place
Output:
{"components": [{"feature": "planar_speed", "weight": -1.0}, {"feature": "alive_bonus", "weight": 1.0}, {"feature": "torso_uprightness", "weight": 0.5}]}
"""


def build_system_prompt(feature_docs: dict[str, str]) -> str:
    return render_prompt(
        env_human="Ant-v5",
        env_blurb=_ENV_BLURB,
        feature_docs=feature_docs,
        examples=_EXAMPLES,
    )
