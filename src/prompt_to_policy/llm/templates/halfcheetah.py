"""Prompt template for HalfCheetah-v5 reward generation."""

from __future__ import annotations

from ._base import PROMPT_VERSION, render_prompt

__all__ = ["PROMPT_VERSION", "build_system_prompt"]


_ENV_BLURB = """\
HalfCheetah-v5 specifics:
- 2D planar morphology with two legs; the body lies roughly horizontal at z_position ~ 0.
- "Forward" is +x; pitch_angle = 0 is level.
- HalfCheetah does NOT terminate on falls. Episodes truncate at 1000 steps.
- alive_bonus is a constant 1.0 — useful only as a bias term (e.g. to keep returns positive).
"""


_EXAMPLES = """\
User: make the cheetah run forward as fast as possible
Output:
{"components": [{"feature": "forward_velocity", "weight": 1.0}, {"feature": "control_cost", "weight": -0.05}]}

User: stand still and stay upright
Output:
{"components": [{"feature": "speed_magnitude", "weight": -1.0}, {"feature": "torso_uprightness", "weight": 0.5}]}
"""


def build_system_prompt(feature_docs: dict[str, str]) -> str:
    return render_prompt(
        env_human="HalfCheetah-v5",
        env_blurb=_ENV_BLURB,
        feature_docs=feature_docs,
        examples=_EXAMPLES,
    )
