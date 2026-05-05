"""Prompt template for Hopper-v5 reward generation."""

from __future__ import annotations

from ._base import PROMPT_VERSION, render_prompt

__all__ = ["PROMPT_VERSION", "build_system_prompt"]


_ENV_BLURB = """\
Hopper-v5 specifics:
- 2D planar one-legged hopper. Default torso height ~1.25m.
- "Forward" is +x; pitch_angle = 0 is upright (vertical torso).
- Hopper TERMINATES the episode if z_position drops below ~0.7m or pitch goes far from
  upright. Encouraging staying-alive (alive_bonus, height, torso_uprightness) is usually
  necessary or the policy collapses immediately.
- Per-step physics is one short bounce; sustained "hopping" is the natural locomotion mode.
"""


_EXAMPLES = """\
User: hop forward as fast as possible without falling
Output:
{"components": [{"feature": "forward_velocity", "weight": 1.0}, {"feature": "alive_bonus", "weight": 1.0}, {"feature": "control_cost", "weight": -0.001}]}

User: hop in place as high as you can
Output:
{"components": [{"feature": "height", "weight": 1.0}, {"feature": "speed_magnitude", "weight": -0.5}, {"feature": "alive_bonus", "weight": 1.0}]}
"""


def build_system_prompt(feature_docs: dict[str, str]) -> str:
    return render_prompt(
        env_human="Hopper-v5",
        env_blurb=_ENV_BLURB,
        feature_docs=feature_docs,
        examples=_EXAMPLES,
    )
