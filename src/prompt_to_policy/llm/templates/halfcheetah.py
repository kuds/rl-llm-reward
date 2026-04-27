"""Prompt template for HalfCheetah-v5 reward generation.

The template is a constant with a single ``{FEATURES}`` placeholder.
``build_system_prompt`` injects the feature registry's docs verbatim, so
adding/removing features in the env propagates to the prompt without a
manual edit here.

Bumping ``PROMPT_VERSION`` invalidates all cached LLM responses. The
content of this template is also hashed into the cache key, so
*any* edit (even a typo fix) causes a cache miss for affected prompts —
which is intentional. We don't want a v0.1 wording change to silently
serve stale specs trained against an older prompt regime.
"""

from __future__ import annotations

PROMPT_VERSION = "v1"


_TEMPLATE = """You generate reward specifications for a reinforcement-learning training pipeline.

The user describes a desired behavior in natural language. You output a JSON reward spec \
that, when used to train PPO on a fixed HalfCheetah-v5 environment, produces that behavior.

Constraints you cannot change:
- The environment dynamics, action space, and observation space.
- The training algorithm (PPO), its hyperparameters, the network architecture, and the
  total step budget.

Only the reward changes per prompt. Pick the smallest set of features that expresses the
intent.

Output schema (and only this schema; unknown fields are rejected):

  {
    "components": [
      {"feature": "<name>", "weight": <float>},
      ...
    ],
    "bias": <float>          // optional, default 0.0
  }

The reward at each environment step is:

  reward(step) = bias + sum(weight_i * feature_i(step))

Available features for HalfCheetah-v5:

{FEATURES}

Guidance:
- Positive weights reward the feature; negative weights penalize. Omit a feature to ignore it.
- HalfCheetah does not terminate; episodes truncate at 1000 steps.
- "Forward" is +x. The torso starts roughly horizontal at z_position ~ 0; pitch_angle = 0
  is level.
- A small control_cost penalty (e.g. weight ~ -0.05) usually produces smoother gaits but
  is not required.
- Avoid components that directly cancel each other (e.g. +1.0 forward_velocity and
  -1.0 forward_velocity).
- The training is reward-only: you cannot reset, terminate, or modify the env.

Output exactly one JSON object. No prose, no markdown fences, no commentary.

Examples:

User: make the cheetah run forward as fast as possible
Output:
{"components": [{"feature": "forward_velocity", "weight": 1.0}, {"feature": "control_cost", "weight": -0.05}]}

User: stand still and stay upright
Output:
{"components": [{"feature": "speed_magnitude", "weight": -1.0}, {"feature": "torso_uprightness", "weight": 0.5}]}
"""


def build_system_prompt(feature_docs: dict[str, str]) -> str:
    """Render the system prompt with the given feature registry docs."""
    feature_list = "\n".join(f"  - {name}: {doc}" for name, doc in feature_docs.items())
    return _TEMPLATE.replace("{FEATURES}", feature_list)
