"""Shared system prompt scaffolding.

Each per-env template plugs an ``env_blurb`` (one paragraph describing
the morphology and termination conditions) and a feature list into a
common skeleton. The skeleton describes the JSON schema, the reward
arithmetic, and the constraints the LLM cannot change. Per-env templates
add their own few-shot examples on top.
"""

from __future__ import annotations

PROMPT_VERSION = "v2"


_HEADER = """\
You generate reward specifications for a reinforcement-learning training pipeline.

The user describes a desired behavior in natural language. You output a JSON reward spec \
that, when used to train PPO on a fixed {ENV_HUMAN} environment, produces that behavior.

Constraints you cannot change:
- The environment dynamics, action space, and observation space.
- The training algorithm (PPO), its hyperparameters, the network architecture, and the
  total step budget.

Only the reward changes per prompt. Pick the smallest set of features that expresses the
intent.

Output schema (and only this schema; unknown fields are rejected):

  {{
    "components": [
      {{"feature": "<name>", "weight": <float>}},
      ...
    ],
    "bias": <float>          // optional, default 0.0
  }}

The reward at each environment step is:

  reward(step) = bias + sum(weight_i * feature_i(step))

{ENV_BLURB}

Available features:

{FEATURES}

General guidance:
- Positive weights reward the feature; negative weights penalize. Omit a feature to ignore it.
- A small control_cost penalty (e.g. weight ~ -0.05) usually produces smoother gaits but
  is not required.
- Avoid components that directly cancel each other.
- The training is reward-only: you cannot reset, terminate, or modify the env.

Output exactly one JSON object. No prose, no markdown fences, no commentary.
"""


def render_prompt(
    *,
    env_human: str,
    env_blurb: str,
    feature_docs: dict[str, str],
    examples: str,
) -> str:
    """Render a system prompt with the given env description and feature docs."""
    feature_list = "\n".join(f"  - {name}: {doc}" for name, doc in feature_docs.items())
    header = _HEADER.format(
        ENV_HUMAN=env_human,
        ENV_BLURB=env_blurb.strip(),
        FEATURES=feature_list,
    )
    return f"{header}\nExamples:\n\n{examples.strip()}\n"
