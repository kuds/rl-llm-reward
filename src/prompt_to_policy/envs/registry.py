"""Environment registry.

Each supported MuJoCo env exposes a uniform interface: a make_env
callable, a feature registry, per-feature docstrings, and a PPO config.
The registry maps a short name (``halfcheetah``, ``hopper``, ``ant``) to
that bundle so the training harness, CLI, and prompt builder don't have
to special-case envs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym

from prompt_to_policy.reward import FeatureFn

PromptTemplateFn = Callable[[dict[str, str]], str]


@dataclass(frozen=True)
class EnvSpec:
    """Everything a training run needs to know about an env."""

    name: str
    env_id: str
    make_env: Callable[..., gym.Env]
    features: dict[str, FeatureFn]
    feature_docs: dict[str, str]
    build_system_prompt: PromptTemplateFn


def get(name: str) -> EnvSpec:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown env {name!r}. Available: {available}")
    return _REGISTRY[name]


def list_envs() -> list[str]:
    return sorted(_REGISTRY)


# Populated by ``envs/__init__.py`` once all env modules are imported.
_REGISTRY: dict[str, EnvSpec] = {}


def _register(spec: EnvSpec) -> None:
    _REGISTRY[spec.name] = spec
