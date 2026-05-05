"""Fixed PPO hyperparameters per env.

The LLM never touches these. Per ADR 0001, the reward is the single
independent variable across runs. Hyperparameters come from SB3 RL Zoo
(HalfCheetah-v4, Hopper-v4, Ant-v4) — the closest thing to a standard
recipe an SB3 user would reach for. Gymnasium's v5 envs are
dynamics-equivalent to their v4 counterparts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    """Generic per-env PPO config. Concrete instances live below.

    ``env_id`` defaults to HalfCheetah-v5 for backwards compatibility with
    callers that constructed ``HalfCheetahPPOConfig(...)`` without an env id;
    Hopper and Ant configs override it explicitly.
    """

    env_id: str = "HalfCheetah-v5"

    # Budget
    total_timesteps: int = 1_000_000
    quick_timesteps: int = 200_000

    # Vectorization
    n_envs: int = 1

    # PPO core
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 20
    learning_rate: float = 2.0633e-05
    gae_lambda: float = 0.92
    gamma: float = 0.98
    clip_range: float = 0.1
    ent_coef: float = 0.000401762
    vf_coef: float = 0.58096
    max_grad_norm: float = 0.8

    # Observation/reward normalization
    normalize_obs: bool = True
    normalize_reward: bool = True

    # Final-policy evaluation
    eval_episodes: int = 5
    video_length_steps: int = 1000


# rl-zoo3 HalfCheetah-v4 entry.
HALFCHEETAH_PPO = PPOConfig(env_id="HalfCheetah-v5")

# rl-zoo3 Hopper-v4 entry.
HOPPER_PPO = PPOConfig(
    env_id="Hopper-v5",
    n_envs=1,
    n_steps=512,
    batch_size=32,
    n_epochs=20,
    learning_rate=9.80828e-05,
    gae_lambda=0.99,
    gamma=0.999,
    clip_range=0.2,
    ent_coef=0.00229519,
    vf_coef=0.835671,
    max_grad_norm=0.7,
)

# rl-zoo3 Ant-v4 entry. Ant typically benefits from a slightly larger budget; we keep the
# same 1M default but the user can override via --timesteps.
ANT_PPO = PPOConfig(
    env_id="Ant-v5",
    n_envs=1,
    n_steps=512,
    batch_size=32,
    n_epochs=10,
    learning_rate=3.0e-04,
    gae_lambda=0.95,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
)


# Registry by env short-name. Keys must match ``envs/registry.EnvSpec.name``.
PPO_CONFIGS: dict[str, PPOConfig] = {
    "halfcheetah": HALFCHEETAH_PPO,
    "hopper": HOPPER_PPO,
    "ant": ANT_PPO,
}


def get_config(env_name: str) -> PPOConfig:
    if env_name not in PPO_CONFIGS:
        available = ", ".join(sorted(PPO_CONFIGS))
        raise KeyError(f"no PPO config for env {env_name!r}. Available: {available}")
    return PPO_CONFIGS[env_name]


# Backwards-compatible alias for ``HalfCheetahPPOConfig`` — kept so older
# imports keep working while tests are migrated.
HalfCheetahPPOConfig = PPOConfig
