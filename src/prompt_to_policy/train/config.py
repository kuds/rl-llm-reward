"""Fixed PPO hyperparameters per env.

The LLM never touches these. Per ADR 0001 / CLAUDE.md, the reward is the
single independent variable across runs. Hyperparameters come from
SB3 RL Zoo (HalfCheetah-v4), which is the closest thing to a standard
recipe an SB3 user would reach for; HalfCheetah-v5 is dynamics-equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HalfCheetahPPOConfig:
    env_id: str = "HalfCheetah-v5"

    # Budget
    total_timesteps: int = 1_000_000
    quick_timesteps: int = 200_000

    # Vectorization
    n_envs: int = 1

    # PPO core (rl-zoo3 HalfCheetah-v4)
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

    # Observation/reward normalization (rl-zoo3 normalize: true)
    normalize_obs: bool = True
    normalize_reward: bool = True

    # Final-policy evaluation
    eval_episodes: int = 5
    video_length_steps: int = 1000


HALFCHEETAH_PPO = HalfCheetahPPOConfig()
