"""SB3 PPO training harness.

Env-agnostic: pick the env by short name (``halfcheetah``, ``hopper``,
``ant``) and the harness pulls the matching ``EnvSpec`` and ``PPOConfig``
from their registries. Hyperparameters are fixed per env; the LLM never
touches them.

The flow:
    1. Build a callable reward from the spec; smoke-test it on real env steps.
    2. Construct a vec env with the DSL reward wrapper and (optional) VecNormalize.
    3. Train PPO with hyperparameters from the matching ``PPOConfig``.
    4. Evaluate the deterministic policy on a fresh non-normalized env, returning
       per-episode returns on the *real* (un-normalized) reward scale.
    5. Render an mp4 rollout video.
    6. Write ``summary.json`` containing prompt, env, spec, config, timing, and metrics.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    import torch.utils.tensorboard  # noqa: F401  — probing import availability

    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False

from prompt_to_policy import envs
from prompt_to_policy.envs.registry import EnvSpec
from prompt_to_policy.render import record_rollout
from prompt_to_policy.reward import RewardSpec, build_reward_fn, smoke_test_reward_fn

from .config import PPO_CONFIGS, PPOConfig
from .wrappers import DSLRewardWrapper

DEFAULT_ENV = "halfcheetah"


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    env: str
    final_mean_return: float
    final_mean_length: float
    eval_episodes: int
    video_path: Path | None
    summary_path: Path
    timesteps: int
    train_seconds: float


def _make_train_env(
    env_spec: EnvSpec,
    reward_fn,
    *,
    n_envs: int,
    normalize_obs: bool,
    normalize_reward: bool,
):
    def make_one():
        env = env_spec.make_env()
        return DSLRewardWrapper(env, reward_fn)

    vec = DummyVecEnv([make_one for _ in range(n_envs)])
    if normalize_obs or normalize_reward:
        vec = VecNormalize(vec, norm_obs=normalize_obs, norm_reward=normalize_reward)
    return vec


def _make_eval_env(env_spec: EnvSpec, reward_fn) -> gym.Env:
    """Single non-normalized env with rgb_array rendering for video and real-scale return."""
    env = env_spec.make_env(render_mode="rgb_array")
    return DSLRewardWrapper(env, reward_fn)


def train(
    *,
    spec: RewardSpec,
    prompt: str,
    env: str = DEFAULT_ENV,
    config: PPOConfig | None = None,
    timesteps: int | None = None,
    seed: int = 0,
    run_dir: Path | str | None = None,
    run_id: str | None = None,
    record_video: bool = True,
    progress_bar: bool = False,
) -> RunResult:
    """Train PPO on the chosen env with an LLM-emitted reward.

    ``env`` selects which env / feature registry / PPO config to use.
    ``config`` overrides the default config for that env. Hyperparameters
    are not LLM-tunable.
    """
    env_spec = envs.get(env)
    if config is None:
        config = PPO_CONFIGS[env]
    if timesteps is None:
        timesteps = config.total_timesteps
    if run_id is None:
        run_id = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    run_dir = Path(run_dir) if run_dir is not None else Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    reward_fn = build_reward_fn(spec, env_spec.features)

    smoke_env = env_spec.make_env()
    try:
        smoke_test_reward_fn(reward_fn, smoke_env, n_steps=8, seed=seed)
    finally:
        smoke_env.close()

    train_vec = _make_train_env(
        env_spec,
        reward_fn,
        n_envs=config.n_envs,
        normalize_obs=config.normalize_obs,
        normalize_reward=config.normalize_reward,
    )

    policy_kwargs = dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = PPO(
        "MlpPolicy",
        train_vec,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        gae_lambda=config.gae_lambda,
        gamma=config.gamma,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir / "tb") if _HAS_TENSORBOARD else None,
        seed=seed,
        verbose=0,
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=progress_bar)
    train_seconds = time.time() - t0

    model_path = run_dir / "model.zip"
    model.save(model_path)
    vecnorm_path: Path | None = None
    if isinstance(train_vec, VecNormalize):
        vecnorm_path = run_dir / "vecnormalize.pkl"
        train_vec.save(str(vecnorm_path))

    # Use the trained VecNormalize to normalize obs at eval time so the policy
    # sees the distribution it was trained on; reward is reported on the
    # un-normalized scale because that's what's pedagogically meaningful.
    def _normalize_obs(obs: np.ndarray) -> np.ndarray:
        if isinstance(train_vec, VecNormalize):
            return train_vec.normalize_obs(obs)
        return obs

    eval_returns: list[float] = []
    eval_lengths: list[int] = []
    eval_env = _make_eval_env(env_spec, reward_fn)
    try:
        for ep_idx in range(config.eval_episodes):
            obs, _ = eval_env.reset(seed=seed + 1 + ep_idx)
            ep_return = 0.0
            ep_length = 0
            while True:
                action, _ = model.predict(_normalize_obs(obs), deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                ep_return += float(reward)
                ep_length += 1
                if terminated or truncated:
                    break
            eval_returns.append(ep_return)
            eval_lengths.append(ep_length)
    finally:
        eval_env.close()

    final_mean_return = float(np.mean(eval_returns))
    final_mean_length = float(np.mean(eval_lengths))

    video_path: Path | None = None
    if record_video:
        video_env = _make_eval_env(env_spec, reward_fn)
        try:
            video_path = run_dir / "rollout.mp4"
            record_rollout(
                model,
                video_env,
                video_path,
                max_steps=config.video_length_steps,
                obs_transform=_normalize_obs if isinstance(train_vec, VecNormalize) else None,
            )
        finally:
            video_env.close()

    train_vec.close()

    summary = {
        "run_id": run_id,
        "env": env,
        "env_id": env_spec.env_id,
        "prompt": prompt,
        "spec": spec.model_dump(),
        "config": asdict(config),
        "timesteps": timesteps,
        "seed": seed,
        "train_seconds": train_seconds,
        "eval_episodes": config.eval_episodes,
        "eval_returns": eval_returns,
        "eval_lengths": eval_lengths,
        "final_mean_return": final_mean_return,
        "final_mean_length": final_mean_length,
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnorm_path) if vecnorm_path else None,
        "video_path": str(video_path) if video_path else None,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    return RunResult(
        run_id=run_id,
        run_dir=run_dir,
        env=env,
        final_mean_return=final_mean_return,
        final_mean_length=final_mean_length,
        eval_episodes=config.eval_episodes,
        video_path=video_path,
        summary_path=summary_path,
        timesteps=timesteps,
        train_seconds=train_seconds,
    )
