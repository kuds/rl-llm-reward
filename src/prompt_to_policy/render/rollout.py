"""Render a rollout video from a trained model.

Uses imageio + ffmpeg to write an mp4. The env is constructed with
``render_mode="rgb_array"`` and frames are sampled by the wrapper — no
on-screen display, so this works in headless environments and inside
SB3's training loop.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np


def record_rollout(
    model: Any,
    env: gym.Env,
    video_path: Path | str,
    *,
    max_steps: int = 1000,
    fps: int = 25,
    deterministic: bool = True,
    obs_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """Roll out ``model`` in ``env`` for one episode and write an mp4.

    Returns a small dict with ``length``, ``return``, ``video_path``.
    The env must have been built with ``render_mode='rgb_array'``.

    ``obs_transform`` lets callers apply a saved VecNormalize obs
    normalization at inference time.
    """
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    obs, _ = env.reset(seed=0)
    total_return = 0.0
    length = 0

    for _ in range(max_steps):
        model_obs = obs_transform(obs) if obs_transform is not None else obs
        action, _ = model.predict(model_obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        total_return += float(reward)
        length += 1
        if terminated or truncated:
            break

    if not frames:
        raise RuntimeError(
            f"render_mode produced no frames; env must be built with render_mode='rgb_array' "
            f"(got render_mode={getattr(env, 'render_mode', None)!r})"
        )

    imageio.mimsave(video_path, frames, fps=fps, codec="libx264", quality=8)
    return {"length": length, "return": total_return, "video_path": str(video_path)}
