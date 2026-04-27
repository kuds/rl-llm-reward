"""Smoke test for the PPO training harness.

Runs ~1k timesteps end-to-end and confirms the harness writes the
expected non-video artifacts. The pedagogically meaningful smoke test
('run forward' reaches a reasonable return at 200k steps) lives in
``examples/smoke_train_forward.py`` because it's too slow for pytest.

Video rendering is exercised by the manual examples/ script too: it
requires an OpenGL backend (egl/osmesa/X11) which this CI-style test
environment may not have. The fast path here uses
``record_video=False``; the video-rendering test below gates on a
runtime probe.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("mujoco")
pytest.importorskip("imageio_ffmpeg")

from prompt_to_policy.reward import RewardSpec  # noqa: E402
from prompt_to_policy.train import HalfCheetahPPOConfig, train  # noqa: E402

SPEC = RewardSpec.model_validate(
    {
        "components": [
            {"feature": "forward_velocity", "weight": 1.0},
            {"feature": "control_cost", "weight": -0.05},
        ]
    }
)


def _can_render_mujoco() -> bool:
    import gymnasium as gym

    try:
        env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    except Exception:
        return False
    try:
        env.reset(seed=0)
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        frame = env.render()
        return frame is not None
    except Exception:
        return False
    finally:
        env.close()


@pytest.mark.slow
def test_harness_runs_briefly_and_writes_artifacts(tmp_path):
    config = HalfCheetahPPOConfig(eval_episodes=1, video_length_steps=40)
    run_dir = tmp_path / "run"

    result = train(
        spec=SPEC,
        prompt="run forward",
        config=config,
        timesteps=1024,
        run_dir=run_dir,
        seed=0,
        record_video=False,
        progress_bar=False,
    )

    assert (run_dir / "model.zip").exists()
    assert (run_dir / "vecnormalize.pkl").exists()
    assert result.summary_path.exists()
    assert result.video_path is None

    summary = json.loads(result.summary_path.read_text())
    assert summary["prompt"] == "run forward"
    assert summary["timesteps"] == 1024
    assert summary["spec"]["components"][0]["feature"] == "forward_velocity"
    assert summary["config"]["env_id"] == "HalfCheetah-v5"
    assert summary["seed"] == 0
    assert len(summary["eval_returns"]) == 1
    assert math.isfinite(summary["final_mean_return"])
    assert math.isfinite(summary["final_mean_length"])
    assert summary["train_seconds"] > 0


@pytest.mark.slow
def test_harness_renders_video_when_renderer_available(tmp_path):
    if not _can_render_mujoco():
        pytest.skip("mujoco renderer unavailable (no egl/osmesa/X11 backend)")

    config = HalfCheetahPPOConfig(eval_episodes=1, video_length_steps=20)
    run_dir = tmp_path / "run"

    result = train(
        spec=SPEC,
        prompt="run forward",
        config=config,
        timesteps=1024,
        run_dir=run_dir,
        seed=0,
        record_video=True,
        progress_bar=False,
    )

    assert result.video_path is not None and result.video_path.exists()
    assert result.video_path.stat().st_size > 0
