"""Manual smoke test: train HalfCheetah on 'run forward' for 200k steps.

Per CLAUDE.md step 6: confirms the harness reaches a reasonable return
on a hand-written reward, before any LLM is in the loop. The reward
spec is the canonical "forward locomotion" baseline.

This is *not* in the pytest suite because 200k steps takes several
minutes on CPU (and ~2 min on a single GPU). Run it manually:

    python examples/smoke_train_forward.py

Reasonable-return target: HalfCheetah PPO on the standard
forward-velocity reward should reach a final mean return well above
500 by 200k steps with the rl-zoo3 hyperparameters; near-baseline
PPO benchmarks reach ~5000 at 1M steps. We're not chasing SOTA — we
just want a clearly non-flat learning curve and a video that visibly
runs forward.
"""

from __future__ import annotations

import time
from pathlib import Path

from prompt_to_policy.reward import RewardSpec
from prompt_to_policy.train import train


SPEC = RewardSpec.model_validate(
    {
        "components": [
            {"feature": "forward_velocity", "weight": 1.0},
            {"feature": "control_cost", "weight": -0.05},
        ]
    }
)
PROMPT = "make the cheetah run forward as fast as possible"
TIMESTEPS = 200_000


def main() -> None:
    print(f"prompt: {PROMPT!r}")
    print(f"spec  : {SPEC.model_dump_json()}")
    print(f"steps : {TIMESTEPS}")
    t0 = time.time()
    result = train(
        spec=SPEC,
        prompt=PROMPT,
        timesteps=TIMESTEPS,
        run_dir=Path("runs/smoke_forward"),
        seed=0,
        record_video=True,
        progress_bar=False,
    )
    elapsed = time.time() - t0
    print(f"\n--- done in {elapsed:.1f}s ---")
    print(f"final_mean_return : {result.final_mean_return:.1f}")
    print(f"final_mean_length : {result.final_mean_length:.1f}")
    print(f"eval_episodes     : {result.eval_episodes}")
    print(f"run_dir           : {result.run_dir}")
    print(f"video_path        : {result.video_path}")
    print(f"summary_path      : {result.summary_path}")


if __name__ == "__main__":
    main()
