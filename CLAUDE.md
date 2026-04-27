This file is the entry point for Claude Code working in this repo. Read it fully before making any changes. Update it when assumptions change.

## Project goal

Build an educational demo, in service of a Finding Theta blog post, that converts natural-language prompts into trained reinforcement learning policies for fixed-physics MuJoCo continuous-control tasks.

The pipeline is:

```
user prompt  →  LLM emits reward spec  →  fixed training harness trains PPO
            →  rollout video + return curve  →  (optional) LLM revises spec
```

The environment dynamics are **fixed**. Only the reward changes per prompt. This constraint is intentional and pedagogical: the post argues that RL is fundamentally reward specification, and that LLMs are surprisingly competent at specification but the optimization half still bites.

This is a demo, not a framework. Optimize for clarity, reproducibility on a single GPU (or CPU for small runs), and a small number of compelling examples. Do not generalize prematurely.

## Non-goals

- Not a general "LLM-as-policy" system. The LLM never acts in the env at runtime.
- Not a system that modifies env XML, dynamics, hyperparameters, or network architecture. Only reward.
- Not a research benchmark. Eureka and Text2Reward already exist; this is a stripped-down pedagogical sibling.
- Not multi-task or prompt-conditioned policies. One prompt → one training run → one policy.

## Target tasks (v0)

Pick from `gymnasium[mujoco]` so the install is one line and the physics is familiar to readers:

- `HalfCheetah-v5` — supports "run forward", "run backward", "hop", "stand still", "flip"
- `Ant-v5` — supports directional locomotion, standing, spinning
- `Hopper-v5` — supports "hop high", "hop slow", "stand still"

Start with HalfCheetah only. Add Ant and Hopper once the loop works end-to-end.

## Open design decision: reward representation

You (Claude Code) should choose between three options for v0 and document the choice in `docs/decisions/0001-reward-representation.md` before writing implementation code. The options:

1. **Constrained DSL** — LLM emits a weighted sum over a fixed feature dict per env (e.g. `forward_vel`, `height`, `energy`, `feet_contact`, `orientation`). Safest, most reliable, easiest to visualize in the post.
2. **Free-form Python** — LLM emits a Python function `reward(obs, action, next_obs, info) -> float`, executed in a sandbox. Most expressive, most brittle.
3. **DSL with Python escape hatch** — DSL by default; allow a `python:` block when the DSL is insufficient. More implementation work but the most honest pedagogically.

Recommended starting point: option 1 for v0, with the architecture written so option 3 is a clean extension. But evaluate the tradeoffs yourself given the goal (one blog post, 4–5 example prompts, faithful to how RL practitioners actually think about reward design) and write the ADR before committing to code.

When evaluating, weigh: implementation cost, failure modes the reader will see, how the choice affects the "reward hacking" stories the post wants to tell, and whether the DSL features can be defined cleanly per env without leaking abstractions.

## Repo structure (target)

```
prompt-to-policy/
├── CLAUDE.md                       # this file
├── README.md                       # human-facing, written last
├── pyproject.toml
├── docs/
│   └── decisions/                  # ADRs (architecture decision records)
├── src/prompt_to_policy/
│   ├── envs/                       # task definitions + feature extractors
│   │   ├── halfcheetah.py
│   │   ├── ant.py
│   │   └── hopper.py
│   ├── reward/                     # DSL parser, validator, reward fn factory
│   ├── llm/                        # prompt templates, Anthropic API client
│   ├── train/                      # SB3 PPO harness, fixed hyperparams per env
│   ├── render/                     # rollout videos, return curves
│   └── cli.py                      # `p2p run "make the cheetah hop"`
├── examples/
│   └── prompts.md                  # the 4–5 prompts the blog post uses
├── runs/                           # gitignored; training outputs
└── tests/
```

## Training harness constraints

- Use Stable-Baselines3 PPO. The user knows SB3 cold; do not introduce RLlib or CleanRL without a strong reason.
- Hyperparameters are **fixed per env** in a config file. The LLM never touches them. This is a feature, not a limitation — it isolates the reward as the single independent variable.
- Default budget: 1M timesteps per run on HalfCheetah. Should complete in roughly 10–20 minutes on a single modern GPU. If iteration feels slow, add a `--quick` flag (200k steps) for development.
- Log to TensorBoard and a simple JSON sidecar (`runs/<run_id>/summary.json`) with the prompt, reward spec, final mean return, and video path.
- Always render a rollout video at the end of training. The video is the deliverable — the post is going to embed these.

## LLM integration

- Use the Anthropic Python SDK. Model: `claude-opus-4-7` for reward generation by default; expose `--model` flag for ablations.
- Prompt template lives in `src/prompt_to_policy/llm/templates/`. It should:
  - Describe the env's observation and the available reward features (or full env interface, depending on the ADR).
  - Provide 1–2 few-shot examples.
  - Specify exact output schema (JSON for DSL, fenced Python for free-form).
- Validate every LLM output before training: schema check, then a smoke test that evaluates the reward on a random rollout to catch NaN/inf/exceptions. Surface failures clearly; do not silently fall back.
- Cache LLM outputs by prompt hash so re-running an example is free and deterministic.

## Optional v0.5: revision loop

After training, optionally feed back to the LLM:
- Final mean return and per-component reward statistics
- A short text description of the rollout (or a few sampled frames)
- Ask: "does this match the user's intent? if not, propose a revised reward."

Ship v0 without this. Add it only after the basic loop works and produces at least three convincing example videos. The revision loop is the most interesting story for the post but also the highest-variance part to build.

## Coding standards

- Python 3.11+, type hints everywhere, `ruff` + `ruff format`.
- `pytest` for tests. Every module in `reward/` and `envs/` gets unit tests. Training and LLM modules get smoke tests only.
- No development notebooks committed. The one exception is `examples/colab_demo.ipynb` — a reader-facing deliverable for the post, kept thin (install cell + a few cells calling the package). All actual logic lives in `src/` and is reachable from scripts in `examples/`.
- Keep dependencies minimal: `gymnasium[mujoco]`, `stable-baselines3`, `anthropic`, `imageio`, `pydantic`, `tyro` (or `typer`) for the CLI. Resist adding more.
- Determinism where possible: seed everything, log seeds, but accept that MuJoCo + PPO has irreducible variance and document it.

## What to do first

1. Read this file. Read `docs/decisions/` if it exists.
2. Write `docs/decisions/0001-reward-representation.md` choosing among the three options above. Justify the choice in terms of the project goal.
3. Set up `pyproject.toml`, the package skeleton, and a passing `pytest` with one trivial test.
4. Implement the HalfCheetah env wrapper and feature extractor. Unit-test the features.
5. Implement the reward representation chosen in step 2, with validation. Unit-test it on hand-written specs (forward locomotion, backward locomotion, standing).
6. Implement the SB3 PPO training harness with fixed hyperparameters. Smoke-test that "run forward" reaches a reasonable return in 200k steps.
7. Implement the LLM client and prompt template. Smoke-test on three prompts; commit the cached outputs.
8. Wire the CLI. Run end-to-end on "make the cheetah run forward as fast as possible". Save video.
8.5. Verify the Colab notebook (`examples/colab_demo.ipynb`) runs end-to-end on a fresh runtime, both the no-key quick test and the API-keyed full E2E.
9. Stop. Show results to the user before adding Ant, Hopper, or the revision loop.

After each step, run tests and commit. Do not skip ahead. If a step reveals that an earlier assumption was wrong, update this CLAUDE.md and the relevant ADR before continuing.

## Things to flag to the user, not solve silently

- Any time a chosen approach trades pedagogical clarity for engineering convenience.
- Any time the LLM produces a reward that trains successfully but doesn't match the prompt's intent (this is the *good* failure mode — capture it for the post).
- Any time MuJoCo physics or env quirks force a deviation from the design (e.g. observation indices changing between Gymnasium versions).
- Cost: log Anthropic API spend per run in `summary.json`.

## Out of scope, but worth noting for future posts

- Free-form Python rewards with sandbox execution (post 2 of the series).
- Visual feedback loop using rendered frames as input to the LLM (post 3).
- Comparing Claude vs other models on reward generation quality (separate ablation).
- Migrating to MuJoCo Playground / MJX for faster iteration if the demo grows.
