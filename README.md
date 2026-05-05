# Prompt → Policy

Train a MuJoCo reinforcement-learning policy from a natural-language prompt.
Companion code for a *Finding Theta* blog post on LLM-generated rewards.

```
"hop forward as fast as possible without falling"
        │
        ▼
   Claude / Gemini / local LLM  →  JSON reward spec
        │
        ▼
  PPO (fixed hyperparameters, fixed env)  →  trained policy
        │
        ▼
                rollout video + summary.json
```

The environment dynamics, action space, and training hyperparameters are **fixed**.
Only the reward changes per prompt. The pedagogical claim of the post: RL is
fundamentally reward specification, LLMs are surprisingly good at it, and the
optimization half still bites.

## What this is, and isn't

| Is                                                   | Isn't                                       |
| ---------------------------------------------------- | ------------------------------------------- |
| A small, faithful pedagogical demo                   | A general LLM-as-policy framework           |
| LLM emits a reward; PPO does the rest                | LLM never acts in the env at runtime        |
| One prompt → one training run → one policy           | Multi-task, prompt-conditioned policies     |
| Reward is a constrained DSL ([ADR 0001](docs/decisions/0001-reward-representation.md)) | Free-form Python in a sandbox |
| HalfCheetah / Hopper / Ant from `gymnasium[mujoco]`  | A research benchmark or RL-zoo replacement  |

## Supported envs

| Env name       | Gymnasium ID    | Morphology                  | Good for prompts like…                   |
| -------------- | --------------- | --------------------------- | ---------------------------------------- |
| `halfcheetah`  | HalfCheetah-v5  | 2D, two legs, never falls   | "run forward / backward", "stand still"  |
| `hopper`       | Hopper-v5       | 2D, **one foot**, terminates on fall | "hop forward", "hop in place high"       |
| `ant`          | Ant-v5          | 3D, four legs, terminates on fall    | "walk forward", "spin in place", "stand"|

Each env has its own feature registry. See `src/prompt_to_policy/envs/<env>.py` for
the exact list (e.g. Ant exposes `lateral_velocity` and `planar_speed`; Hopper
exposes `pitch_velocity` for "stay stable" / "flip" prompts).

## Supported providers

| Provider     | Setup                                          | When to use it                              |
| ------------ | ---------------------------------------------- | ------------------------------------------- |
| `anthropic`  | `pip install prompt-to-policy`, set `ANTHROPIC_API_KEY` | Default. Strong reward generation; small per-run cost. |
| `gemini`     | `pip install "prompt-to-policy[gemini]"`, set `GEMINI_API_KEY` | Cheaper API alternative, useful for ablations.  |
| `local`      | `pip install "prompt-to-policy[local]"` + a GPU | No API spend; runs a quantized HF model on your GPU. Best on Colab A100/L4; works on T4 in 4-bit. |

Pick at the CLI: `--provider {anthropic,gemini,local}`. No auto-detection — the
choice is explicit so a Colab cell that worked yesterday doesn't silently pick a
different backend tomorrow.

## Quickstart

The fastest way to try it is the Colab notebook — no install, no API key needed
for the smoke-test path.

[**Open the Colab notebook**](examples/colab_demo.ipynb) (or download and upload
to colab.research.google.com).

For local development:

```bash
git clone https://github.com/kuds/rl-llm-reward.git
cd rl-llm-reward
pip install -e ".[dev]"
pytest -q                              # ~120 fast tests, runs in seconds
```

### Run a training without the LLM (no API key required)

```bash
p2p train-spec --env hopper examples/specs/hopper_forward.json --quick
```

`--quick` is the per-env quick development budget (200k steps for HalfCheetah/Hopper,
similar for Ant). Default budget is 1M steps. Output artifacts (model,
VecNormalize stats, rollout video, `summary.json`) land in `runs/<auto-id>/`.

### End-to-end with Claude

```bash
export ANTHROPIC_API_KEY=...
p2p run --env hopper "hop forward as fast as possible without falling" --quick
```

### End-to-end with Gemini

```bash
pip install "prompt-to-policy[gemini]"
export GEMINI_API_KEY=...
p2p run --env ant --provider gemini "walk forward steadily without falling over" --quick
```

### End-to-end with a local LLM (Colab GPU recommended)

```bash
pip install "prompt-to-policy[local]"
p2p run --env hopper --provider local "hop in place as high as you can" --quick
```

The first time you run a given prompt the LLM is called and the response is
cached in `examples/cached_responses/`. Re-running the same `(env, provider, prompt)`
is free and deterministic.

### Just see the reward, without training

```bash
p2p generate --env ant "spin clockwise in place"
```

Prints the JSON spec and the estimated cost. Same caching behavior.

## How it works

```
src/prompt_to_policy/
├── envs/
│   ├── registry.py           # EnvSpec + lookup by short name
│   ├── halfcheetah.py        # env wrapper + feature registry
│   ├── hopper.py             # env wrapper + feature registry
│   └── ant.py                # env wrapper + feature registry
├── reward/                   # Pydantic schema + builder + smoke test
├── llm/
│   ├── client.py             # BaseRewardClient + Anthropic implementation
│   ├── gemini_client.py      # Gemini implementation
│   ├── local_client.py       # HuggingFace local-model implementation
│   ├── pricing.py            # USD cost estimator (token-based, Claude + Gemini)
│   └── templates/            # one prompt template per env
├── train/                    # SB3 PPO harness + per-env hyperparameters
├── render/rollout.py         # rgb_array → mp4
└── cli.py                    # tyro-based `p2p` entrypoint
```

The reward DSL is a weighted sum over named per-step features. The LLM emits
JSON like:

```json
{
  "components": [
    {"feature": "forward_velocity", "weight": 1.0},
    {"feature": "alive_bonus", "weight": 1.0},
    {"feature": "control_cost", "weight": -0.001}
  ],
  "bias": 0.0
}
```

Each feature is a pure function `(obs, action, next_obs, info) -> float` defined
in the env module. Different envs expose different features — Ant has
`upright_projection` (3D quaternion-based) where Hopper has `pitch_velocity`
(planar). The full lists live in `src/prompt_to_policy/envs/<env>.py`.

Why a constrained DSL and not free-form Python? See
[`docs/decisions/0001-reward-representation.md`](docs/decisions/0001-reward-representation.md).
Short version: the post's interesting failures are *specification* mistakes, not
sandbox engineering bugs.

## Hyperparameters

PPO hyperparameters are fixed per env in `src/prompt_to_policy/train/config.py`,
taken from `rl-baselines3-zoo`'s v4 entries (which target v5 too).
**They are not LLM-tunable**: the reward is the single independent variable
across runs.

This is a feature, not a limitation — it isolates the variable the demo is about.

## Headless rendering

MuJoCo's `rgb_array` rendering needs an OpenGL backend. On Colab GPU runtimes
EGL works out of the box (`MUJOCO_GL=egl`, set automatically by the notebook).
Locally, install GLFW or use OSMesa:

```bash
sudo apt install libglfw3 libosmesa6
export MUJOCO_GL=egl                  # or osmesa
```

If video rendering is unavailable, pass `--no-video` to skip it. Training and
metrics still work.

## Development

```bash
pytest -q                     # fast tests, runs in seconds
pytest -q -m slow             # adds the 1k-step training smoke
ruff check src tests examples # lint
ruff format src tests         # autoformat
```

Project docs:

- [`docs/decisions/`](docs/decisions/) — architecture decision records.
- [`examples/specs/`](examples/specs/) — hand-written reward baselines per env.

## Status

v0. All three envs (HalfCheetah, Hopper, Ant) and all three providers
(Anthropic, Gemini, local HF) are wired up. The optional v0.5 revision loop
(LLM critiques its own reward after training) is deferred until v0 produces
three convincing example videos per env.

Out of scope for this demo, on the roadmap for follow-up posts:

- Free-form Python rewards with sandboxed execution
- Visual feedback (rendered frames) as input to the LLM
- Cross-model ablations (Claude vs Gemini vs local) on reward generation quality
