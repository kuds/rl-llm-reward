# Prompt → Policy

Train a HalfCheetah reinforcement-learning policy from a natural-language prompt.
Companion code for a *Finding Theta* blog post on LLM-generated rewards.

```
"make the cheetah run forward as fast as possible"
        │
        ▼
   Anthropic Claude  →  JSON reward spec
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
| HalfCheetah-v5 from `gymnasium[mujoco]`              | A research benchmark or RL-zoo replacement  |

## Quickstart

The fastest way to try it is the Colab notebook — no install, no API key needed
for the quick path.

[**Open the Colab notebook**](examples/colab_demo.ipynb) (or download and upload
to colab.research.google.com).

For local development:

```bash
git clone https://github.com/kuds/rl-llm-reward.git
cd rl-llm-reward
pip install -e ".[dev]"
pytest -q                              # 87 fast tests; runs in ~2s
```

### Run a training without the LLM (no API key required)

```bash
p2p train-spec examples/specs/forward_locomotion.json --quick
```

`--quick` is the 200k-step development budget. Default budget is 1M steps. Output
artifacts (model, VecNormalize stats, rollout video, `summary.json`) land in
`runs/<auto-id>/`.

### Full end-to-end with an LLM-generated reward

```bash
export ANTHROPIC_API_KEY=...
p2p run "make the cheetah run forward as fast as possible" --quick
```

The first time you run a given prompt the LLM is called and the response is
cached in `examples/cached_responses/`. Re-running the same prompt is free and
deterministic.

### Just see the reward, without training

```bash
p2p generate "stand still and stay upright"
```

Prints the JSON spec and the estimated cost. Same caching behavior.

## How it works

```
src/prompt_to_policy/
├── envs/halfcheetah.py         # env wrapper + per-step feature registry
├── reward/                     # Pydantic schema + builder + smoke test
├── llm/
│   ├── client.py               # LLMRewardClient with on-disk cache
│   ├── pricing.py              # USD cost estimator (token-based)
│   └── templates/halfcheetah.py # system prompt + few-shot examples
├── train/                      # SB3 PPO harness, fixed hyperparameters
├── render/rollout.py           # rgb_array → mp4
└── cli.py                      # tyro-based `p2p` entrypoint
```

The reward DSL is a weighted sum over named per-step features. The LLM emits
JSON like:

```json
{
  "components": [
    {"feature": "forward_velocity", "weight": 1.0},
    {"feature": "control_cost", "weight": -0.05}
  ],
  "bias": 0.0
}
```

Each feature is a pure function `(obs, action, next_obs, info) -> float` defined
in the env module. The current HalfCheetah feature set:

| Name                | Meaning                                                           |
| ------------------- | ----------------------------------------------------------------- |
| `forward_velocity`  | Signed x-velocity (positive = forward)                            |
| `speed_magnitude`   | `|x_velocity|` (always ≥ 0)                                       |
| `vertical_velocity` | Signed z-velocity                                                 |
| `height`            | z-position of the torso                                           |
| `torso_uprightness` | `cos(pitch_angle)`; +1 level, −1 inverted                         |
| `control_cost`      | `sum(action²)`; ≥ 0, use a negative weight to penalize            |
| `alive_bonus`       | Constant 1.0, useful as a bias term                               |

Why a constrained DSL and not free-form Python? See
[`docs/decisions/0001-reward-representation.md`](docs/decisions/0001-reward-representation.md).
Short version: the post's interesting failures are *specification* mistakes, not
sandbox engineering bugs.

## Hyperparameters

PPO hyperparameters are fixed per env in `src/prompt_to_policy/train/config.py`,
taken from `rl-baselines3-zoo`'s HalfCheetah-v4 entry (which targets v5 too).
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
pytest -q                     # fast tests (~2s, 87 tests)
pytest -q -m slow             # adds the 1k-step training smoke (~5s)
ruff check src tests examples # lint
ruff format src tests         # autoformat
```

Project docs:

- [`CLAUDE.md`](CLAUDE.md) — the working contract for this repo (what we're
  building, what's out of scope, the step-by-step plan).
- [`docs/decisions/`](docs/decisions/) — architecture decision records.

Verified end-to-end: a 100k-step `train-spec` run on the forward-locomotion
baseline reaches mean return ≈ 3800 (5 deterministic eval episodes, σ ≈ 22) in
~3 min on a single CPU core. HalfCheetah PPO benchmarks reach ~5000 at 1M
steps, so this is squarely "the harness works" territory.

## Status

v0. Currently HalfCheetah-only. The architecture is set up to add Ant and
Hopper (declare a feature registry, add a prompt template, copy the harness
config). The optional v0.5 revision loop (LLM critiques its own reward after
training) is deferred until v0 produces three convincing example videos.

Out of scope for this demo, on the roadmap for follow-up posts:

- Free-form Python rewards with sandboxed execution
- Visual feedback (rendered frames) as input to the LLM
- Cross-model ablations (Claude vs others) on reward generation quality
