# 0001 — Reward representation

- Status: accepted
- Date: 2026-04-27

## Context

CLAUDE.md frames the central pedagogical claim of the blog post: RL is fundamentally reward specification, and LLMs are surprisingly competent at specification but the optimization half still bites. Our pipeline is `prompt → LLM → reward → fixed PPO harness → rollout video`, with HalfCheetah-v5 as the v0 environment and 4–5 example prompts.

The open question: in what form does the LLM emit a reward? Three options were on the table:

1. **Constrained DSL.** LLM emits a structured spec (JSON: weighted sum over a fixed feature dict per env, plus a small set of declared transforms).
2. **Free-form Python.** LLM emits a function `reward(obs, action, next_obs, info) -> float` executed in a sandbox.
3. **DSL with Python escape hatch.** DSL by default, optional Python block when DSL is insufficient.

## Decision

**Option 1: a constrained DSL.** Per-env declared features; LLM output is a JSON document validated by Pydantic; a small registry of feature transforms (identity, abs, square, threshold) lives in code, not in the LLM output.

The architecture should keep the reward layer behind a single `RewardFn = Callable[[obs, action, next_obs, info], float]` interface, so option 3 can be added later without touching the training harness or the env wrappers.

## Justification

Three lenses, in order of weight:

**1. Fidelity to the post's argument.** The post wants the reader to *read* the reward — it is the unit of pedagogical content. A 6-line JSON spec sitting next to a rollout video is the right artifact. Option 2 puts a 30-line Python function in front of the reader for every example, half of which is plumbing (unpacking obs, handling info dict, numpy boilerplate); the signal-to-noise ratio for "what is this reward actually rewarding" is much worse. Option 1 makes the prompt, the spec, and the resulting behavior fit on one slide.

**2. Failure-mode quality.** The post's most interesting failures are *reward hacking* — the LLM specifies a reward that trains successfully but produces unintended behavior. The DSL produces those failures cleanly: wrong feature picked, wrong sign, wrong relative weight, missing penalty. Free-form Python adds a layer of *engineering* failures — NameError, dtype bugs, accidental in-place mutation, sandbox escape attempts — which are uneducational and burn reader attention. We want the failure spectrum centered on specification mistakes, not implementation mistakes.

**3. Implementation cost and surface area.** Option 1 needs a Pydantic schema, a feature registry per env, and a thin reward-builder function. Option 2 needs all of the above for at least one example (we still want hand-written rewards for the smoke tests) *plus* a sandbox. RestrictedPython, subprocess isolation, and resource limits are real engineering and not the project's purpose. Option 3 is option 1 plus option 2's costs, paid up front for marginal expressiveness gain on a 4–5-prompt v0.

## Consequences

**What this enables.**
- Validation is a Pydantic schema check + a single random-rollout smoke evaluation for NaN/inf/exception. No sandbox.
- The LLM prompt is short: it describes the env, lists the features by name with one-line semantics, and asks for JSON. This is a regime LLMs are very strong at.
- Caching by prompt hash is trivial — the spec is data.
- Visualizing per-component reward contributions during training is mechanical: each component is named and additive.

**What this gives up.**
- Rewards that genuinely need temporal logic (e.g. "do a flip" requires detecting a full rotation, not a per-step scalar) cannot be expressed without adding a feature for that behavior in the env. This is a feature, not a bug, for v0: it forces us to enumerate which prompts the demo can support and which need post-2's escape hatch. We will *not* attempt "flip" in v0 unless a clean feature definition emerges.
- The set of expressible rewards is bounded by the feature registry. We accept this and treat the registry as part of each env's public API.

**Architectural commitment so option 3 stays cheap.**
- Reward construction returns a `RewardFn`. Nothing downstream knows whether it came from a DSL spec or a Python function.
- The LLM prompt template lives in a single file per env and is parameterized by the feature list; swapping in a Python-emitting prompt does not touch the training harness.
- The validator is a layered pipeline: schema check → semantic check (no unknown features, finite weights) → smoke evaluation. A future Python-spec validator slots in as another schema variant.

## DSL sketch (informative, not normative; final shape lives in code)

```json
{
  "components": [
    {"feature": "forward_velocity", "weight": 1.0},
    {"feature": "control_cost",     "weight": -0.05},
    {"feature": "alive_bonus",      "weight": 1.0}
  ],
  "bias": 0.0
}
```

Features are pure callables `(obs, action, next_obs, info) -> float` declared per env. If a prompt seems to need a transformed feature (e.g. "stay near height 1.0"), we add it to the registry as a named feature (`height_proximity_1m`) rather than expanding the DSL. This keeps the DSL trivially analyzable and pushes complexity into the env, where it is reviewable and testable.

## Open questions deferred to implementation

- Exact feature list for HalfCheetah — defined in `src/prompt_to_policy/envs/halfcheetah.py` with unit tests.
- Whether to expose a `clip` or `scale` operator at the DSL level. Default: no. Add only if a real prompt forces it.
- Whether the LLM should be allowed to define new features. Default: no. The feature set is the contract between the env author and the LLM; broadening it weakens the pedagogical frame.

## Revisit triggers

Reopen this decision if any of the following hold:
- More than one of the v0 target prompts cannot be expressed in the DSL even after a reasonable feature is added to the env.
- The few-shot prompt for the LLM grows past ~40 lines to compensate for DSL rigidity.
- A blog-post draft reveals that readers want to see the reward as code rather than as data.
