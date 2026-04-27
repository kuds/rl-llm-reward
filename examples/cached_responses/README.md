# Cached LLM responses

This directory holds responses from the Anthropic API, keyed by a hash
of `(model, system_prompt, user_prompt)`. Hits here mean re-running an
example is free and deterministic.

A cache file looks like:

```json
{
  "cache_key": "abc123...",
  "model": "claude-opus-4-7",
  "prompt_version": "v1",
  "user_prompt": "make the cheetah run forward as fast as possible",
  "raw_response": "{...}",
  "spec": { "components": [...], "bias": 0.0 },
  "usage": { "input_tokens": 0, "output_tokens": 0 },
  "estimated_cost_usd": 0.0,
  "created_at": "2026-04-27T..."
}
```

## Populating the cache

```
export ANTHROPIC_API_KEY=...
python examples/smoke_llm_three_prompts.py
```

## Invalidating

The cache key incorporates the system-prompt content, so any edit to
`src/prompt_to_policy/llm/templates/halfcheetah.py` causes a miss for
affected prompts. To force a refresh for a specific prompt, pass
`force_refresh=True` to `LLMRewardClient.generate`, or just delete the
relevant `.json` file.
