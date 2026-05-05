"""Reward generators with on-disk prompt-hash caching.

Two implementations live here:

- ``LLMRewardClient`` calls the Anthropic API.
- ``LocalLLMRewardClient`` (in ``local_client.py``) loads a HuggingFace
  causal-LM and runs it locally — the typical use case is Colab with a
  GPU runtime.

Both inherit from ``BaseRewardClient``, which owns the shared flow:

    cache_key = sha256(provider_id || system_prompt || user_prompt)[:16]
    cache_dir/<key>.json  on hit  -> return parsed spec, no model call
                          on miss -> call the model, parse, validate, write file

Validation flow (on cache miss):
    1. Strip markdown fences from the response
    2. Parse as JSON and validate against RewardSpec (raises ValidationError on bad shape)

Smoke-testing the resulting RewardFn against a real env is the caller's
job (see ``reward.smoke_test_reward_fn``). It is not done here because
the client doesn't know which env the spec is bound to.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prompt_to_policy.reward import RewardSpec

from .pricing import estimate_cost_usd
from .templates import PROMPT_VERSION
from .templates import build_halfcheetah_system_prompt as _default_build_prompt

DEFAULT_MODEL = "claude-opus-4-7"

PromptBuilder = Callable[[dict[str, str]], str]


@dataclass
class GeneratedReward:
    spec: RewardSpec
    raw_response: str
    user_prompt: str
    model: str
    cached: bool
    usage: dict[str, int]
    estimated_cost_usd: float
    cache_key: str


class BaseRewardClient(ABC):
    """Shared cache-and-parse flow. Subclasses implement ``_call_model``.

    The cache key mixes the provider's identifying string (``model_id``)
    with the system prompt and the user prompt, so changing any of them
    (or switching envs, since each env produces a different system
    prompt) invalidates affected cache entries automatically.
    """

    def __init__(
        self,
        feature_docs: dict[str, str],
        model_id: str,
        cache_dir: Path | str | None,
        *,
        build_prompt: PromptBuilder | None = None,
        env_name: str = "halfcheetah",
    ) -> None:
        self.feature_docs = feature_docs
        self.model_id = model_id
        self.env_name = env_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        builder = build_prompt or _default_build_prompt
        self.system_prompt = builder(feature_docs)

    # The "model" recorded on GeneratedReward / cache files.
    @property
    def model(self) -> str:
        return self.model_id

    @abstractmethod
    def _call_model(self, user_prompt: str) -> tuple[str, dict[str, int]]:
        """Run the underlying model and return (raw_text, usage)."""

    def _estimated_cost_usd(self, usage: dict[str, int]) -> float:
        """Override in subclasses that aren't billed as Anthropic."""
        return estimate_cost_usd(self.model_id, usage)

    def cache_key(self, user_prompt: str) -> str:
        h = hashlib.sha256()
        h.update(self.model_id.encode("utf-8"))
        h.update(b"\0")
        h.update(self.system_prompt.encode("utf-8"))
        h.update(b"\0")
        h.update(user_prompt.encode("utf-8"))
        return h.hexdigest()[:16]

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str, user_prompt: str) -> GeneratedReward | None:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        data = json.loads(path.read_text())
        return GeneratedReward(
            spec=RewardSpec.model_validate(data["spec"]),
            raw_response=data.get("raw_response", ""),
            user_prompt=data.get("user_prompt", user_prompt),
            model=data.get("model", self.model_id),
            cached=True,
            usage=data.get("usage", {}),
            estimated_cost_usd=float(data.get("estimated_cost_usd", 0.0)),
            cache_key=key,
        )

    def _save_cache(self, result: GeneratedReward) -> None:
        path = self._cache_path(result.cache_key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_key": result.cache_key,
            "model": result.model,
            "env": self.env_name,
            "prompt_version": PROMPT_VERSION,
            "user_prompt": result.user_prompt,
            "raw_response": result.raw_response,
            "spec": result.spec.model_dump(),
            "usage": result.usage,
            "estimated_cost_usd": result.estimated_cost_usd,
            "created_at": datetime.now(UTC).isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2))

    def generate(self, user_prompt: str, *, force_refresh: bool = False) -> GeneratedReward:
        key = self.cache_key(user_prompt)
        if not force_refresh:
            cached = self._load_cache(key, user_prompt)
            if cached is not None:
                return cached

        text, usage = self._call_model(user_prompt)
        spec = parse_reward_spec(text)
        result = GeneratedReward(
            spec=spec,
            raw_response=text,
            user_prompt=user_prompt,
            model=self.model_id,
            cached=False,
            usage=usage,
            estimated_cost_usd=self._estimated_cost_usd(usage),
            cache_key=key,
        )
        self._save_cache(result)
        return result


class LLMRewardClient(BaseRewardClient):
    """Generate a ``RewardSpec`` from a natural-language prompt via Anthropic."""

    def __init__(
        self,
        feature_docs: dict[str, str],
        model: str = DEFAULT_MODEL,
        cache_dir: Path | str | None = None,
        anthropic_client: Any | None = None,
        max_tokens: int = 1024,
        *,
        build_prompt: PromptBuilder | None = None,
        env_name: str = "halfcheetah",
    ) -> None:
        super().__init__(
            feature_docs=feature_docs,
            model_id=model,
            cache_dir=cache_dir,
            build_prompt=build_prompt,
            env_name=env_name,
        )
        self._anthropic = anthropic_client
        self.max_tokens = max_tokens

    @property
    def anthropic(self) -> Any:
        if self._anthropic is None:
            import anthropic  # local import: anthropic SDK is heavy

            self._anthropic = anthropic.Anthropic()
        return self._anthropic

    def _call_model(self, user_prompt: str) -> tuple[str, dict[str, int]]:
        response = self.anthropic.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = _join_text_blocks(response.content)
        usage = {
            "input_tokens": int(getattr(response.usage, "input_tokens", 0)),
            "output_tokens": int(getattr(response.usage, "output_tokens", 0)),
        }
        return text, usage


def _join_text_blocks(content: Any) -> str:
    """Concatenate the text from an Anthropic messages.create() content list."""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def parse_reward_spec(text: str) -> RewardSpec:
    """Strip markdown fences and parse the LLM output as a ``RewardSpec``.

    Tolerates ```` ``` ```` and ```` ```json ```` fences. Anything else
    (extra prose, multiple JSON objects) raises ``ValidationError`` at
    parse time, which is the desired behavior — the LLM was told to
    output a single JSON object.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()
    return RewardSpec.model_validate_json(cleaned)
