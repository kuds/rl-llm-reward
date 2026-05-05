"""Tests for ``GeminiRewardClient`` with a stubbed Gemini SDK.

No live API calls and no ``google-genai`` import: the stub mimics just
enough of the SDK surface that the client uses (``models.generate_content``
returning an object with ``.text`` and ``.usage_metadata``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import ValidationError

from prompt_to_policy.envs import halfcheetah as hc
from prompt_to_policy.llm import GeminiRewardClient, GeneratedReward

GOOD_RESPONSE = json.dumps(
    {
        "components": [
            {"feature": "forward_velocity", "weight": 1.0},
            {"feature": "control_cost", "weight": -0.05},
        ]
    }
)


@dataclass
class _StubUsageMetadata:
    prompt_token_count: int = 234
    candidates_token_count: int = 89


@dataclass
class _StubResponse:
    text: str
    usage_metadata: _StubUsageMetadata = field(default_factory=_StubUsageMetadata)


@dataclass
class _StubModels:
    response_text: str
    calls: list[dict] = field(default_factory=list)
    usage: _StubUsageMetadata = field(default_factory=_StubUsageMetadata)

    def generate_content(self, **kwargs: Any) -> _StubResponse:
        self.calls.append(kwargs)
        return _StubResponse(text=self.response_text, usage_metadata=self.usage)


class _StubGeminiClient:
    def __init__(self, response_text: str, usage: _StubUsageMetadata | None = None):
        self.models = _StubModels(response_text=response_text, usage=usage or _StubUsageMetadata())


def test_generate_calls_api_with_expected_arguments(tmp_path):
    stub = _StubGeminiClient(
        GOOD_RESPONSE,
        usage=_StubUsageMetadata(prompt_token_count=234, candidates_token_count=89),
    )
    client = GeminiRewardClient(
        feature_docs=hc.FEATURE_DOCS,
        model="gemini-2.5-pro",
        cache_dir=tmp_path,
        gemini_client=stub,
    )
    result = client.generate("make the cheetah run forward")

    assert isinstance(result, GeneratedReward)
    assert not result.cached
    assert result.spec.components[0].feature == "forward_velocity"
    assert result.usage == {"input_tokens": 234, "output_tokens": 89}
    assert result.estimated_cost_usd > 0

    assert len(stub.models.calls) == 1
    call = stub.models.calls[0]
    assert call["model"] == "gemini-2.5-pro"
    assert call["contents"] == "make the cheetah run forward"
    # System prompt is passed via config.system_instruction.
    assert "forward_velocity" in call["config"]["system_instruction"]


def test_generate_writes_cache_file(tmp_path):
    stub = _StubGeminiClient(GOOD_RESPONSE)
    client = GeminiRewardClient(
        hc.FEATURE_DOCS, cache_dir=tmp_path, gemini_client=stub, env_name="halfcheetah"
    )

    result = client.generate("run forward")

    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    payload = json.loads(cache_files[0].read_text())
    assert payload["user_prompt"] == "run forward"
    assert payload["model"] == client.model
    assert payload["env"] == "halfcheetah"
    assert payload["spec"]["components"][0]["feature"] == "forward_velocity"
    assert payload["cache_key"] == result.cache_key


def test_generate_serves_cache_on_repeat(tmp_path):
    stub = _StubGeminiClient(GOOD_RESPONSE)
    client = GeminiRewardClient(hc.FEATURE_DOCS, cache_dir=tmp_path, gemini_client=stub)

    first = client.generate("run forward")
    second = client.generate("run forward")

    assert len(stub.models.calls) == 1
    assert second.cached is True
    assert first.cache_key == second.cache_key


def test_force_refresh_bypasses_cache(tmp_path):
    stub = _StubGeminiClient(GOOD_RESPONSE)
    client = GeminiRewardClient(hc.FEATURE_DOCS, cache_dir=tmp_path, gemini_client=stub)
    client.generate("run forward")
    client.generate("run forward", force_refresh=True)
    assert len(stub.models.calls) == 2


def test_cache_key_changes_with_model(tmp_path):
    stub = _StubGeminiClient(GOOD_RESPONSE)
    a = GeminiRewardClient(
        hc.FEATURE_DOCS, model="gemini-2.5-pro", cache_dir=tmp_path, gemini_client=stub
    )
    b = GeminiRewardClient(
        hc.FEATURE_DOCS, model="gemini-2.5-flash", cache_dir=tmp_path, gemini_client=stub
    )
    assert a.cache_key("run forward") != b.cache_key("run forward")


def test_invalid_response_surfaces_clearly(tmp_path):
    stub = _StubGeminiClient("this is not json")
    client = GeminiRewardClient(hc.FEATURE_DOCS, cache_dir=tmp_path, gemini_client=stub)

    with pytest.raises(ValidationError):
        client.generate("run forward")

    assert list(tmp_path.glob("*.json")) == []


def test_provider_id_distinguishes_anthropic_and_gemini_caches(tmp_path):
    """A Gemini cache file shouldn't satisfy a Claude lookup, even at matching prompts."""
    from prompt_to_policy.llm import LLMRewardClient

    gem = GeminiRewardClient(
        hc.FEATURE_DOCS,
        model="gemini-2.5-pro",
        cache_dir=tmp_path,
        gemini_client=_StubGeminiClient(GOOD_RESPONSE),
    )

    class _StubAnthropic:
        def __init__(self):
            class _M:
                def create(self, **kwargs):
                    raise AssertionError("Anthropic stub should not be called")

            self.messages = _M()

    claude = LLMRewardClient(
        hc.FEATURE_DOCS,
        model="claude-opus-4-7",
        cache_dir=tmp_path,
        anthropic_client=_StubAnthropic(),
    )

    assert gem.cache_key("run forward") != claude.cache_key("run forward")


def test_works_with_hopper_env_prompt(tmp_path):
    """Plumbing: a Hopper env spec produces a Hopper-flavored system prompt."""
    from prompt_to_policy import envs

    hopper_spec = envs.get("hopper")
    stub = _StubGeminiClient(GOOD_RESPONSE)
    client = GeminiRewardClient(
        feature_docs=hopper_spec.feature_docs,
        cache_dir=tmp_path,
        gemini_client=stub,
        build_prompt=hopper_spec.build_system_prompt,
        env_name="hopper",
    )
    client.generate("hop forward")
    call = stub.models.calls[0]
    assert "Hopper-v5" in call["config"]["system_instruction"]
