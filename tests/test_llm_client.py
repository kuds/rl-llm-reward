"""Tests for LLMRewardClient with a stub Anthropic client.

No live API calls. The stub records every messages.create() it sees so
we can assert on prompt construction, and returns a canned response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from prompt_to_policy.envs import halfcheetah as hc
from prompt_to_policy.llm import GeneratedReward, LLMRewardClient, parse_reward_spec

# --- A minimal stub of the Anthropic Python SDK -----------------------------


@dataclass
class _StubUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class _StubTextBlock:
    text: str
    type: str = "text"


@dataclass
class _StubResponse:
    content: list[_StubTextBlock]
    usage: _StubUsage = field(default_factory=_StubUsage)


@dataclass
class _StubMessages:
    response_text: str
    calls: list[dict] = field(default_factory=list)
    usage: _StubUsage = field(default_factory=_StubUsage)

    def create(self, **kwargs) -> _StubResponse:
        self.calls.append(kwargs)
        return _StubResponse(content=[_StubTextBlock(text=self.response_text)], usage=self.usage)


class _StubAnthropic:
    def __init__(self, response_text: str, usage: _StubUsage | None = None):
        self.messages = _StubMessages(response_text=response_text, usage=usage or _StubUsage())


GOOD_RESPONSE = json.dumps(
    {
        "components": [
            {"feature": "forward_velocity", "weight": 1.0},
            {"feature": "control_cost", "weight": -0.05},
        ]
    }
)


# --- parse_reward_spec ------------------------------------------------------


def test_parse_plain_json():
    spec = parse_reward_spec(GOOD_RESPONSE)
    assert spec.components[0].feature == "forward_velocity"


def test_parse_strips_json_fence():
    fenced = f"```json\n{GOOD_RESPONSE}\n```"
    spec = parse_reward_spec(fenced)
    assert spec.components[0].weight == 1.0


def test_parse_strips_unlabeled_fence():
    fenced = f"```\n{GOOD_RESPONSE}\n```"
    spec = parse_reward_spec(fenced)
    assert len(spec.components) == 2


def test_parse_strips_surrounding_whitespace():
    spec = parse_reward_spec(f"   \n\n{GOOD_RESPONSE}\n\n   ")
    assert spec.components[0].feature == "forward_velocity"


def test_parse_rejects_prose_around_json():
    with pytest.raises(ValidationError):
        parse_reward_spec(f"Here you go:\n\n{GOOD_RESPONSE}\n\nHope that helps!")


def test_parse_rejects_unknown_field():
    bad = json.dumps(
        {"components": [{"feature": "forward_velocity", "weight": 1.0, "transform": "abs"}]}
    )
    with pytest.raises(ValidationError):
        parse_reward_spec(bad)


# --- LLMRewardClient: API path ----------------------------------------------


def test_generate_calls_api_with_expected_arguments(tmp_path):
    stub = _StubAnthropic(GOOD_RESPONSE, usage=_StubUsage(input_tokens=234, output_tokens=89))
    client = LLMRewardClient(
        feature_docs=hc.FEATURE_DOCS,
        model="claude-opus-4-7",
        cache_dir=tmp_path,
        anthropic_client=stub,
    )
    result = client.generate("make the cheetah run forward")

    assert isinstance(result, GeneratedReward)
    assert not result.cached
    assert result.spec.components[0].feature == "forward_velocity"
    assert result.usage == {"input_tokens": 234, "output_tokens": 89}
    assert result.estimated_cost_usd > 0

    assert len(stub.messages.calls) == 1
    call = stub.messages.calls[0]
    assert call["model"] == "claude-opus-4-7"
    assert call["messages"] == [{"role": "user", "content": "make the cheetah run forward"}]
    # System prompt contains the feature names from the registry
    assert "forward_velocity" in call["system"]
    assert "torso_uprightness" in call["system"]


def test_generate_writes_cache_file(tmp_path):
    stub = _StubAnthropic(GOOD_RESPONSE)
    client = LLMRewardClient(hc.FEATURE_DOCS, cache_dir=tmp_path, anthropic_client=stub)

    result = client.generate("run forward")

    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    payload = json.loads(cache_files[0].read_text())
    assert payload["user_prompt"] == "run forward"
    assert payload["model"] == client.model
    assert payload["spec"]["components"][0]["feature"] == "forward_velocity"
    assert "created_at" in payload
    assert payload["prompt_version"]
    assert payload["cache_key"] == result.cache_key


def test_generate_serves_cache_on_repeat(tmp_path):
    stub = _StubAnthropic(GOOD_RESPONSE)
    client = LLMRewardClient(hc.FEATURE_DOCS, cache_dir=tmp_path, anthropic_client=stub)

    first = client.generate("run forward")
    second = client.generate("run forward")

    assert len(stub.messages.calls) == 1, "second call should hit the cache"
    assert second.cached is True
    assert first.cache_key == second.cache_key
    assert second.spec.components[0].feature == "forward_velocity"


def test_force_refresh_bypasses_cache(tmp_path):
    stub = _StubAnthropic(GOOD_RESPONSE)
    client = LLMRewardClient(hc.FEATURE_DOCS, cache_dir=tmp_path, anthropic_client=stub)

    client.generate("run forward")
    client.generate("run forward", force_refresh=True)

    assert len(stub.messages.calls) == 2


def test_no_cache_dir_means_no_caching(tmp_path):
    stub = _StubAnthropic(GOOD_RESPONSE)
    client = LLMRewardClient(hc.FEATURE_DOCS, cache_dir=None, anthropic_client=stub)

    client.generate("run forward")
    client.generate("run forward")

    assert len(stub.messages.calls) == 2


def test_cache_key_changes_with_user_prompt(tmp_path):
    stub = _StubAnthropic(GOOD_RESPONSE)
    client = LLMRewardClient(hc.FEATURE_DOCS, cache_dir=tmp_path, anthropic_client=stub)
    k1 = client.cache_key("run forward")
    k2 = client.cache_key("stand still")
    assert k1 != k2


def test_cache_key_changes_with_model(tmp_path):
    stub = _StubAnthropic(GOOD_RESPONSE)
    a = LLMRewardClient(
        hc.FEATURE_DOCS, model="claude-opus-4-7", cache_dir=tmp_path, anthropic_client=stub
    )
    b = LLMRewardClient(
        hc.FEATURE_DOCS, model="claude-sonnet-4-6", cache_dir=tmp_path, anthropic_client=stub
    )
    assert a.cache_key("run forward") != b.cache_key("run forward")


def test_cache_key_changes_with_feature_registry(tmp_path):
    """A new feature in the registry invalidates cached responses."""
    stub = _StubAnthropic(GOOD_RESPONSE)
    a = LLMRewardClient(hc.FEATURE_DOCS, cache_dir=tmp_path, anthropic_client=stub)
    b = LLMRewardClient(
        {**hc.FEATURE_DOCS, "new_feature": "a brand new feature"},
        cache_dir=tmp_path,
        anthropic_client=stub,
    )
    assert a.cache_key("run forward") != b.cache_key("run forward")


def test_invalid_response_surfaces_clearly(tmp_path):
    stub = _StubAnthropic("this is not json")
    client = LLMRewardClient(hc.FEATURE_DOCS, cache_dir=tmp_path, anthropic_client=stub)

    with pytest.raises(ValidationError):
        client.generate("run forward")

    # Failed call must NOT poison the cache.
    assert list(tmp_path.glob("*.json")) == []
