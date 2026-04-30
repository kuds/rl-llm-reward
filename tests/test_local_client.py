"""Tests for ``LocalLLMRewardClient`` with stubbed model + tokenizer.

No real model is downloaded. The stubs implement just enough of the
HuggingFace interface that the client uses: chat template rendering,
tensor batch encoding, ``generate()``, and ``decode()``. Real ``torch``
tensors are used because ``torch`` is already a hard dep via SB3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from prompt_to_policy.envs import halfcheetah as hc
from prompt_to_policy.llm import GeneratedReward, LocalLLMRewardClient

GOOD_RESPONSE = json.dumps(
    {
        "components": [
            {"feature": "forward_velocity", "weight": 1.0},
            {"feature": "control_cost", "weight": -0.05},
        ]
    }
)


class _StubBatch(dict):
    """Mimics HF BatchEncoding: dict-like + a .to(device) method."""

    def to(self, _device: Any) -> _StubBatch:
        return self


@dataclass
class _StubTokenizer:
    response_text: str
    eos_token_id: int | None = 0
    apply_calls: list[list[dict]] = field(default_factory=list)
    decode_calls: list[Any] = field(default_factory=list)

    # Constants used to fabricate a deterministic "prompt" tensor.
    _prompt_token_count: int = 11

    def apply_chat_template(
        self, messages: list[dict], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        # Record + return a rendered prompt; assert on it in tests.
        assert tokenize is False
        assert add_generation_prompt is True
        self.apply_calls.append(messages)
        return f"<|prompt|>{messages[-1]['content']}<|/prompt|>"

    def __call__(self, prompt_text: str, *, return_tensors: str) -> _StubBatch:
        assert return_tensors == "pt"
        # Shape (1, _prompt_token_count) — values don't matter; only shape does.
        ids = torch.zeros((1, self._prompt_token_count), dtype=torch.long)
        return _StubBatch(input_ids=ids)

    def decode(self, ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.decode_calls.append(ids)
        return self.response_text


@dataclass
class _StubModel:
    response_text: str
    tokenizer: _StubTokenizer
    generate_calls: list[dict] = field(default_factory=list)
    eval_called: bool = False

    @property
    def device(self) -> str:
        return "cpu"

    def eval(self) -> _StubModel:
        self.eval_called = True
        return self

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generate_calls.append(kwargs)
        # Echo back the prompt followed by some "generated" tokens. The
        # client slices off the prompt portion before decoding; the
        # tokenizer stub then ignores the actual tensor and returns
        # ``response_text``.
        prompt_ids = kwargs["input_ids"]
        # Pretend the model emitted 7 new tokens (so output_tokens == 7).
        new_ids = torch.zeros((1, 7), dtype=torch.long)
        return torch.cat([prompt_ids, new_ids], dim=-1)


def _make_client(tmp_path, response_text: str = GOOD_RESPONSE) -> LocalLLMRewardClient:
    tokenizer = _StubTokenizer(response_text=response_text)
    model = _StubModel(response_text=response_text, tokenizer=tokenizer)
    return LocalLLMRewardClient(
        feature_docs=hc.FEATURE_DOCS,
        model_id="stub/qwen-tiny",
        cache_dir=tmp_path,
        quantization="4bit",
        hf_model=model,
        hf_tokenizer=tokenizer,
    )


# --- Generation path --------------------------------------------------------


def test_generate_calls_tokenizer_and_model(tmp_path):
    client = _make_client(tmp_path)
    result = client.generate("make the cheetah run forward")

    assert isinstance(result, GeneratedReward)
    assert not result.cached
    assert result.spec.components[0].feature == "forward_velocity"

    tok: _StubTokenizer = client._tokenizer  # type: ignore[assignment]
    mdl: _StubModel = client._model  # type: ignore[assignment]

    # Chat template received system + user roles in that order.
    assert len(tok.apply_calls) == 1
    messages = tok.apply_calls[0]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "make the cheetah run forward"
    # System prompt is the rendered feature-doc template, so feature names appear.
    assert "forward_velocity" in messages[0]["content"]

    assert len(mdl.generate_calls) == 1


def test_generate_reports_token_usage(tmp_path):
    client = _make_client(tmp_path)
    result = client.generate("run forward")
    # _StubTokenizer fakes a prompt of 11 tokens; _StubModel emits 7 new tokens.
    assert result.usage == {"input_tokens": 11, "output_tokens": 7}


def test_estimated_cost_is_zero(tmp_path):
    """Local inference is not billed, regardless of token counts."""
    client = _make_client(tmp_path)
    result = client.generate("run forward")
    assert result.estimated_cost_usd == 0.0


def test_generate_uses_greedy_decoding_by_default(tmp_path):
    """temperature=0 should turn off sampling; high temperature should turn it on."""
    client = _make_client(tmp_path)
    client.generate("run forward")
    call = client._model.generate_calls[0]  # type: ignore[union-attr]
    assert call["do_sample"] is False
    assert "temperature" not in call


def test_generate_passes_temperature_when_sampling(tmp_path):
    tokenizer = _StubTokenizer(response_text=GOOD_RESPONSE)
    model = _StubModel(response_text=GOOD_RESPONSE, tokenizer=tokenizer)
    client = LocalLLMRewardClient(
        feature_docs=hc.FEATURE_DOCS,
        model_id="stub/qwen-tiny",
        cache_dir=tmp_path,
        temperature=0.7,
        hf_model=model,
        hf_tokenizer=tokenizer,
    )
    client.generate("run forward")
    call = model.generate_calls[0]
    assert call["do_sample"] is True
    assert call["temperature"] == 0.7


# --- Caching ----------------------------------------------------------------


def test_generate_writes_cache_file(tmp_path):
    client = _make_client(tmp_path)
    result = client.generate("run forward")

    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    payload = json.loads(cache_files[0].read_text())
    assert payload["user_prompt"] == "run forward"
    # Cache "model" string includes quantization mode for cache invalidation.
    assert payload["model"] == "stub/qwen-tiny@4bit"
    assert payload["spec"]["components"][0]["feature"] == "forward_velocity"
    assert payload["estimated_cost_usd"] == 0.0
    assert payload["cache_key"] == result.cache_key


def test_generate_serves_cache_on_repeat(tmp_path):
    client = _make_client(tmp_path)

    first = client.generate("run forward")
    second = client.generate("run forward")

    assert len(client._model.generate_calls) == 1, "second call should hit the cache"  # type: ignore[union-attr]
    assert second.cached is True
    assert first.cache_key == second.cache_key
    assert second.spec.components[0].feature == "forward_velocity"


def test_force_refresh_bypasses_cache(tmp_path):
    client = _make_client(tmp_path)
    client.generate("run forward")
    client.generate("run forward", force_refresh=True)
    assert len(client._model.generate_calls) == 2  # type: ignore[union-attr]


def test_quantization_mode_affects_cache_key(tmp_path):
    """Switching quantization should miss the cache: different modes can produce different specs."""
    tok4 = _StubTokenizer(response_text=GOOD_RESPONSE)
    mdl4 = _StubModel(response_text=GOOD_RESPONSE, tokenizer=tok4)
    a = LocalLLMRewardClient(
        feature_docs=hc.FEATURE_DOCS,
        model_id="m",
        cache_dir=tmp_path,
        quantization="4bit",
        hf_model=mdl4,
        hf_tokenizer=tok4,
    )
    tok8 = _StubTokenizer(response_text=GOOD_RESPONSE)
    mdl8 = _StubModel(response_text=GOOD_RESPONSE, tokenizer=tok8)
    b = LocalLLMRewardClient(
        feature_docs=hc.FEATURE_DOCS,
        model_id="m",
        cache_dir=tmp_path,
        quantization="8bit",
        hf_model=mdl8,
        hf_tokenizer=tok8,
    )
    assert a.cache_key("run forward") != b.cache_key("run forward")


def test_cache_key_changes_with_user_prompt(tmp_path):
    client = _make_client(tmp_path)
    assert client.cache_key("run forward") != client.cache_key("stand still")


# --- Failure modes ----------------------------------------------------------


def test_invalid_response_raises_and_does_not_cache(tmp_path):
    client = _make_client(tmp_path, response_text="this is not json")
    with pytest.raises(ValidationError):
        client.generate("run forward")
    assert list(tmp_path.glob("*.json")) == []
