"""Tests for the HalfCheetah prompt template builder."""

from __future__ import annotations

from prompt_to_policy.envs import halfcheetah as hc
from prompt_to_policy.llm import PROMPT_VERSION, build_system_prompt


def test_template_includes_every_feature_doc():
    prompt = build_system_prompt(hc.FEATURE_DOCS)
    for name, doc in hc.FEATURE_DOCS.items():
        assert name in prompt, f"feature name {name!r} missing from prompt"
        assert doc in prompt, f"feature doc for {name!r} missing from prompt"


def test_template_substitutes_placeholder():
    prompt = build_system_prompt({"foo": "bar baz"})
    assert "{FEATURES}" not in prompt
    assert "foo: bar baz" in prompt


def test_template_describes_output_schema():
    prompt = build_system_prompt(hc.FEATURE_DOCS)
    # Schema-level keywords the model should see
    for keyword in ("components", "feature", "weight", "bias", "JSON"):
        assert keyword in prompt


def test_prompt_version_is_set():
    assert PROMPT_VERSION
