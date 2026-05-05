"""Reward generator backed by Google's Gemini API.

Mirrors ``LLMRewardClient`` (Anthropic) so the rest of the pipeline
treats the two interchangeably. Uses the ``google-genai`` SDK; the
import is lazy so that users without ``google-genai`` installed can
still use the Anthropic and local backends.

The system prompt and user prompt are sent as a single user message
(the SDK's ``system_instruction`` parameter would also work, but
sending both content blocks together keeps caching semantics identical
to ``LLMRewardClient``: the cache key hashes the exact text the model
sees).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .client import BaseRewardClient, PromptBuilder

DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"


class GeminiRewardClient(BaseRewardClient):
    """Generate a ``RewardSpec`` from a natural-language prompt via Google Gemini.

    Authentication: reads ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``)
    from the environment when no client is injected. Pass
    ``gemini_client`` to inject a pre-configured ``google.genai.Client``
    instance, primarily useful for tests.
    """

    def __init__(
        self,
        feature_docs: dict[str, str],
        model: str = DEFAULT_GEMINI_MODEL,
        cache_dir: Path | str | None = None,
        gemini_client: Any | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
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
        self._gemini = gemini_client
        self.max_tokens = max_tokens
        self.temperature = temperature

    @property
    def gemini(self) -> Any:
        if self._gemini is None:
            from google import genai  # local import: SDK is optional

            self._gemini = genai.Client()
        return self._gemini

    def _call_model(self, user_prompt: str) -> tuple[str, dict[str, int]]:
        # The google-genai SDK accepts ``config`` as either a
        # ``types.GenerateContentConfig`` or a plain dict. Using a dict
        # avoids importing ``google.genai.types`` here, so the import
        # cost (and the optional dependency) only applies when the
        # caller actually instantiates a real Gemini client.
        config = {
            "system_instruction": self.system_prompt,
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_mime_type": "application/json",
        }
        response = self.gemini.models.generate_content(
            model=self.model_id,
            contents=user_prompt,
            config=config,
        )
        text = _extract_text(response)
        usage = _extract_usage(response)
        return text, usage


def _extract_text(response: Any) -> str:
    """Pull the text out of a Gemini GenerateContentResponse."""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text
    # Fall back to walking candidates / parts if .text isn't populated.
    parts: list[str] = []
    for cand in getattr(response, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            t = getattr(part, "text", None)
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts)


def _extract_usage(response: Any) -> dict[str, int]:
    """Map Gemini's usage_metadata to our standard {input,output}_tokens dict."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(getattr(meta, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(meta, "candidates_token_count", 0) or 0),
    }
