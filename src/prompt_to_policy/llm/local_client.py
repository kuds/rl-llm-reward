"""Local-model reward generator using HuggingFace ``transformers``.

The intended use is Colab: download a small instruction-tuned causal LM
once, run it on the same GPU that PPO will later use, and skip the
Anthropic API entirely. The same on-disk cache as ``LLMRewardClient``
applies — re-running the same prompt is free.

Quantization defaults to 4-bit (bitsandbytes), which lets a 7-8B model
fit on a free Colab T4 (16 GB) with room to spare for SB3+MuJoCo. On a
larger card (L4, A100) you can switch to ``"8bit"`` or ``"none"`` for
faster generation at the cost of VRAM.

The cache key includes the model id and the quantization mode, so
swapping either invalidates affected cache entries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .client import BaseRewardClient

DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_QUANTIZATION: Literal["4bit", "8bit", "none"] = "4bit"

QuantizationMode = Literal["4bit", "8bit", "none"]


class LocalLLMRewardClient(BaseRewardClient):
    """Generate a ``RewardSpec`` from a natural-language prompt via a local HF model.

    The model and tokenizer are loaded lazily on the first call to
    ``generate`` so that a fully cached run never touches ``transformers``
    or ``torch``. This matters in Colab where importing transformers is
    slow and the cache hit case should be ~instant.
    """

    def __init__(
        self,
        feature_docs: dict[str, str],
        model_id: str = DEFAULT_LOCAL_MODEL,
        cache_dir: Path | str | None = None,
        *,
        quantization: QuantizationMode = DEFAULT_QUANTIZATION,
        device_map: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        torch_dtype: str = "auto",
        hf_model: Any | None = None,
        hf_tokenizer: Any | None = None,
    ) -> None:
        # The cache key has to distinguish e.g. Qwen-4bit from Qwen-8bit, which
        # may emit subtly different specs. Bake the quantization mode into the
        # model_id used for cache hashing.
        cache_model_id = f"{model_id}@{quantization}"
        super().__init__(feature_docs=feature_docs, model_id=cache_model_id, cache_dir=cache_dir)
        self.hf_model_id = model_id
        self.quantization = quantization
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.torch_dtype = torch_dtype
        self._model = hf_model
        self._tokenizer = hf_tokenizer

    def _load(self) -> tuple[Any, Any]:
        """Load model + tokenizer if not already loaded. Lazy and idempotent."""
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        # Lazy imports so that import-time cost only hits users who actually
        # instantiate this client and miss the cache.
        import torch  # noqa: F401  (transformers needs torch)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs: dict[str, Any] = {"device_map": self.device_map}

        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "8bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == "none":
            load_kwargs["torch_dtype"] = self.torch_dtype
        else:  # pragma: no cover - guarded by Literal type
            raise ValueError(f"unknown quantization mode: {self.quantization}")

        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.hf_model_id, **load_kwargs)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        return model, tokenizer

    def _call_model(self, user_prompt: str) -> tuple[str, dict[str, int]]:
        import torch

        model, tokenizer = self._load()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_tokens = int(inputs["input_ids"].shape[-1])

        do_sample = self.temperature > 0.0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
        if tokenizer.eos_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)

        # Slice off the prompt tokens; only keep what the model generated.
        generated_ids = output[0, inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        output_tokens = int(generated_ids.shape[-1])

        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        return text, usage

    def _estimated_cost_usd(self, usage: dict[str, int]) -> float:
        # Local inference is "free" from an API-spend perspective.
        return 0.0
