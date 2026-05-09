"""Model loading and generation primitives shared by eval_harness.py and
generate.py. Keeps a single canonical code path for "load Gemma + LoRA,
apply the SFT chat template, generate, and clean the output."
"""
from __future__ import annotations

from typing import Callable, Optional

from prompts import SFT_SYSTEM_PROMPT


# Chat-template stop markers that mlx-lm doesn't always honor on its own.
# Without trimming, Gemma in particular produces hundreds of trailing
# `<end_of_turn>` repeats and post-EOS garbage tokens until max_tokens.
_STOP_MARKERS = ("<end_of_turn>", "<eos>", "<|im_end|>", "<|endoftext|>")


def clean_generation(text: str) -> str:
    """Strip everything from the first stop marker onward, then trim space."""
    earliest = len(text)
    for marker in _STOP_MARKERS:
        i = text.find(marker)
        if i != -1 and i < earliest:
            earliest = i
    return text[:earliest].strip()


def build_prompt(complex_text: str, tokenizer) -> str:
    """Apply the model's chat template with the SFT system prompt.

    `add_generation_prompt=True` appends the assistant-turn prefix so the
    model knows to start generating its reply.
    """
    messages = [
        {"role": "system", "content": SFT_SYSTEM_PROMPT},
        {"role": "user", "content": complex_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def load_model_with_adapter(model_id: str, adapter_path: Optional[str]):
    """mlx-lm load wrapper. `adapter_path=None` loads the base model."""
    from mlx_lm import load

    print(
        f"[inference] loading {model_id}"
        + (f" + adapter {adapter_path}" if adapter_path else " (base, no adapter)"),
        flush=True,
    )
    if adapter_path:
        return load(model_id, adapter_path=adapter_path)
    return load(model_id)


def make_generate_fn(model, tokenizer, max_tokens: int = 512) -> Callable[[str], str]:
    """Closure that takes one complex paragraph and returns the cleaned
    simplification using the SFT chat template."""
    from mlx_lm import generate

    def gen(complex_text: str) -> str:
        prompt = build_prompt(complex_text, tokenizer)
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        return clean_generation(raw)

    return gen
