"""System prompts for the language-simplification pipeline.

Two prompts do most of the work:

* `DISTILL_SYSTEM_PROMPT` is the long, rule-heavy version sent to the Teacher
  (Opus / Gemma / etc.) when generating training data. It needs to be detailed
  because we are relying on the teacher's in-context obedience.

* `SFT_SYSTEM_PROMPT` is the short version baked into the chat template for SFT
  and DPO. The model learns the rules from gradient updates, so the inference-
  time prompt only needs to carry intent.

Plus three `REJECTED_*` prompts used to diversify the DPO rejected pool.

The prompts themselves live in `prompts.yaml` (easier to edit, no Python
escaping). This module is just the loader.
"""

from pathlib import Path

import yaml

_PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.yaml"
_data: dict[str, str] = yaml.safe_load(_PROMPTS_PATH.read_text())

DISTILL_SYSTEM_PROMPT: str = _data["distill_system"]
SFT_SYSTEM_PROMPT: str = _data["sft_system"]
REJECTED_SUMMARIZE_PROMPT: str = _data["rejected_summarize"]
REJECTED_ELI5_PROMPT: str = _data["rejected_eli5"]
REJECTED_CLARIFY_PROMPT: str = _data["rejected_clarify"]
