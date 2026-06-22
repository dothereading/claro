"""Reward-function registry shim.

mlx-lm-lora discovers reward functions through a decorator. We re-export it from
one place so the shipped reward and the archived experiments register the same
way, with a no-op fallback so these modules import without mlx-lm-lora present
(e.g. under pytest).
"""

from __future__ import annotations

try:
    from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function
except ImportError:  # tests / environments without the trainer
    def register_reward_function(name=None):
        def deco(fn):
            return fn

        return deco
