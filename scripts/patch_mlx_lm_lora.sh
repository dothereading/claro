#!/usr/bin/env bash
# Patch mlx-lm-lora's hardcoded GRPO end_answer_token from "</answer>" to
# "<end_of_turn>" so Gemma chat-template stopping works during rollouts.
#
# Why: mlx-lm-lora was built primarily for math chain-of-thought tasks where
# the answer is wrapped in <answer>...</answer>. The GRPO trainer adds that
# string as a custom EOS via tokenizer.add_eos_token(). For Gemma, "</answer>"
# is multi-token, so add_eos_token raises ValueError, use_eos_token=False, and
# generation runs to max_completion_length. Result: model emits one clean
# response, then "stacks" 4–5 more responses separated by <end_of_turn>
# markers (which mlx-lm doesn't recognize as EOS). Reward function sees the
# stacked text and zeroes the length factor on every rollout → no gradient.
#
# Patching the default to "<end_of_turn>" (a single Gemma token, id 106) makes
# mlx-lm-lora register it as the per-rollout EOS. Generation stops cleanly.
#
# Idempotent: safe to run repeatedly. Re-apply after `uv sync` reinstalls
# mlx-lm-lora. See PLAN.md "Phase 3: GRPO" for context.
set -euo pipefail

cd "$(dirname "$0")/.."

TARGET=".venv/lib/python3.12/site-packages/mlx_lm_lora/trainer/grpo_trainer.py"
if [[ ! -f "$TARGET" ]]; then
    echo "ERROR: $TARGET not found. Run 'uv sync' first." >&2
    exit 1
fi

if grep -q 'end_answer_token: str = "<end_of_turn>"' "$TARGET"; then
    echo "[mlx-lm-lora patch] already applied to $TARGET"
    exit 0
fi

if ! grep -q 'end_answer_token: str = "</answer>"' "$TARGET"; then
    echo "ERROR: expected default not found in $TARGET. The library may have changed." >&2
    echo "Inspect manually:" >&2
    grep -n 'end_answer_token' "$TARGET" >&2 || true
    exit 1
fi

sed -i.bak 's|end_answer_token: str = "</answer>"|end_answer_token: str = "<end_of_turn>"|g' "$TARGET"
echo "[mlx-lm-lora patch] applied to $TARGET"
grep -n 'end_answer_token: str' "$TARGET"
