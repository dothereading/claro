#!/usr/bin/env bash
# Patches to mlx-lm-lora's grpo_trainer.py. Idempotent; safe to run
# repeatedly. Re-apply after `uv sync` reinstalls mlx-lm-lora.
#
# Patch 1: end_answer_token "</answer>" → "<end_of_turn>"
#   mlx-lm-lora was built primarily for math chain-of-thought tasks where the
#   answer is wrapped in <answer>...</answer>. The GRPO trainer adds that
#   string as a custom EOS via tokenizer.add_eos_token(). For Gemma,
#   "</answer>" is multi-token, so add_eos_token raises ValueError,
#   use_eos_token=False, and generation runs to max_completion_length.
#   Result: model emits one clean response, then "stacks" 4–5 more responses
#   separated by <end_of_turn> markers (which mlx-lm doesn't recognize as
#   EOS). Reward function sees the stacked text and zeroes the length factor
#   on every rollout → no gradient. Patching the default to "<end_of_turn>"
#   (a single Gemma token, id 106) makes mlx-lm-lora register it as the
#   per-rollout EOS. Generation stops cleanly.
#
# Patch 2: logits FP16 → FP32 in get_per_token_logps
#   mlx-lm-lora casts model output logits to mx.float16 (max ~65504).
#   Gemma's vocab is 256K; with BF16 weights and any training that pushes a
#   logit above 65504, the FP16 cast produces inf. Then nn.log_softmax(inf)
#   yields NaN, which propagates into log_ratio → loss → gradient. Result:
#   v7 GRPO at G=8 NaN'd by iter ~15 regardless of lr or importance-sampling
#   level. Casting to FP32 keeps the full exponent range and prevents the
#   saturation. See LESSONS.md for the diagnostic trail.
set -euo pipefail

cd "$(dirname "$0")/.."

TARGET=".venv/lib/python3.12/site-packages/mlx_lm_lora/trainer/grpo_trainer.py"
if [[ ! -f "$TARGET" ]]; then
    echo "ERROR: $TARGET not found. Run 'uv sync' first." >&2
    exit 1
fi

# ---- Patch 1: end_answer_token ----
if grep -q 'end_answer_token: str = "<end_of_turn>"' "$TARGET"; then
    echo "[mlx-lm-lora patch] end_answer_token already applied"
elif grep -q 'end_answer_token: str = "</answer>"' "$TARGET"; then
    sed -i.bak 's|end_answer_token: str = "</answer>"|end_answer_token: str = "<end_of_turn>"|g' "$TARGET"
    echo "[mlx-lm-lora patch] end_answer_token applied"
    grep -n 'end_answer_token: str' "$TARGET"
else
    echo "ERROR: expected end_answer_token default not found. Library may have changed." >&2
    grep -n 'end_answer_token' "$TARGET" >&2 || true
    exit 1
fi

# ---- Patch 2: logits FP16 → FP32 ----
if grep -q 'logits = model(inputs).astype(mx.float32)' "$TARGET"; then
    echo "[mlx-lm-lora patch] logits-cast already applied"
elif grep -q 'logits = model(inputs).astype(mx.float16)' "$TARGET"; then
    sed -i.bak 's|logits = model(inputs).astype(mx.float16)|logits = model(inputs).astype(mx.float32)|g' "$TARGET"
    echo "[mlx-lm-lora patch] logits-cast applied"
    grep -n 'logits = model(inputs).astype' "$TARGET"
else
    echo "WARNING: expected logits FP16 cast not found — skipping logits-cast patch" >&2
    grep -n 'logits = model(inputs).astype' "$TARGET" >&2 || true
fi
