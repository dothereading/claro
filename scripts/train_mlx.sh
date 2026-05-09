#!/usr/bin/env bash
# Fine-tune Gemma 4 E2B on the language-simplification SFT dataset using mlx-lm.
#
# Prereqs (one-time):
#   uv add mlx mlx-lm
#   huggingface-cli login   # if the model requires gated access
#
# Run from the repo root:
#   bash scripts/train_mlx.sh
#
# Outputs a LoRA adapter at adapters/sft-a2/.
set -euo pipefail

# Run from the repo root regardless of where the script was invoked from.
cd "$(dirname "$0")/.."

MODEL="${MODEL:-mlx-community/gemma-3-1b-it-bf16}"   # Gemma-4 E2B is broken in mlx-lm 0.31.3 (k/v proj mismatch); using Gemma-3 1B-it as a working stand-in
DATA_DIR="${DATA_DIR:-data/mlx}"
ADAPTER_DIR="${ADAPTER_DIR:-adapters/sft-a2}"
ITERS="${ITERS:-300}"          # ~3 epochs on 90 train rows w/ batch 1, grad-accum 1
BATCH="${BATCH:-1}"
LR="${LR:-1e-4}"
LORA_LAYERS="${LORA_LAYERS:-16}"

mkdir -p "$ADAPTER_DIR"

echo "[train] model=$MODEL data=$DATA_DIR iters=$ITERS"
uv run python -m mlx_lm lora \
    --model "$MODEL" \
    --train \
    --data "$DATA_DIR" \
    --adapter-path "$ADAPTER_DIR" \
    --batch-size "$BATCH" \
    --num-layers "$LORA_LAYERS" \
    --iters "$ITERS" \
    --learning-rate "$LR" \
    --val-batches 5 \
    --steps-per-eval 50 \
    --steps-per-report 10 \
    --grad-checkpoint
