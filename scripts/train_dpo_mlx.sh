#!/usr/bin/env bash
# DPO fine-tune via mlx-lm-lora using preference pairs in data/dpo_mlx/.
# `chosen`  = Opus simplification (from data/sft.jsonl)
# `rejected` = Gemma-3-4B simplification (from data/dpo.jsonl)
#
# Run from the repo root:
#   uv run python mlx_data.py dpo
#   bash scripts/train_dpo_mlx.sh
#
# Resumes from the SFT LoRA adapter at adapters/sft-a2/ and writes the new
# adapter to adapters/dpo-a2/.
set -euo pipefail

# Run from the repo root regardless of where the script was invoked from.
cd "$(dirname "$0")/.."

MODEL="${MODEL:-mlx-community/gemma-3-1b-it-bf16}"
DATA_DIR="${DATA_DIR:-data/dpo_mlx}"
ADAPTER_DIR="${ADAPTER_DIR:-adapters/dpo-a2}"
RESUME_ADAPTER="${RESUME_ADAPTER:-adapters/sft-a2/adapters.safetensors}"
ITERS="${ITERS:-300}"
BATCH="${BATCH:-1}"
LR="${LR:-5e-6}"
LORA_LAYERS="${LORA_LAYERS:-16}"
BETA="${BETA:-0.1}"

mkdir -p "$ADAPTER_DIR"

RESUME_FLAG=()
if [[ -f "$RESUME_ADAPTER" ]]; then
    echo "[train] resuming from $RESUME_ADAPTER"
    RESUME_FLAG=(--resume-adapter-file "$RESUME_ADAPTER")
fi

echo "[train] model=$MODEL data=$DATA_DIR iters=$ITERS beta=$BETA"
uv run python -m mlx_lm_lora.train \
    --model "$MODEL" \
    --train \
    --train-mode dpo \
    --data "$DATA_DIR" \
    --adapter-path "$ADAPTER_DIR" \
    --batch-size "$BATCH" \
    --num-layers "$LORA_LAYERS" \
    --iters "$ITERS" \
    --learning-rate "$LR" \
    --beta "$BETA" \
    --dpo-cpo-loss-type sigmoid \
    --val-batches 5 \
    --steps-per-eval 50 \
    --steps-per-report 10 \
    --grad-checkpoint \
    "${RESUME_FLAG[@]}"
