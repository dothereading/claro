#!/usr/bin/env bash
# GSPO fine-tuning against the cardinal CEFR-A2 reward
# (level_band x vocab x fidelity x format_gates). Sequence-level importance
# sampling (GSPO); KL-to-reference uses mlx-lm-lora's default beta=0.1.
# Needs OPENROUTER_API_KEY (loaded from .env by the runner). Logs -> runs/gspo/.
#
# Defaults reproduce the shipped 4B recipe. Env-var overrides (any subset):
#   MODEL DATA_DIR RESUME_ADAPTER ADAPTER_DIR ITERS GROUP_SIZE SAVE_EVERY \
#   MAX_COMPLETION_LENGTH LR TEMPERATURE WANDB_PROJECT
#
#   OPENROUTER_API_KEY=... RESUME_ADAPTER=adapters/sft/adapters.safetensors \
#     bash scripts/train_gspo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ARGS=(grpo
    --model "${MODEL:-mlx-community/gemma-3-4b-it-bf16}"
    --data "${DATA_DIR:-data/grpo}"
    --resume-adapter "${RESUME_ADAPTER:-adapters/sft/adapters.safetensors}"
    --reward-functions "cefr_a2_reward"
    --reward-functions-file "claro/training/rewards.py"
    --reward-weights "[1.0]"
    --iters "${ITERS:-200}"
    --batch-size "${BATCH:-1}"
    --lr "${LR:-1e-6}"
    --lora-layers "${LORA_LAYERS:-16}"
    --group-size "${GROUP_SIZE:-8}"
    --temperature "${TEMPERATURE:-1.0}"
    --max-completion-length "${MAX_COMPLETION_LENGTH:-384}"
    --importance-sampling-level "${IMPORTANCE_SAMPLING_LEVEL:-sequence}"
    --save-every "${SAVE_EVERY:-50}"
    --adapter-path "${ADAPTER_DIR:-adapters/gspo}"
    --project "${WANDB_PROJECT:-lang-simp-gspo}")

echo "[gspo] reward=cefr_a2_reward -> ${ADAPTER_DIR:-adapters/gspo}, logs runs/gspo/"
uv run python -m claro.training.runner "${ARGS[@]}"
