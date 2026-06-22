#!/usr/bin/env bash
# v10 cardinal-reward GRPO run (§6). Two arms, identical except the reward:
#
#   ARM=nojudge  -> v10_nojudge_reward  (level_band x vocab x gates; free)
#   ARM=full     -> v10_full_reward     (x fidelity; needs OPENROUTER_API_KEY)
#
# Launch Arm B (free) first to confirm the pipeline, then Arm A:
#   ARM=nojudge bash scripts/train_grpo_v10.sh
#   ARM=full    OPENROUTER_API_KEY=... bash scripts/train_grpo_v10.sh
#
# Fixed hyperparameters match the spec; override any via env. Per-iteration
# metrics + per-rollout logs land in runs/<arm>/ (metrics.jsonl, rollouts.jsonl).
set -euo pipefail
cd "$(dirname "$0")/.."

ARM="${ARM:-nojudge}"
case "$ARM" in
  full)    REWARD_FN="v10_full_reward" ;;
  nojudge) REWARD_FN="v10_nojudge_reward" ;;
  *) echo "ARM must be 'full' or 'nojudge', got '$ARM'" >&2; exit 2 ;;
esac

ARGS=(grpo
    --model "${MODEL:-mlx-community/gemma-3-1b-it-bf16}"
    --data "${DATA_DIR:-data/grpo}"
    --resume-adapter "${RESUME_ADAPTER:-adapters/sft_n750/adapters.safetensors}"
    --reward-functions "$REWARD_FN"
    --reward-functions-file "experiments/rewards/rewards_v10.py"
    --reward-weights "[1.0]"
    --iters "${ITERS:-50}"
    --batch-size "${BATCH:-1}"
    --lr "${LR:-1e-6}"
    --lora-layers "${LORA_LAYERS:-16}"
    --group-size "${GROUP_SIZE:-8}"
    --temperature "${TEMPERATURE:-1.0}"
    --max-completion-length "${MAX_COMPLETION_LENGTH:-512}"
    --importance-sampling-level "${IMPORTANCE_SAMPLING_LEVEL:-sequence}"
    --adapter-path "${ADAPTER_DIR:-adapters/grpo_v10_$ARM}"
    --project "${WANDB_PROJECT:-lang-simp-grpo-v10}")

echo "[v10] arm=$ARM reward_fn=$REWARD_FN -> adapters/grpo_v10_$ARM, logs runs/$ARM/"
uv run python -m langsimp.training.runner "${ARGS[@]}"
