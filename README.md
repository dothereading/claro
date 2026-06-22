![Claro](simple_lm_banner.jpg)

# Claro — Language Simplification Fine-Tuning

Fine-tune a small **Gemma 3** model to rewrite complex English at **CEFR A2**
(elementary) level while preserving the source's facts. The shipped model,
**Claro** (Gemma-3 4B), is on the Hub:
[`miguelconner4/gemma-3-4b-cefr-a2-en`](https://huggingface.co/miguelconner4/gemma-3-4b-cefr-a2-en).

Pipeline: **data → SFT → GSPO → eval.** Training runs on Apple Silicon via MLX
(LoRA). The interesting part is the reward — a decomposed, mostly-deterministic
signal for "is this simple *and* faithful?" — which lives in the `reward/`
package. See [`LESSONS.md`](LESSONS.md) for the story of how the verifier evolved,
and [`docs/`](docs/) for the detailed design notes / findings.

## Layout

```
langsimp/                 core package
  data/                   distill (teacher), mlx_format, sources, audit
  training/
    runner.py             sft / dpo / grpo CLI -> mlx-lm(-lora), W&B + versioning
    rewards.py            the shipped reward (cefr_a2_reward) — thin trainer adapter
    registry.py           register_reward_function shim
  inference/              engine, generate, eval_harness
  prompts.py, verifier.py
reward/                   THE reward: level_band x vocab x fidelity x format_gates
  compose, level_band, vocab, fidelity, gates, nlp
scripts/                  the entrypoints needed to reproduce Claro
  train_sft.sh            SFT
  train_gspo.sh           GSPO against cefr_a2_reward
  build_vocab_list.py     \  build the reward's config
  calibrate_band.py       /  (data/vocab_1500.txt, config/band.json)
  eval_difficulty_paired.py, eval_composite.py, select_checkpoint.py, rescore_judge_swap.py
  patch_mlx_lm_lora.sh    required mlx-lm-lora patch (run after every `uv sync`)
tests/                    pytest suite mirroring core (run: `uv run pytest`)
experiments/              archived: abandoned reward arms (v6–v10), DPO, one-off probes/audits
docs/                     design specs + findings (PLAN, V*_FINDINGS, ...)
config/  prompts/  samples.jsonl     reward config, judge prompts, CEFR anchors
```

Generated artifacts (`data/`, `adapters/`, `runs/`, `eval_results/`, `logs/`,
`wandb/`) are gitignored.

## Setup

```bash
uv sync
bash scripts/patch_mlx_lm_lora.sh                 # required; idempotent, re-run after uv sync
echo "OPENROUTER_API_KEY=sk-..." > .env           # teacher (distill) + fidelity judge
```

## Reproduce Claro

```bash
# 1. Data: distill (complex -> A2) pairs from a teacher over random Wikipedia,
#    carve a frozen held-out eval set, convert to mlx-lm format.
uv run python -m langsimp.data.distill sft --n 1500
uv run python -m langsimp.data.mlx_format carve-eval --n 30
uv run python -m langsimp.data.mlx_format sft
uv run python -m langsimp.data.mlx_format grpo

# 2. (one time) Build the reward's config from real A2 reference text.
uv run python scripts/build_vocab_list.py         # -> data/vocab_1500.txt
uv run python scripts/calibrate_band.py           # -> config/band.json

# 3. SFT (defaults to gemma-3-4b-it). WANDB_MODE=disabled to skip W&B.
MODEL=mlx-community/gemma-3-4b-it-bf16 bash scripts/train_sft.sh

# 4. GSPO from the SFT checkpoint, against the cardinal reward.
OPENROUTER_API_KEY=... \
  MODEL=mlx-community/gemma-3-4b-it-bf16 \
  RESUME_ADAPTER=adapters/sft/latest/adapters.safetensors \
  bash scripts/train_gspo.sh

# 5. Evaluate: difficulty (DeepSeek CEFR classifier, paired) + faithfulness.
uv run python scripts/eval_difficulty_paired.py \
    --adapter adapters/gspo --baseline eval_results/<sft_eval>.json --out eval_results/gspo_eval.json
uv run python scripts/eval_composite.py eval_results/gspo_eval.json
```

## The reward (`reward/`)

`reward = level_band × vocab × fidelity × format_gates` — multiplicative, each in
[0, 1], so any component can veto and none can rescue:

- **level_band** — deterministic A2 difficulty from readability (Flesch), mean
  sentence length, and passive / subordination density, with bands calibrated to
  the 10th–90th percentiles of real A2 reference texts.
- **vocab** — penalty for off-A2-list words, with gloss-aware exemption.
- **fidelity** — an LLM judge, decomposed into fact-level recall + hallucination
  counts (not a holistic score). Cached; the only paid component.
- **format_gates** — hard 0/1 mask for markdown / degenerate loops.

## Tests

```bash
uv run pytest        # core + archived experiments; no network (judge/client mocked)
```
