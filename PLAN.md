# Project Plan: Language Simplification LLM (Gemma 4 E2B)

## Goal
Fine-tune **Gemma 4 E2B** to perform language simplification, transforming complex text into **CEFR A2 level (Elementary English)**. The model must optimize for brevity and simplicity while strictly preserving semantic meaning. The project will compare "Non-Thinking" vs "REASONING" model modes.

## Core Pipeline

### Phase 1: Data Preparation (Synthetic/Distillation)
*   [x] Wikipedia source wired up (`sources.py`); ArXiv still TODO.
*   [x] Teacher distillation through OpenRouter (`distill.py`, `Teacher` class). Opus produces `chosen`; weaker model (Gemma-3-4B) produces `rejected`.
*   [x] SFT dataset: 194 pairs in `data/sft.jsonl`.
*   [x] DPO preference dataset: 194 triples in `data/dpo.jsonl`.

### Phase 2: Supervised Fine-Tuning (SFT)
*   **Model:** Currently `mlx-community/gemma-3-1b-it-bf16` (Gemma 4 E2B is broken in mlx-lm 0.31.3 — k/v proj mismatch). Plan target was Gemma 4 E2B; revisit when fixed upstream.
*   **Engine:** mlx-lm LoRA (`scripts/train_mlx.sh`). Unsloth path was removed.
*   **Status:** trained 300 iters; train loss 0.2–0.5, val loss climbing 1.9 → 2.4 → likely overfitting on 90 train rows. More data needed.

### Phase 3: Reinforcement Learning (GRPO)
Using **Group Relative Policy Optimization** to optimize against three specific reward components via an **E2B-based Reward Verifier**:

1.  **Reward A: Length Constraint ($\text{R}_{\text{len}}$)**
    *   Penalty starts for sentences $> 10$ words.
    *   Penalty increases monotonically as sentence length grows.
2.  **Reward B: Vocabulary Simplicity ($\text{R}_{\text{vocab}}$)**
    *   Identify "uncommon" words using frequency/difficulty metrics.
    *   Constraint: Maximum of 1 or 2 uncommon words per sentence; otherwise, apply penalty.
3.  **Reward C: Semantic Preservation ($\text{R}_{\text{meaning}}$)**
    *   **Method:** Use an independent "Judge" (via E2B) to compare the source and simplified text.
    *   Check for information loss or hallucinated additions.
4.  **Reward D: Difficulty Ranking/Ordering ($\text{R}_{\text{difficulty}}$)**
    *   **Method:** An LLM judge ranks a set of rollouts (e.g., A1, A2, and B1 versions).
    *   **Objective:** Ensure the model's output correctly sits within the target A2 difficulty tier by rewarding correct ordinal ranking of complexity levels.

### Phase 4: Preference Alignment (DPO)
*   **Engine:** mlx-lm-lora (`scripts/train_dpo_mlx.sh`), resumes from the SFT adapter.
*   **Status:** trained 300 iters with β=0.1; train loss saturated at 0.000 / accuracy 1.0 very early; val rewards near zero. Suspicious — needs eval against held-out prompts before trusting.

### Phase 5: Comparative Analysis
*   [ ] Run evaluation on **Non-Thinking** Gemma 4 E2B.
*   [ ] Run evaluation on **Thinking** Gemma 4 E2B (using `<|thought|>` tokens).
*   [ ] Metric comparison: SARI, BLEU, and Semantic Similarity across both modes.

## Tech Stack
*   **Base Model:** Currently Gemma-3-1B-it (mlx); target Gemma 4 E2B once mlx-lm support lands.
*   **Training Engine:** mlx-lm + mlx-lm-lora (LoRA)
*   **RL Framework:** GRPO (not yet implemented)
*   **Preference Alignment:** DPO via mlx-lm-lora
*   **Reward Sandbox:** local LM Studio judge (CEFR difficulty ranking); other rewards TBD
*   **Dependency Management:** `uv`
*   **Tests:** `pytest` (mocked OpenAI client + mocked judge)
