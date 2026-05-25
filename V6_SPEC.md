# V6 reward spec for GRPO training

**Status:** design — not yet implemented. This document is the contract for
the implementing agent.

## Why v6

The v5 stack (see `langsimp.training.rewards.CombinedRewardV5`) addressed
v3's known biases but has two remaining problems:

1. **Single-axis fidelity** (`f`/`h` scalars from the judge) misses
   specific failure modes the model exhibits — duplicated entities,
   misattribution, off-by-one number errors, dropped key facts. Need
   structured fact-by-fact analysis.
2. **Pointwise scoring of per-rollout difficulty** creates the
   "loss=0 forever" pathology: at G=2, both rollouts often score the same
   CEFR bucket → advantage 0 → no gradient. Need a comparative signal.

v6 introduces **two complementary judge calls** per training step:

* **Per-rollout 8-axis fidelity** (adapted from Guidroz et al, 2025,
  Google) — atomic-claim decomposition with weighted error categories.
* **Per-group A2-quality ranking** (single call, one permutation of IDs) —
  comparative judgment with CEFR anchors.

These push the policy in different directions (faithfulness vs A2-style
register). Both forces are intentional. The combined reward balances
them.

## Architecture overview

```
┌──────────────────────────────────────────────────────────┐
│  v6_combined_reward(prompts, completions, answer)        │
│                                                          │
│  1. Group rollouts by prompt (mlx_lm_lora calls this fn  │
│     with batch_size × G rollouts; group by shared        │
│     `prompts[i]` == `prompts[j]`).                       │
│                                                          │
│  2. For each rollout, apply deterministic hard gates.    │
│     Gate-failed rollouts get reward=0 BUT still get      │
│     ranked (so GRPO retains advantage signal).           │
│                                                          │
│  3. Per-rollout fidelity call:                           │
│       judge → {n_missing_full, n_missing_specificity,    │
│                n_missing_nuance, n_hallucinations, ...}  │
│       fidelity_i = clamp(1 − weighted_errors / max, 0,1) │
│                                                          │
│  4. Per-group ranking call (one per group of G rollouts):│
│       judge → [id_best, id_2, ..., id_worst]             │
│       rank_score_i = (G − rank_i − 1) / (G − 1)          │
│                                                          │
│  5. Combine per rollout:                                 │
│       base_i = 0.5 * fidelity_i + 0.5 * rank_score_i     │
│       reward_i = base_i × length_factor_i × gate_i       │
│                                                          │
│  6. Return [reward_0, reward_1, ..., reward_{N-1}]       │
│     (in the same order as the input completions)         │
└──────────────────────────────────────────────────────────┘
```

## Pre-requisite (must be applied before any v6 training run)

`bash scripts/patch_mlx_lm_lora.sh` — patches mlx-lm-lora's hardcoded
`</answer>` rollout EOS to `<end_of_turn>` so Gemma rollouts stop at the
chat-template turn boundary. Without this, every rollout maxes
`max_completion_length` and the length factor zeroes all rewards. See
LESSONS.md "Investigation 2" for details.

## Component specifications

### 1. Hard gates (deterministic, multiplicative)

`gate_i = 1.0` if **all** of these hold, else `0.0`:

* `not _v5_has_markdown(output_i)` — uses existing
  `langsimp.training.rewards._MARKDOWN_PATTERNS` (8 regex patterns,
  already implemented).
* `not _v5_has_loop(output_i)` — reuse existing `_v5_has_loop`
  (4-gram distinct ratio + repeated-sentence detector).

**Important:** gates zero the reward, but the rollout still
participates in the group ranking call. The ranker should rank
gate-failed outputs at the bottom naturally (gated text is usually
garbage); GRPO advantage is preserved.

The v5 `meaning_gate` (zero if meaning < 0.3) is **removed** in v6.
Catastrophic faithfulness failure is now expressed via the fidelity
score, which can independently go to 0 (and we add a soft fidelity
floor on the combined reward — see §4).

### 2. 8-axis fidelity score (per rollout, 1 judge call each)

Replaces v5's `SemanticPreservationReward`.

Judge prompt asks for atomic-claim decomposition with error counting.
Output is structured JSON:

```json
{
  "missing_full": 2,            // claims entirely dropped (weight 2)
  "missing_specificity": 1,     // claims kept but lost detail (weight 1)
  "missing_nuance": 0,          // claims kept but lost nuance (weight 2)
  "hallucinated": 1,            // unfactual content in output (weight 4)
  "off_topic": 0,               // off-topic but not unfactual (weight 1)
  "factuality_distorted": 0,    // factual but wrong by mistake (weight 4)
  "fidelity_major": 1,          // significant fidelity loss (weight 3)
  "fidelity_minor": 2,          // minor distortion (weight 1)
  "n_source_claims": 7,         // total claims found in source
  "notes": "Optional rationale" // judge's reasoning, for debugging only
}
```

**Scoring:**
```
weighted_errors = 2*missing_full + 1*missing_specificity + 2*missing_nuance
                + 4*hallucinated + 1*off_topic
                + 4*factuality_distorted + 3*fidelity_major + 1*fidelity_minor

max_errors = 4 * n_source_claims   # rough max if every claim were a 4-weight failure

fidelity = clamp(1 - weighted_errors / max_errors, 0.0, 1.0)
```

Weights match the Google paper (2025). They came from "subjective
judgement on the relative severity of each type of error" — we adopt as
starting values; can be re-calibrated later if our audit suggests
otherwise.

**Judge model:** `deepseek/deepseek-v4-pro:gmicloud/fp8` (same as v5,
~$0.0008/call validated cost-effective).

**Prompt template:** see §6 for the exact text to use.

**Caching:** add a fidelity-only cache keyed on `(source, output)`,
analogous to the existing `_judge_cache`. Separate from the v5 cache
(different prompt + response schema).

### 3. Group ranking score (per group, 1 judge call total)

A single judge call ranks all G rollouts in the group. Replaces v5's
`SmoothDifficultyReward`.

**Input** to the judge: source paragraph + all G rollouts (IDs 0..G-1) +
CEFR reference anchors A1/A2/B1 (loaded from `samples.jsonl`, same
anchors the existing `DifficultyRankingTest` uses).

**Output** from the judge: a JSON array of IDs sorted best-to-worst,
e.g. `[3, 0, 7, 1, 5, 2, 6, 4]`. Length G.

**Scoring (linear rank → reward):**
```
for rollout i in group:
    rank_i = position of i in returned array     # 0 = best, G-1 = worst
    rank_score_i = (G - rank_i - 1) / (G - 1)    # 1.0 = best, 0.0 = worst
```

GRPO standardizes (reward − mean) / std internally, so this linear
mapping gives well-spread advantages.

**Judge prompt:** see §6.

**Why ranking and not pointwise:** at G=2, pointwise CEFR scores have a
high chance of identical buckets → advantage 0 → no gradient. Ranking
*forces* differentiation (some rollout is best, some is worst), so
advantage signal never collapses to zero.

**Robustness:**
* If the judge returns malformed JSON, fall back to all-equal rank
  (`rank_score = 0.5` for everyone). Log a warning.
* If the array doesn't contain all G IDs exactly once, ditto.

**Cost:** 1 call/group vs G calls/group (per-rollout judging). 8× cheaper
at G=8, and the ranking task fits cleanly in a single prompt (G outputs
of ~150 words each + anchors = ~3000 prompt tokens, well within DeepSeek's
context budget).

### 4. Length factor (smooth, never zero)

v5 used a hard cliff (ratio>1.8 → 0). v6 uses a Gaussian to preserve
gradient signal even for badly-misshapen rollouts (cold-start matters).

```python
def length_factor(source: str, output: str, sigma: float = 0.4) -> float:
    sw = len(source.split())
    ow = len(output.split())
    if sw == 0:
        return 0.0
    r = ow / sw
    # Gaussian centered at 1.0
    return math.exp(-((r - 1.0) ** 2) / (2 * sigma ** 2))
```

At `r=1.0`: factor = 1.0. At `r=2.0`: 0.29. At `r=4.75` (old stacked
rollouts): 0.0001 (very low but non-zero — preserves rank ordering).

### 5. Combined reward

```python
base_i = 0.5 * fidelity_i + 0.5 * rank_score_i
reward_i = base_i * length_factor_i * gate_i

# Soft fidelity floor: don't reward high-rank-but-hallucinated outputs
if fidelity_i < 0.2 and rank_score_i > 0.5:
    reward_i *= 0.2   # heavy attenuation, but not zero
```

The soft fidelity floor handles the asymmetric-verification edge case:
if all G rollouts are hallucinating, the ranker still picks "best of
bad", but we don't want to reward it normally. Multiplicative
attenuation keeps a small gradient pointing toward less-bad.

## Implementation contract

### File locations

* Code: add to `langsimp/training/rewards.py` (don't create new files —
  follow CLAUDE.md "deep modules" guidance).
* Tests: add to `tests/training/test_rewards.py`.
* Judge prompts: as `_FIDELITY_PROMPT_TEMPLATE` and `_RANK_PROMPT_TEMPLATE`
  constants near the top of the v6 section in `rewards.py`. Don't
  externalize to YAML unless they get >50 lines.

### Required new symbols (public API)

* `class FidelityReward(RewardComponent)` — wraps the 8-axis call,
  returns `fidelity_i ∈ [0, 1]`. `compute(output, ctx, judge)` per the
  existing `RewardComponent` interface.
* `class GroupRankReward` — does *not* fit the per-rollout
  `RewardComponent.compute` shape because it needs the full group.
  Take group-level inputs explicitly: `compute_group(source, outputs:
  list[str], judge)` returns `list[float]` of per-rollout rank scores.
* `class CombinedRewardV6` — top-level reward. Exposes both
  `compute_group(source, outputs, judge)` and the per-rollout
  `compute(output, ctx, judge)` for offline audit.
* `_default_combined_v6() -> CombinedRewardV6` factory.
* `@register_reward_function() def v6_combined_reward(prompts,
  completions, answer, types=None) -> list[float]` — the mlx-lm-lora
  entry point. Internally:
  1. Walk `prompts` looking for runs of identical source. mlx-lm-lora
     calls reward functions with `(batch_size × G)` items where the
     first G share `prompts[0..G-1]` (all the same source), then next G
     share `prompts[G..2G-1]`, etc.
  2. For each group, run one ranking judge call. For each rollout, run
     one fidelity judge call. Cache fidelity per `(source, output)`.
  3. Apply gates + length factor + combination math.
  4. Return rewards in the original order.

### Tests (TDD per CLAUDE.md — write red first)

Required test classes (use `StubJudge` like existing tests):

1. **`TestFidelityReward`** (≥6 tests):
   * Zero errors → fidelity = 1.0
   * All claims missing → fidelity = 0.0
   * Each error type contributes its weight correctly
   * Malformed judge response → fidelity = 0.5 (neutral fallback)
   * `n_source_claims = 0` → fidelity = 1.0 (nothing to lose)
   * Caching: second compute on same `(source, output)` doesn't re-call
     judge.

2. **`TestGroupRankReward`** (≥5 tests):
   * Judge returns valid ranking → rewards are `[1.0, ..., 0.0]` in
     correct order
   * Judge returns malformed JSON → fall back to all-equal (0.5)
   * Judge returns array missing IDs → same fallback
   * Single rollout (G=1) → returns `[1.0]`
   * Group size of 8 with all reasonable judge response → linear spread

3. **`TestCombinedRewardV6`** (≥8 tests):
   * Clean output, A2-quality rank=0, high fidelity → reward ≈ 1.0
   * Markdown detected → reward = 0 regardless of judge calls
   * Loop detected → reward = 0 regardless
   * Very short output (ratio ~0.1) → length factor ~0 → reward ~0
   * Slightly long output (ratio 1.3) → reward ≈ same as ratio=1.0
     (gaussian fall-off is gentle there)
   * Soft fidelity floor: rank=0 but fidelity=0.15 → reward attenuated
     to ~20% of unattenuated
   * Output order preserved when called with multiple groups
   * `compute_group` returns G floats for G outputs

4. **`TestV6RegisteredFunction`** (≥3 tests):
   * Group detection on `prompts = [src1, src1, src2, src2]` (G=2,
     batch=2) → 2 group calls + 4 fidelity calls
   * Group detection on `prompts = [src1, src1, src1, src1]` (G=4,
     batch=1) → 1 group call + 4 fidelity calls
   * Result list length matches input completions list length

### Judge prompt templates (§6)

**Fidelity prompt** (replaces the current `_JUDGE_PROMPT_TEMPLATE`'s
`f`/`h` axes). Should return ONLY the JSON object specified in §2.

```
You are auditing a text simplification for fidelity. You will return ONLY a JSON object.

First, decompose the SOURCE into atomic claims (one fact each). Then check each claim against the OUTPUT.

Then count errors across these categories:

INFORMATION LOSS (claim from source missing or weakened in output):
- missing_full: claim entirely absent (weight 2)
- missing_specificity: claim present but lost specific detail (weight 1)
- missing_nuance: claim present but lost nuance or connotation (weight 2)

INFORMATION GAIN (content in output not in source):
- hallucinated: unfactual claim invented (weight 4)
- off_topic: present but irrelevant tangent (weight 1)

DISTORTION (claim present but altered):
- factuality_distorted: claim present but factually wrong (weight 4)
- fidelity_major: significant fidelity loss (weight 3)
- fidelity_minor: minor wording shift (weight 1)

SOURCE:
{source}

OUTPUT:
{output}

Respond with ONLY this JSON, nothing else:
{
  "n_source_claims": <int>,
  "missing_full": <int>,
  "missing_specificity": <int>,
  "missing_nuance": <int>,
  "hallucinated": <int>,
  "off_topic": <int>,
  "factuality_distorted": <int>,
  "fidelity_major": <int>,
  "fidelity_minor": <int>
}
```

**Ranking prompt:**

```
You are ranking text simplifications. The TARGET level is CEFR A2 (Elementary English) while preserving source meaning.

Reference levels (for calibration):

A1 example:
{a1_anchor}

A2 example (TARGET):
{a2_anchor}

B1 example:
{b1_anchor}

SOURCE:
{source}

CANDIDATE SIMPLIFICATIONS:
[0] {output_0}
[1] {output_1}
...
[N] {output_N}

Rank candidates from best to worst as A2 simplifications of the source. Favor outputs that:
- Use A2-appropriate vocabulary and sentence structure
- Preserve the source's meaning (rank outputs lower if they invent or distort facts)

Respond with ONLY a JSON array of candidate IDs, best first:
[<id_best>, ..., <id_worst>]
```

Anchors are loaded from `samples.jsonl` via the existing helpers
(`_load_cefr_anchor("A1")` etc).

## Pre-flight validation (mandatory before training)

The implementing agent MUST run these checks before launching any v6
GRPO training run. Each is a single Python script — code them as
`scripts/validate_v6.py` or inline.

1. **Fidelity ranks known examples correctly.** Score the 10 audit
   prompts × 5 variants (in `/tmp/reward_audit5.json` and
   `/tmp/vocab_validate.json`) with the new `FidelityReward`. Compute
   Spearman ρ against the subjective ranks already recorded in
   `/tmp/v5_test.py`. Threshold: mean ρ ≥ 0.5 across the 10 prompts.

2. **Group ranking respects fidelity.** Submit 8 deliberately-crafted
   outputs for a single prompt where one is a clear hallucination
   (invent a fake fact like "Posad was founded by Christina Pagel").
   Verify the hallucination ranks at position 7 (worst). Repeat for 3
   prompts.

3. **Group ranking matches subjective.** On the 5 audit prompts, submit
   all 5 known variants as a group; check the resulting permutation
   matches subjective rank within ±1 position for the top and bottom
   slots.

4. **Smoke train.** Per CLAUDE.md, run 2 iterations of GRPO with the
   v6 reward and inspect:
   - `train/avg_tokens` < 200 (EOS patch active)
   - `train/reward_mean` is a real number, not zero
   - Rollout text is a single A2-style response (no stacked turns)

Only after all four pass should the implementing agent launch a longer
GRPO run.

## What the agent should NOT do

* Don't drop the v5 hard gates (markdown, loop). They're cheap and
  catch real degenerate outputs.
* Don't add new judges or signals beyond the two specified here without
  first re-running the §"Pre-flight validation" against the new
  candidate.
* Don't change reward weights from 0.5/0.5 without empirical evidence
  on the 10 audit examples that asymmetric weighting helps.
* Don't introduce the previous vocab reward (`VocabSimplicityReward`)
  back into the combined stack. It had negative correlation with
  subjective quality on held-out examples (see LESSONS.md
  "Investigation 2").
* Don't bypass the `scripts/patch_mlx_lm_lora.sh` step. If the patch
  isn't applied, every rollout will max `max_completion_length` and
  destroy the length factor signal.

## Suggested training launch (post-validation)

```bash
bash scripts/patch_mlx_lm_lora.sh   # required, idempotent
MEANING_JUDGE_BACKEND=openrouter \
MEANING_JUDGE_MODEL="deepseek/deepseek-v4-pro:gmicloud/fp8" \
uv run python -m langsimp.training.runner grpo \
  --model adapters/sft_n750_fused \
  --data data/grpo \
  --resume-adapter "" \
  --iters 100 \
  --batch-size 1 \
  --lr 1e-6 \
  --lora-layers 16 \
  --group-size 4 \
  --temperature 0.8 \
  --max-completion-length 256 \
  --reward-functions v6_combined_reward \
  --reward-weights '[1.0]' \
  --project lang-simp-grpo
```

Notes on these settings:
* `group-size 4` (up from 2): more rollouts per group = more spread for
  the ranker to work with. Cost stays low because the ranker is a
  single call per group regardless of G.
* `max-completion-length 256` (down from 512): after the EOS patch,
  clean rollouts finish in ~120 tokens. 256 leaves headroom without
  burning compute on stacked rubbish if the model regresses mid-run.
* Resume from `adapters/sft_n750_fused` (best 3-seed mean SFT). Fused so
  policy and reference start aligned (initial KL ≈ 0).

## Expected outcomes (so the agent knows when something is wrong)

* Iter 1: rewards in [0.4, 0.7] range (SFT-n750 is decent A2 already,
  Opus-like reference scores would be ~0.85).
* Rewards trend slowly upward over iters; expect +0.05 to +0.10 over
  100 iters at lr=1e-6.
* KL grows gradually — should reach ~0.1-1.0 by iter 100, NOT 0 flat.
  If KL stays 0 throughout, the policy isn't moving (something is
  zeroing the gradient — most likely the EOS patch wasn't applied).
* Loss is non-zero on most iters. Steady `loss=0` means no advantage
  signal — group ranking isn't generating spread, probably because the
  ranking judge is returning malformed JSON or the fallback path is
  firing.
