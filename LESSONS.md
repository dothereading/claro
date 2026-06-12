# Lessons from training a small Gemma to write A2 English

> The initial table below was a **single-seed, 30-record gemma-judge** snapshot.
> Later investigation (see "Investigation 2: noise, judges, and reward design"
> at the bottom) showed all of those numbers had wide CIs and the GRPO uplift
> over SFT was inside the noise floor. Read both sections together.

A frozen 30-paragraph held-out set, judged by `gemma-4-26b-a4b`, tracks A2 hit
rate across stages.

| Stage | % A2 | % too easy (A1) | % too hard (B1) | Length ratio |
|---|---|---|---|---|
| Base (Gemma-3-1B-it) | 6.7% | 93.3% | 0% | 0.68 |
| SFT (621 Opus pairs, 600 iters) | 76.7% | 13.3% | 10.0% | 1.09 |
| DPO v1 (4-strategy rejected pool) | 50.0% | 50.0% | 0% | 1.07 |
| DPO v2 (self-rollout rejected pool) | 60.0% | 40.0% | 0% | 0.89 |
| **GRPO (from fused SFT, 100 iters)** | **83.3%** | 10.0% | 6.7% | 1.07 |

## What actually moved the needle

**SFT does the heavy lifting.** Base→SFT is +70 points of A2 hit-rate. The
base model wasn't failing at vocabulary — it was failing at *register*. Default
behaviour was telegraphic A1 prose ("He is a leader. He is from Nigeria."),
length ratio 0.68 (dropping a third of source content). SFT taught paragraph
prose, content preservation, and A2 vocabulary. Everything else is fine-tuning
of fine-tuning.

**GRPO converted +6.7 more points** from a tiny policy adjustment. W&B showed
KL ≈ 0 and loss = 0 throughout — the model barely moved from SFT. But the
small movement was *correctly directed*: borderline outputs flipped onto the
A2 line in both directions (less A1 over-simplification, less B1
domain-vocab leakage). This is GRPO at its best — refine, don't overshoot.

## What didn't work (and why)

**DPO regressed twice.** Two failure modes worth naming:

*v1 — pool too easy.* Four rejected strategies (weak-distill, summarize,
eli5, clarify) made Opus-vs-rejected trivially separable. Train accuracy hit
1.0 by iter ~140 with no useful gradient remaining, but margin kept growing
to 18 — the model collapsed onto "look Opus-y at the surface" features and
started dropping content. Result: -26.7 A2 points.

*v2 — quality gap too small.* Self-rollout pool (chosen = Opus, rejected =
our own SFT model) gave healthy training (val acc 0.79, margin grew
gradually 1.6 → 3.3, val loss decreased smoothly). But the held-out eval
still regressed -16.7 points. The chosen-vs-rejected difference between Opus
and SFT-Gemma is mostly stylistic richness, not faithfulness — so DPO
optimized the wrong axis. The same fidelity slips ("Each room can hold 1,700
people") persisted post-DPO.

**Generalisable lesson:** DPO works when the chosen-vs-rejected axis matches
your evaluation axis. Healthy training metrics ≠ held-out gain. When in
doubt, prefer GRPO with explicit reward shaping over DPO preference
learning, especially at small model sizes.

## Footguns hit during the run

**bf16 KL overflow in GRPO.** Resuming policy from an SFT adapter while
leaving the reference model as bare base produces astronomical initial KL
(val loss 2.26 × 10²² → NaN). Fix: fuse the SFT adapter into a full model
and use *that* as both `--model` and (implicitly) the reference. Initial KL
≈ 0, training proceeds normally.

**Wikipedia random-summary 4xx kills the iterator.** A title containing `/`
got into a redirect URL and 400'd, halting a 1000-row distill at 478. The
retry loop only handled 429/5xx. Now any 4xx is treated as "skip this draw"
(test + fix in `c0f1a2d`).

**Stale DPO `chosen` after regenerating SFT data.** After deleting the old
194-row `sft.jsonl` and generating 621 fresh rows with the iterated prompt,
the old `dpo.jsonl` still pointed at chosen outputs from the *old* prompt.
Always rebuild DPO data when SFT data changes — otherwise DPO actively
unlearns SFT.

**OpenRouter "Key limit exceeded (monthly limit)"** is a per-key spend cap,
not a credit balance. Raising it is done at the key settings page, not by
adding credits.

## The 1B-parameter ceiling

GRPO got the *register* and *length* right basically everywhere. What's left
is genuine LM failure — specific-token errors no reward signal can repair:
"*Rebecca Shipley and Rebecca Shipley* also lead this research group", "*The
name 'Lord Cornwallis' is in doubt*" (collapsing a "historians dispute
whether he used the house" claim into a meaningless phrase about the name).
These are 1B-parameter limitations, not difficulty-calibration problems.
Beyond this point, the next 5 percentage points of held-out A2 come from a
bigger model, not from more training tricks.

---

## Investigation 2: noise, judges, and reward design

Followups to the table above. Most of the headline numbers got smaller —
not because the model regressed, but because we instrumented the
measurement noise we had been ignoring.

### Judge variance is huge at the A2/B1 boundary

Three-way audit on 30 GRPO outputs across `gemma-4-26b-a4b` (LM Studio),
Haiku 4.5, Sonnet 4.5, DeepSeek v4 Pro, and Opus:

| Pair | Agreement |
|---|---|
| gemma vs Opus | 53% |
| Sonnet vs Opus | 48% |
| DeepSeek vs Opus | 45% |
| gemma vs DeepSeek | 83% |

The smaller judges cluster together; Opus is the outlier (stricter). The
disagreement is one-directional: small judges call B1-difficulty text "A2".
No single LLM judge is ground truth — **CEFR classification at this
boundary is genuinely subjective**.

### Seed variance dominates single-run differences

Re-trained each scale (n=250/500/750/1000/1250/1500) on 3 seeds. SDs across
seeds grew with iters: 2.0 pp at n=750 → 9.2 pp at n=1500. The single-seed
n=1500=78% point we celebrated was the high tail of (78, 66, 60). The
honest 3-seed plot plateaus around 70-74%, not 78-80%.

**Lesson: any one model's eval is a wide-CI draw at N=50 records. Report
3-seed means with SE bars or accept ±10 pp uncertainty.**

### v3 vocab reward had NEGATIVE correlation with quality

Manual audit on 25 outputs (5 prompts × 5 variants), ranked by my reading
against Opus reference. Spearman ρ of each reward signal vs subjective
rank:

| Signal | Mean ρ (held-out 5) |
|---|---|
| meaning (judge) | +0.81 |
| **entity preservation** | **+0.57** |
| difficulty (judge) | +0.15 |
| H1c per-sentence vocab cap | -0.02 |
| H2 source-relative vocab | -0.04 |
| **v3 vocab reward (current)** | **-0.57** |

The current vocab reward *punishes* Opus's correct A2-style behavior:
glossing technical terms ("mausoleum (a big building for the dead)") adds
tokens that aren't in the top-3000, which v3 penalizes. The "right" output
for an A2 simplification routinely scored lower than a content-dropping
SFT output. v6 drops vocab entirely.

### Repetition and markdown rewards never discriminated

Across all 25 audit outputs, both stayed at 1.000. They were inflating the
combined-reward baseline without contributing gradient. v6 converts both
to hard gates: 0 reward if triggered, no contribution otherwise.

### mlx-lm-lora's hardcoded `</answer>` EOS broke every prior GRPO run

The GRPO trainer hardcodes `end_answer_token="</answer>"` as the rollout
EOS (designed for math chain-of-thought tasks). For Gemma:
* `</answer>` encodes to 3 tokens → `tokenizer.add_eos_token` raises
  `ValueError` → `use_eos_token=False`.
* mlx-lm's natural EOS for Gemma is `<eos>` (id 1), but the model emits
  `<end_of_turn>` (id 106) at turn end — never `<eos>`.
* Generation runs to `max_completion_length` (512) every time. The model
  dutifully *stacks* 4-5 complete responses separated by
  `<end_of_turn>` markers inside the 512-token budget.
* Reward function sees the stacked text. Ratio ≈ 5× source → length gate
  zeroes the reward. Every rollout = 0. No gradient. `train/kl=0` flat.

Patch: `scripts/patch_mlx_lm_lora.sh` overrides the default to
`<end_of_turn>`. **Run after every `uv sync`** — idempotent. Smoke test
with `train/avg_tokens` < 200 confirms it's working.

### SFT data scaling plateaus around 70-74% A2

3-seed × 6 scales (n=250 → n=1500) shows mean A2% climbing 63% → 74% from
n=250 to n=750, then **flat or declining** from n=750 onward (n=1500 mean
68%). All means within their own SD bars from n≥750. The "iters=n" training
protocol overtrains larger datasets — n=1500 bottoms val loss at iter 450
of 1500, so most of the run is amplifying seed-specific weights, not
learning. Suggests SFT is mostly tapped at this scale for this model.

### DPO is the wrong tool here at all sizes tried

Both DPO attempts regressed vs SFT-only. The chosen-vs-rejected gap (Opus
vs anything else) is mostly stylistic richness, not faithfulness. DPO
optimizes "look like Opus at the surface", which hurts when the model
can't actually be Opus-quality. GRPO with explicit reward shaping is the
right tool for refining the 1B model further.

### Real verifier improvements come from decomposition

Group-ranking via single judge call + atomic-claim fidelity scoring (per
the Google self-refinement paper) are the verifier directions worth
investing in. See `V6_SPEC.md`. The v3-v5 stack treats "is this A2?" as a
single noisy classification call; v6 splits it into a structural ranking
call (style/A2-ness) and a fact-by-fact analysis call (faithfulness), and
combines them. Each sub-call is easier to verify than the joint
"is this output good?" question — Wei's asymmetry-of-verification idea
applied at the reward-component level.

## Investigation 3: v7 GRPO — making the trainer work, then learning the reward was misaimed (2026-06)

### mlx-lm-lora's GRPO has two latent bugs for Gemma

**FP16 logits cast NaN's gradients.** `grpo_trainer.py:92` does
`logits = model(inputs).astype(mx.float16)`. Gemma's vocab is 256K; with
BF16 weights and any training that pushes a logit above FP16's max
(~65504), the cast yields `inf`, then `nn.log_softmax(inf) → NaN`, then
NaN propagates into `log_ratio → loss → gradient`. Symptom: v7 GRPO at
G=8 NaN'd by iter ~15 regardless of lr or importance-sampling level.
Fix: patch to `.astype(mx.float32)` — see `scripts/patch_mlx_lm_lora.sh`.

**Default `--importance-sampling-level=token` NaN's earlier than
`sequence` (GSPO).** Per-token importance weights amplify any
per-token log-prob outlier; sequence-averaged weights tolerate the
same outliers. Both eventually fail without the FP32 patch, but
sequence-level survives ~10 iters longer in our setup. With FP32 patch
applied, sequence-level is the only one that trains stably for 100+
iters.

### Offline ranker validation doesn't predict online training utility

The v7 sparse-geometric ranker validated cleanly offline: Spearman ρ
= +0.87 vs subjective rank, hallucinated outputs always last, top-1
agreement 4/6 across 3 examples × 2 calls (round 2 prompt: top-4 set
agreement 11/12). And online training was stable: μ=0.229, σ=0.329,
cov=100% throughout 100 iters. But the trained adapter underperformed
its SFT base by −20pp on strict A2 hit-rate (80-prompt eval, 68% → 48%).

The mechanism: v7 rewarded "faithfulness" (preserve named entities,
specific terms) and "shorter is better when faithful" simultaneously.
Together those two pushed outputs into B1 register — dense sentences
that pack 2-3 specific terms each. The reward shape was internally
consistent and the trainer learned to maximize it; the reward shape
itself just didn't aim at A2.

### Sample size on the eval matters more than I assumed

The same adapter pair (sft_n750, grpo_v7) measured at n=30 vs n=80:
- n=30: GRPO +14pp on A2 (40% → 54%)
- n=80: GRPO −20pp on A2 (68% → 48%)

The first 30 prompts in `eval.jsonl` happened to be the ones where
SFT struggles and GRPO recovers; the broader 80 reveals the
generic-case regression. Lesson: a 30-prompt eval can flip directionally
between adapters. Run 80 minimum before committing to a training
direction.

### Strict-A2 may be the wrong target — A2-or-B1 reframes the result

Under "strict A2 only" the SFT model wins 68% vs 48%. Under "A2 OR
B1 acceptable" (i.e. readable + factually preserved, B1 vocab tolerated)
the GRPO model wins 80% vs 84%. The difference matters because:

- A1 outputs (over-simplified) destroy source information — "Carver
  is a powerful businessman" instead of "media mogul"; "Each stalk
  has six petals" instead of "four petals, four sepals".
- B1 outputs (dense vocab) preserve information but ask more of the
  reader.

For most A2-simplification use cases (an A2 learner reading general
content), preserving facts at B1 vocab beats destroying them at A1.
The 1B SFT model's 20% A1 share is hidden tax — that 20% is outputs
where the source's value got lost. v7 trades 4pp of that for an extra
24pp of B1; net +4pp on "readable+faithful" but −20pp on the
classifier's strict A2 bucket.

### DeepSeek judges hang in GRPO training context, Haiku doesn't

`deepseek/deepseek-v4-pro:gmicloud/fp8` (and `deepseek-v4-flash`,
and bare `deepseek-v4-pro`) work fine for offline batch evaluation
(`check_v7_stability.py`, `eval_harness`) but consistently hang on
the first judge call inside an mlx-lm-lora training process —
SSL socket `poll()` stuck until our 45-180s timeout fires. Likely
DeepSeek's reasoning-model output takes >timeout for an 8-candidate
ranking prompt that's ~1500 words. Haiku-4.5 returns in ~5s for
the same prompt and trains reliably. Workhorse for training =
Haiku; gold-standard for evaluation = DeepSeek.
