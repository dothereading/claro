# Strategy brief: where to invest next on a 1B A2-simplifier

Looking for an outside read on where to invest the next chunk of effort.

## Project

Fine-tune Gemma-3-1B-it to rewrite arbitrary English (Wikipedia paragraphs) at CEFR A2 level — short simple sentences, common ~1500 vocabulary, **faithful to source**, adult tone. Apple Silicon compute (M-series), no NVIDIA. Judges/teachers via OpenRouter API (DeepSeek-v4-pro, Haiku, Opus).

Pipeline so far:
- **Distillation**: Opus-4.5 simplifications of ~750 Wikipedia paragraphs → `data/sft.jsonl`
- **SFT**: Gemma-1B LoRA (rank 8, 16 layers), 750 iters on those pairs
- **DPO**: tried twice, both regressed; abandoned
- **GRPO**: three iterations (v7/v8/v9), the focus below

Eval: 50 held-out paragraphs from `data/eval.jsonl`. A DeepSeek judge classifies each output as A1/A2/B1 using a 3-window sliding classifier with A1/A2/B1 anchor examples. SEM at n=50 is ±7.1pp.

## Current metrics (n=50)

| Stage | % A2 | % A1 (too easy) | % B1 (too hard) | % A2+B1 (readable + faithful) |
|---|---|---|---|---|
| Gemma-1B base (no FT) | 6.7% | 93.3% | 0% | 6.7% |
| **SFT_n750 (1B)** | **68%** | 20% | 12% | 80% |
| GRPO v7 (sparse-geometric reward, 100 iters) | 48% | 16% | 36% | 84% |
| GRPO v8 (ranker prompt tweak, 100 iters) | 52% | 16% | 32% | 84% |
| GRPO v9 (more tweaks, 200 iters, temp=1.0) | 58% | **12%** | 30% | **88%** |
| Gemma-3-4B SFT_n1500 | ~70% | (similar) | | |

Reading: SFT wins on strict-A2 (68%). GRPO_v9 wins on "readable + faithful" (88%, +8pp over SFT, outside SEM), but trades 10pp of A2 → B1.

## The GRPO design we used (v7-v9)

One judge-backed ranker per group of G=8 rollouts. The judge sees source + 8 candidates, returns a JSON array of candidate IDs best→worst. We map rank → reward:

```
sparse-geometric: [1.0, 0.5, 0.25, 0.125, 0, 0, 0, 0]
× Gaussian length factor (σ=0.4 on output/source word ratio)
× hard gates (no markdown, no n-gram loops)
```

Ranker prompt evolved across versions: v7 was a basic "rank A2 simplifications, faithfulness first then accessibility"; v8 added "don't over-pack sentences, break dense info, give context from source"; v9 hardened the anti-invention clause ("importing context is invention even if true in the wider world") and bumped rollout temperature 0.8 → 1.0.

Trainer: `mlx-lm-lora`'s GRPO. Two bug patches were required before it would train at all:
- **FP16 → FP32 logits cast** in `get_per_token_logps` (FP16 max ~65504 saturated on Gemma's 256K vocab → log_softmax(inf) → NaN gradients by iter ~15).
- **`--importance-sampling-level sequence` (GSPO)** instead of the default per-token. Per-token NaN'd earlier; sequence-level is stable for 100+ iters.

lr=1e-6, G=8, max_completion=512, batch_size=1, lora_layers=16.

## What worked

**Fidelity improved consistently across GSPO runs.** This is the most important finding and it's hidden by the strict-A2 metric. Concrete wins, identical across v7/v8/v9:
- Removed hallucinations (TND removed "he does not know how to do it"; Selling Kabul fixed wrong gender).
- Preserved named entities the SFT model paraphrased away (media mogul vs media tycoon; State Preceptor vs leader of Yuan dynasty; screenplay vs script; World War III).
- Fixed factual errors (Melicope's "six petals and six stamens" → "four petals, four sepals, four stamens"; matches source).
- Dropped chatbot preambles ("Here is the text rewritten…") and redundant cappers.

A1 share dropped 20% → 12% across SFT→v9. That 8pp is outputs where the source's content was being destroyed.

**Trainer stable after the patches.** No NaN, healthy KL drift, reward μ pegged at theoretical max from iter 1 (μ=0.234 for sparse-geometric / G=8). cov=100% throughout (every group ranked successfully).

## What didn't work

**Strict A2 dropped 10pp** (68% → 58%). The fidelity gains pushed outputs denser → classifier reads dense-but-faithful text as B1.

**Prompt iteration past v7 mostly produced classifier noise.** Among 10 qualitative comparisons:
- **5 of 10** outputs were word-for-word identical across v7/v8/v9 but classifier flipped labels (A2/A1/B1 on the same text).
- **2 of 10** showed real v9 improvements (e.g., sentence-completion fix).
- **2 of 10** showed v9 *regressions* (Margot Sponer: v9 dropped "Battle of Berlin" detail; Ali Darassa: v9's hallucinated geographic context got worse, see below).
- **1 of 10** had trivial rephrases of equivalent quality.

**The anti-invention prompt change failed its direct test.** This was the specific failure mode I built the v9 prompt change to fix:
- v7: "He is a Fula, which means he is from the ethnic group of the same kind as the people of **the Nile River**." (invented)
- v8: "…the people of **Senegal**." (invented, different country)
- v9: "…the people of **the Gambia and Senegal**." (invented, *two* countries; v9's prompt change made it strictly worse)

Source says only "He is an ethnic Fula and his UPC is largely Fula" — no geographic context. The judge is rewarding context-for-hard-terms and missing that the geography is an outside fact.

**Reward saturation kills the gradient signal.** Sparse-geometric at G=8 has theoretical mean 0.234. By iter 1 we're already at μ=0.229; from iter 5 onward μ ≈ 0.23 with σ ≈ 0.32 essentially flat. GRPO advantage = (r − group_mean) / group_std collapses to whatever the judge's noisy ranking calls when all candidates hit the gates. Training longer just averages over more noise.

## Qualitative examples

### Faithfulness win — Tomorrow Never Dies

**Source (66w)**: "...follows Bond as he attempts to prevent Elliot Carver, a power-mad media mogul, from engineering world events to initiate World War III."

**SFT (82w)**: "...Carver is a rich media tycoon. He wants to start World War III, **but he does not know how to do it**. [...] The film is the 18th James Bond film."

**GRPO v7/v8/v9 (58w, identical)**: "Bond tries to stop Elliot Carver, a wealthy media mogul. Carver wants to start World War III."

GRPO killed the "does not know how" hallucination, killed the redundant capper, kept "media mogul" / "World War III" / "screenplay" exactly. **30% shorter, more faithful.** But classifier called SFT A2 and GRPO A1 here (A1 = "too easy").

### Faithfulness win — Melicope

**Source**: "...four sepals, four petals and four or eight stamens... fruit composed of up to four follicles."

**SFT (78w, A2)**: "Each stalk has **six petals and six stamens**. The fruit is made of up to four **small groups of flowers**." (botanically wrong — invented petals/stamens count, "groups of flowers" wrong for follicles)

**GRPO v7-v9 (78w, A1 or B1 depending on classifier run)**: "Each flower has **four petals, four sepals, and four stamens**. The fruit is made of up to four small sacs." (correct numbers, "small sacs" passable for follicles)

### Faithfulness regression — Ali Darassa

This is the case where GRPO's "add context for A2 readers" prompt directive fights with "don't invent." Across v7/v8/v9 the model invents a Fula geographic origin not in source, and the v9 prompt change to suppress this made it worse, not better. See "What didn't work" above.

## Diagnosis

The bottleneck is **misalignment between two notions of "good A2"**:

- **Ranker reward** rewards faithfulness (keeping specific terms, named entities, numbers) and clean output. That naturally pushes outputs toward denser, B1-flavored prose.
- **Difficulty classifier** rewards short simple sentences with common words. It reads denser-but-faithful text as B1, even when the text is factually superior to a shorter A2 version with a missing or wrong fact.

Compounding problems:
- **Reward ceiling.** Sparse-geometric rewards saturate at μ=0.234 from iter 1. GRPO has nothing to optimize once every group hits the ceiling.
- **Judge variance is large.** Same exact text classified A1/A2/B1 across runs of the same eval. Model differences past v7 are smaller than classifier re-roll noise.

## Options I'm weighing (open question)

1. **Ship SFT.** Best strict-A2 (68%). Loses the fidelity wins. The 12% info-lossy A1 share (Melicope's "six petals" world) is a real product downside.
2. **DPO from v9 ranker.** Use v9 ranker to score K rollouts per prompt offline, take top-1 = chosen, bottom-1 = rejected, generate ~1500 pairs (~$5-10 in judge calls), DPO train from sft_n750. Avoids GRPO's reward-saturation. DPO infrastructure is built. We have two existing DPO adapters (both regressed earlier — but those used different rejected-pool strategies, not a ranker-scored pool).
3. **GRPO on 4B.** ~30-70 hours of training (4B at G=8 ≈ 15-20 min/iter on Apple Silicon). Same reward saturation expected. 4B SFT already ~70% A2 so headroom is small.
4. **Different reward design.** Combine v9 ranker with an explicit A2-difficulty term in the reward (so fidelity wins don't sacrifice level alignment). Risk: more reward components = more variance, possibly worse than v9.
5. **Improve the difficulty classifier.** It's the source of strict-A2 numbers and apparently the source of half our "v9 noise." Re-rolling it with a stronger judge or more anchors might both tighten the metric and reveal v9 is actually better than current numbers say.

## Questions for you

1. **Is the option ranking right?** I lean toward (2) DPO with v9-ranker-scored pairs as the highest-EV next step. Is there an option I'm not seeing?
2. **Reward design**: any obvious flaw in the v7-v9 ranker that's causing the saturation, beyond the "all groups hit theoretical max" problem? Should we try non-sparse (e.g., calibrated A2-difficulty score per rollout)?
3. **Validation methodology**: the classifier is clearly noisy. Is there a cheap way to get a more stable per-output A2 score so we can actually distinguish v7 from v9 from SFT without burning 100+ judge calls per eval?
4. **For the Ali Darassa-style invented-context failure**: is the right fix a prompt change, a separate reward component that explicitly checks "is every fact in the output also in the source," or something else?
