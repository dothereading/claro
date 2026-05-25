# Next investigations

Where the budget goes next, ranked by expected leverage given the GRPO ceiling
at 83.3% A2 and the residual token-level fidelity slips at 1B parameters.

## 1. Bigger base model

**Why:** The remaining errors are LM failures (duplicated entities, broken
phrases), not difficulty failures. We hit a 1B-parameter ceiling, not a recipe
ceiling.

**Investigate:** Train the same SFT recipe on a 4B base (Gemma-3-4B-it via
MLX when it lands, or Qwen-2.5-3B if MLX support is mature). Hold data,
hyperparams, and eval set constant to isolate the size variable.

**Measure:**
- Held-out A2% on the same frozen eval set (direct comparison vs current numbers)
- **Fidelity-slip rate** — per-output count of duplicated entities, name swaps,
  number errors. Expect this to drop more than A2% climbs.
- Cost-per-A2-point: if 4B costs 4× compute for +10 points, that's worse
  return than another doubling of SFT data. Price it out before committing.

**Status:** Gated on MLX support landing for Gemma-3-4B (or willingness to
switch base family).

## 2. SFT data scale + quality

**Why:** Our 621-row run was capped by OpenRouter cost, not diminishing
returns. The 90→558 jump killed the val-loss climb entirely and train loss
was still falling at iter 600. We don't know where SFT plateaus.

**Investigate (in parallel):**
- **Scale**: train SFT on 250 / 500 / 750 / 1000 / (and beyond if still climbing) rows.
  Plot held-out A2% vs row count.
- **Quality**: generate two Opus completions per prompt, keep the one with
  higher meaning-judge `f+h` score (best-of-2 distillation). Compare
  best-of-2 at N rows vs single-sample at 2N rows for equal cost.

**Measure:**
- Held-out A2% at each scale; look for inflection point
- Val loss curve — fully flat at 1000 rows, or still climbing?
- Best-of-2 vs single-sample at equal Opus spend: which wins?

**Status:** **Active.** Tackling scale dimension first.

## 3. Fidelity-targeted reward in GRPO

**Why:** Current GRPO rewards (40% meaning / 25% difficulty / 15% repetition /
10% vocab / 5% length / 5% markdown) catch catastrophic faithfulness failures
via the meaning gate at 0.3, but don't sharply punish a single mis-copied
name or duplicated entity. The Christina-Pagel-style "Rebecca Shipley and
Rebecca Shipley" pattern is the addressable residual.

**Investigate:** Add a *deterministic* fidelity reward alongside the LLM
meaning judge:
- Named-entity overlap (spaCy NER source ∩ output, Jaccard)
- Number-string preservation (`\d+(\.\d+)?(%|m|km|years|…)?` must all appear)
- Duplicate-noun detector (penalise `\b(\w+)( and \1)+`)

Microsecond cost per rollout, clean per-step gradient, targets the exact
failure mode that survives current GRPO.

**Measure:**
- Audit current GRPO outputs by hand: how many have entity errors? number
  errors? duplicates? That's the *addressable* set.
- Run GRPO with the new fidelity reward (weight ~0.15, reallocate from
  meaning since they overlap). Compare A2% **and** per-error counts.
- Win condition is *not* higher A2% — it's lower fidelity-slip rate at
  constant A2%.

**Status:** Pending. 1-day implementation plus one GRPO run.

## What I would *not* spend budget on

- More DPO experiments. Two coherent attempts both regressed; the failure
  mode is well-characterised (preference axis ≠ eval axis at this gap size).
- Longer GRPO runs with the same reward stack. KL≈0 means the bottleneck is
  signal density per group, not compute.
- More reward components without an audit of what each catches.

## Sequencing

1. **#2 first** — cheapest, fastest, we genuinely don't know where SFT plateaus.
2. **#3 next** — independent of model size, 1-day side project.
3. **#1 last** — biggest swing, most expensive, ecosystem-gated.
