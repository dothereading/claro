# v10 Reward — Findings & Open Questions (for outside review)

A self-contained briefing. Goal: get expert guidance on the reward design and
the GRPO setup before committing to longer/more expensive runs.

## 1. Project context

Fine-tuning **Gemma-3-1B** (LoRA, MLX, Apple Silicon, `mlx-lm-lora`) to rewrite
complex English at **CEFR A2** ("Elementary"). Pipeline: data → SFT → (DPO) →
GRPO → eval.

- **SFT baseline (`sft_n750`)**: lifts held-out **A2 hit-rate from 6.7% (base) → 76.7%**.
  A2-ness is measured by an LLM "difficulty classifier" (DeepSeek + few-shot
  CEFR anchors) that labels a text A1/A2/B1/B2+.
- **Remaining SFT gap (the target of this work)**: *faithfulness* — the SFT model
  drops or invents facts (e.g. invents "six petals and six stamens"; hallucinates
  a James-Bond plot clause; adds a geographic origin not in the source). Difficulty
  is mostly solved; faithfulness is not.

GRPO history: prior rewards (v7–v9) mapped a judge's **ranking** of G=8 rollouts to
a sparse-geometric vector `[1, .5, .25, .125, 0,0,0,0]`. Problem: the group **mean is
constant by construction** (≈0.234), so after GRPO normalizes advantages
`(r-mean)/std`, the signal is driven entirely by judge tie-break noise once all
candidates pass the gates. We replaced ranking with **per-rollout cardinal scores**.

## 2. The v10 reward (what we built)

```
reward(source, candidate) = level_band × vocab × fidelity × gates      # all ∈ [0,1]
```

Multiplicative: any component can veto, none can rescue. Components:

- **level_band** (deterministic, free): targets the *readability band* of A2, not
  max simplicity. `trapezoid(FRE) × trapezoid(MSL)` where FRE = Flesch Reading Ease
  (textstat), MSL = mean sentence length in words (spaCy). Trapezoid = 1.0 inside the
  anchors' [q1,q3], linear falloff, floor 0.2. Degenerate (<2 sentences or <10 words)
  → 0.04. Calibrated band: **FRE [73.1, 87.2], MSL [9.0, 11.8]**.
- **vocab** (deterministic, free): per sentence, count "hard" words = lemma not in a
  ~3.8k A2 allow-list, with exemptions (proper nouns, numbers, and **any lemma already
  in the source** — so domain terms it must keep aren't punished). `sentence_score =
  1.0 if hard≤1 else 0.5^(hard-1)`; `vocab = max(0.2, Π sentence_scores)`.
- **fidelity** (one LLM judge call/rollout): judge extracts source facts (recall) and
  unsupported claims (invention). Scored in code:
  ```
  recall_term = max(0.2, present_facts / total_facts)      # 1.0 if no facts
  halluc_term = 0.1 ** min(n_unsupported, 2)               # 1.0 / 0.1 / 0.01
  fidelity    = recall_term × halluc_term                   # ≈ 0.002 … 1.0
  ```
  Soft (×0.1 per claim, not hard 0) so a single misflag can't zero a good rollout.
  Fails OPEN (term=1.0) on judge error, counts failures. Verdicts cached (SQLite,
  keyed on sha256(model+prompt_version+source+candidate)).
- **gates** (hard 0/1): markdown/formatting detection + n-gram loop detection
  (lifted verbatim from v9). Gated rollouts short-circuit before the judge call.

**GRPO config**: start from `sft_n750`; lr=1e-6; **G=8**; batch_size=1; lora_layers=16;
max_completion=512; **temperature=1.0**; `--importance-sampling-level sequence` (GSPO);
data **not shuffled**.

**Judge**: `anthropic/claude-haiku-4-5` via OpenRouter, temp 0, **structured outputs**
(`response_format: json_schema, strict:true`) → **0% parse failures** (was ~4–8% free-form).

## 3. Data & two alignment decisions (these mattered)

- **Band anchors**: Kaggle `cefr-levelled-english-texts` A2 split (272 texts). But A2
  there is *conversational* (emails/essays, high FRE). Calibrating on Kaggle alone sent
  **~30% of gold Opus A2 Wikipedia references to the band floor** (expository prose reads
  harder). Fix: **blend** Kaggle A2 + Opus refs (eval `simple` + dpo `chosen`) → floor
  hits drop to ~10%.
- **Vocab list**: Oxford-3000-by-CEFR (A1+A2 = 1681 words) was British-spelled and too
  small — flagged `mom`, `favorite`, `neighbor`, `excited` as "hard". Diagnostic: scoring
  the A2 anchors themselves gave median **0.20 (floor)**. Root cause was genre + the
  metric being source-relative; on the *real* task (Opus refs scored **with source**) it
  was already ~1.0. Fix: union a **wordfreq top-5000 backstop** → Opus refs score mean
  **1.00**, residual flags are genuinely rare/domain words (`ditch`, `pollen`, `slavic`).

General lesson we acted on: *score your own gold/anchor set with each component before
training; if the anchors look "bad", the metric and your notion of the target disagree —
find out before training, not after.*

## 4. Validation gate (§5, offline, before training)

`scripts/validate_reward.py` scores stored eval generations for SFT, GRPO-v7, GRPO-v9 on
50 held-out paragraphs and asserts known-correct orderings. **All 4 hard checks pass** with
the structured judge (0/120 failures):
1. Melicope: reward(GRPO) > reward(SFT)  ✅
2. Tomorrow Never Dies: reward(GRPO) > reward(SFT)  ✅
3. Ali Darassa: reward(v9) < reward(v7)  ✅  (v9 invented 2 countries, v7 invented 1)
4. No brevity bias: Spearman(reward, word_count) = −0.23 (> −0.4)  ✅

Mean reward separates models: **SFT 0.061 · v7 0.266 · v9 0.253**.

## 5. Training runs

- **20-iter smoke, both arms** (Arm A = full; Arm B = no judge): pipeline works
  end-to-end; reward variance alive (~80% of groups have within-group fidelity std > 0.05).
- **100-iter Arm A** (full reward). Within-training trajectory by run-fifths:

  | metric | 1st | 2nd | 3rd | 4th | 5th |
  |---|---|---|---|---|---|
  | halluc-flag rate | 0.74 | 0.70 | 0.65 | 0.54 | **0.80** |
  | fidelity_mean | 0.29 | 0.31 | 0.36 | 0.44 | **0.24** |
  | reward_mean | 0.10 | 0.13 | 0.16 | 0.20 | **0.11** |
  | level_band_mean | 0.41 | 0.49 | 0.53 | 0.51 | 0.51 |

  Improvement through ~80%, then a regression in the final fifth. **Confound**: batch_size=1,
  no shuffle → "fifths" partly reflect *which paragraphs* were seen, not pure learning.
  Judge: 0/918 parse failures.

## 6. Headline result — held-out greedy eval (off-policy, prompt-order confound removed)

Greedy decoding, GRPO-v10_full vs SFT, **same 50 held-out paragraphs**, scored with the
v10 reward (SFT scored identically):

| metric | SFT | GRPO-v10 | change |
|---|---|---|---|
| **hallucination-flag rate** | 0.92 | **0.42** | −54% |
| **unsupported claims / output** | 1.58 | **0.68** | −57% |
| **fact recall** | 0.78 | **0.85** | +9% |
| fidelity term | 0.11 | 0.53 | ~5× |
| **level_band (readability)** | 0.484 | 0.474 | flat |
| reward (total) | 0.061 | 0.250 | ~4× |

GRPO roughly **halved the hallucination rate and unsupported-claim count while raising
recall** — a genuine faithfulness gain that holds off-policy on held-out data.

**Judge-swap confirms the faithfulness gain is real (not Haiku-gaming).** Re-scoring the
same outputs with a disjoint judge family (deepseek-v4-pro, identical prompt) reproduces the
gap: halluc-flag 0.91→0.56, unsupported 2.34→1.18 (halved), recall 0.69→0.75. Caveat: **both
judges flag SFT at ~0.90**, which is suspiciously high (the brief described SFT as having
*some* hallucinations) — the *relative* gap is robust across families, but the *absolute*
rates are likely inflated by both judges counting paraphrase/gloss drift as "unsupported".
A ~20-claim blind human audit (still TODO) is needed to calibrate the absolute numbers.

## 6b. THE CATCH — A2 difficulty regressed (the headline metric dropped)

Running the actual DeepSeek difficulty classifier (temp 0, **mode-of-3**, **paired** on the
same 50 items, same judge/anchors as the SFT baseline) tells a different story than
`level_band` did:

| | SFT | GRPO-v10 |
|---|---|---|
| **A2 hit-rate** | **72%** (36/50) | **56%** (28/50) |
| levels | A2 36 · A1 8 · B1 6 | A2 28 · **B1 15** · A1 7 |

Flip matrix (baseline → adapter): **A2→B1 = 8**, A2→A1 = 4, A1→A2 = 4, others stable. Net
**−8 A2 (−16 points)**. So GRPO **traded A2-simplicity for faithfulness** — the opposite
corner of the same tension.

**`level_band` held flat (0.48) but A2 dropped 16 pts — the proxy lied.** FRE/MSL cannot
see lexical density / jargon / passive voice, which is exactly what the classifier keys on.
Reading the 8 A2→B1 flips, the mechanism is clear: GRPO **retains source-specific
terminology and precise phrasing that SFT simplified away**, because that raises fidelity.
E.g. *Christina Pagel*: SFT "Her work helps doctors and hospitals use data to make better
decisions" (A2) → GRPO "CORU uses mathematical models and data analysis" (B1). *Blackledge
House*: SFT "People built it around 1750" → GRPO "It is a historic house... documented it"
(passive).

**None of the three guardrails resisted this:** (a) FRE/MSL stayed in band (blind to
density); (b) **`vocab` exempts source lemmas** — so retaining source jargon ("CORU",
"operational research") is *literally free*, which is the exemption we added to avoid
punishing necessary domain terms but which also removed all pressure to gloss/simplify them;
(c) fidelity actively *rewarded* the retention. The multiplicative veto only works if a
component actually penalizes the failure — and A2-density has no penalizing component.

**Bottom line: not shippable as-is** (56% < 72% A2). This is the "rescue, not polish"
branch. The faithfulness win is real but it came at the cost of the primary objective.

## 7. Open questions (where we want guidance)

0. **CENTRAL QUESTION (new, from §6b): how should the reward defend A2-difficulty against
   fidelity pressure?** Candidates: (a) put the difficulty classifier (or a cheap distilled
   proxy) into the reward as a multiplicative term/gate — costs a 2nd judge call/rollout;
   (b) replace/augment FRE/MSL with a feature that tracks the classifier (lexical-frequency
   density, passive-voice count, subordination depth); (c) **rethink the vocab source-lemma
   exemption** — it currently gives retained source jargon a free pass; maybe exempt source
   lemmas from the *hallucination* view but still reward *glossing* them for the *vocab*
   view; (d) re-balance so a difficulty term can veto. Which is most robust, and how do we
   avoid simply swinging back to over-simplification (the A2→A1 flips show that corner too)?

1. **Is the multiplicative + `0.1^min(n,2)` shape right?** It crushes most totals into
   0.001–0.1, and when a *whole* G=8 group is bad (all n≥2 → all ≈0.01) the group std → 0
   and the advantage is dead for that group (~20–40% of groups). GRPO normalizes per-group,
   so relative spread is what matters — but does the compressed magnitude / dead-group rate
   hurt advantage estimation? Would a **smoother penalty** (e.g. `exp(-α·n)`, or log-reward,
   or additive-in-log-space) preserve gradient better without losing the "hallucination is
   the worst defect" asymmetry?

2. **Component balance / drowning.** Fidelity dominates the total (range ~0.002–1.0);
   level_band (0.2–1.0) and vocab (≈1.0 almost always) barely move it. The validation
   harness flagged vocab/gates as "drowned" (IQR ≈ 0). Is it fine for fidelity to dominate
   (it *is* the target gap), or should components be rebalanced / rescaled (`x^k`) so band
   and vocab still shape behavior?

3. **Circularity — PARTIALLY ANSWERED.** Judge-swap (deepseek) reproduced the SFT→GRPO
   faithfulness gap (see §6), so it isn't Haiku-specific gaming. STILL OPEN: both judges flag
   SFT at ~0.90, so absolute rates are probably inflated (gloss/paraphrase counted as
   invention) — a ~20-claim blind human audit is the remaining check, and would also tell us
   whether the GRPO 0.42–0.56 number means what we think.

4. **A2 metric — ANSWERED (see §6b): it dropped 72% → 56%.** `level_band` flat was a false
   reassurance. This reframes the whole effort around question 0 above.

   Note on Q1 (already actioned): we replaced `0.1^min(n,2)` with `exp(-1.2·n)` (uncapped,
   smooth) to remove the dead-group ties on the hardest prompts. Not yet trained with it.

5. **GRPO hyperparameters / stability.** G=8, lr=1e-6, 100 iters, GSPO sequence-level
   importance sampling, temp=1.0, no shuffle, no explicit KL/entropy term mentioned. The
   late-fifth regression + unshuffled data: should we shuffle, add KL-to-SFT regularization,
   lower temp, change G or lr, and how long to actually train (50? 200? 500 iters)?

6. **Trainer call pattern.** `mlx-lm-lora` calls the reward function ~2–3× per optimizer step
   on identical completions (the cache absorbs the repeats). Anything to worry about there
   (e.g. is it re-scoring for a reason we're missing)?

## 8. Repo pointers

- Reward: `reward/{compose,level_band,vocab,fidelity,gates,nlp}.py`; trainer adapters
  `langsimp/training/rewards_v10.py` (`v10_full_reward` / `v10_nojudge_reward`).
- Config/data: `config/{band.json,reward.yaml}`, `data/vocab_1500.txt`, `data/anchors/`,
  `claro/prompts.yaml` (`fidelity_judge`).
- Scripts: `build_vocab_list.py`, `calibrate_band.py`, `validate_reward.py`, `probe_v10.py`,
  `train_grpo_v10.sh`.
- Logs: `runs/<arm>/{metrics,rollouts}.jsonl`. Adapter: `adapters/grpo_v10_full`.
- Tests: `tests/test_reward.py`, `tests/test_verifier.py` (348 passing).
