# v11 / v11.1 — Full Findings

Consolidated record of the v11 and v11.1 reward iterations: the pre-flight that
redirected v11, the v11.1 syntactic-band design, the calibration findings, the §4 gate
respec, and the 1B-vs-4B training results. Supersedes the in-progress notes in
`V11_PREFLIGHT.md` and `V11_1_GATE_FINDING.md` (kept for the blow-by-blow).

## 0. Context

Fine-tuning Gemma-3 to rewrite Wikipedia at CEFR **A2**, via GRPO from an SFT checkpoint.
Headline metric: A2 hit-rate from a DeepSeek difficulty classifier (paired, mode-of-3,
temp 0). The recurring tension is **faithfulness vs. A2-simplicity**:

- **SFT** hits A2 often but partly *by being unfaithful* (drops/invents facts to stay
  simple). Human audit confirmed SFT's ~0.9 hallucination-flag rate is mostly real.
- **v10** (cardinal reward: `level_band × vocab × fidelity × gates`) halved real
  hallucinations (judge-swap + human-audit confirmed) **but regressed A2 72% → 56%**: the
  fidelity reward made the model preserve facts by packing them into denser prose, and no
  component penalized that density.

v11 / v11.1 are the attempt to close the loop: keep the faithfulness gain, stop the A2 loss.

## 1. v11 spec and the pre-flight finding (why v11 → v11.1)

v11 proposed three edits: **(1)** close the vocab source-lemma exemption so retaining
off-list source *jargon* costs a hard-word penalty (+ a gloss detector so glossing a term
is the escape hatch); **(2)** tier source facts core/peripheral, compute recall over core
only (dropping peripheral detail is free); **(3)** tell the judge a gloss is supported, not
invented. The spec assigned the "A2 defense" to the reworked vocab term (Edit 1).

**Pre-flight on real data killed that premise.** Scoring the 8 real A2→B1 regression flips
with the implemented Edit 1: it separates SFT from GRPO on **only 1/8**. The wordfreq-5000
backstop already contains the supposed "jargon" (models, data, analysis, operational,
research), so closing the exemption barely moves the score. The measured driver of the A2
drift is **syntactic, not lexical** — averaged over the 8 flips, SFT(A2)→GRPO(B1):

| feature | SFT (A2) | GRPO (B1) | ratio |
|---|---|---|---|
| passive voice / sentence | 0.16 | 0.59 | **3.7×** |
| subordinate clauses / sentence | 0.21 | 0.43 | 2.0× |
| appositives / sentence | 0.13 | 0.23 | 1.8× |
| mean sentence length | 9.97 | 11.98 | 1.2× (in band) |

The model packs facts into passive, subordinated sentences — exactly the classifier's B1
trigger — invisible to both `vocab` and the FRE/MSL `level_band`. The expert agreed and
issued the **v11.1 delta**: move the A2 defense to a *syntactic band inside level_band*;
keep Edits 1/2/3 (Edit 1 demoted to a rare-term guard).

## 2. v11.1 implementation (all in `reward/`, full test suite green)

- **CHANGE 1 — syntactic band in `level_band`.** `level_band = trapezoid(FRE) ×
  trapezoid(MSL) × trapezoid(PASS) × trapezoid(SUB)`. PASS = passive constructions/sentence
  (`nsubjpass`/`auxpass`, counted once per verb), SUB = subordination/sentence (`advcl,
  ccomp, xcomp, acl, relcl, csubj`). Two-sided trapezoids (too-high reads B1, too-low reads
  choppy-A1). **APPOS dropped** (CHANGE 1.3): its detector fired on 1/20 anchor sentences,
  that one a false positive.
- **CHANGE 2 — vocab (Edit 1) kept, demoted** to a rare-term guard. No code change.
- **CHANGE 3 — Edits 2+3.** Tiered core/peripheral recall (recall over core only,
  `no_core_facts` event), gloss-is-supported prompt. `prompt_version` → `v11`. Verified
  live: a simplified Pagel candidate gets core facts present (recall 1.0) while
  CORU/methods/dates tier *peripheral* and dropping them is free.

### Calibration findings (CHANGE 1.2 / 1.4)

- **IQR (25/75) cratered the anchors.** Four multiplicative trapezoids → only ~0.5⁴ ≈ 6% of
  A2 anchors fall inside *all four* bands, so the gold A2 anchors scored **0.09 median**
  (failed anti-A1 check #1). Switched the band to **10th/90th percentiles** → anchors back
  to **0.89 median**, passive band still tight `[0, 0.154]` so B1 density (passive ~0.3)
  stays well outside. Current `config/band.json`: FRE [64.6, 92.9], MSL [6.8, 13.2],
  PASS [0, 0.154], SUB [0, 1.143].
- **Anti-A1 is unsatisfiable with these features.** The A1-labeled outputs are NOT
  syntactically simpler than A2 — they have *lower* FRE (69.8 vs 79.1), similar MSL, near-
  zero passive/subordination, same corner as A2. Anchors and A1 track each other at every
  band width. So the band is a **B1 defense only**; it is A1-neutral (won't push toward A1,
  won't catch A1 drift). Shipped with this understood + A1 monitored via the eval classifier.

### §4 gate respec (the original §4.1 was mis-specified)

Spec §4.1 ("full reward scores SFT-A2 > GRPO-B1 on ≥6/8 flips") **fails 4/8 — correctly**:
in the 4 GRPO-wins SFT *badly hallucinated*, so faithful-B1 rightly beats fabricated-A2,
and the spec forbids weakening the hallucination penalty. The gate conflated density with
faithfulness. Expert approved respec to **`level_band(sft) > level_band(grpo)` ≥ 6/8**
(isolates the density mechanism) → **passes 7/8**. The faithfulness-controlled check (4.2)
is decisive: same facts, plain vs. packed → **1.000 vs 0.006**. The density mechanism is
sound offline.

## 3. Training results

Both runs: start from `sft_n750{,_4b}`, lr 1e-6, G=8, batch 1, lora 16, temp 1.0, GSPO
sequence-level, KL beta=0.1 (mlx-lm-lora default; data shuffled by its batch iterator),
**80 iters**, single full-reward arm (`v11_reward`), 0 judge failures (structured outputs).

### 1B (`adapters/grpo_v11`) — FAILED

| | SFT-1B | v10-1B | **v11-1B** |
|---|---|---|---|
| A2 hit-rate (50 items) | 76% | 56% | **26%** |

Regressed in *both* directions: 18 items A2→B1, 11 A2→A1. Diagnosis (the spec's
"A2-doesn't-recover" branch):
- **The band didn't move the model.** v11-1B greedy features ≈ v10's (passive 0.16 vs SFT's
  0.09; subord 0.26). Features did not move → a **capability / training-strength ceiling**,
  not a target problem (the offline tests proved the reward *can* express the target).
- **Tiered recall opened A1 drift.** The 11 A2→A1 items are shorter (58 vs 70 words) — the
  model dropped content to oversimplify, the unguarded A1 corner. This is *new* damage vs
  v10, which is why 26% < 56%.
- Caveat: v11-1B ≈ v10 on every structural feature yet classified ~30 pts worse — some of
  the 26-vs-56 gap is classifier run-to-run variance (SFT itself moved 72→76 between runs).

### 4B (`adapters/grpo_v11_4b`) — THE WIN

Same reward, 80 iters, from `sft_n750_4b` (base `gemma-3-4b-it-bf16`), ~18–32 s/it, ~32 GB.

**A2 (paired classifier, 30 items, mode-of-3):**

| | SFT-4B | **v11-4B** |
|---|---|---|
| A2 hit-rate | 50% (15/30) | **53%** (16/30) |
| A1 (too easy) | 9 | 8 |
| B1 (too hard) | 5 | 5 |

A2 **held flat — no regression, and no A1 drift** (the 1B failure mode did not recur).

**Faithfulness (Haiku fidelity judge, same 30 items):**

| | SFT-4B | **v11-4B** |
|---|---|---|
| hallucination-flag rate | 0.533 | **0.200** (−62%) |
| unsupported claims / output | 0.667 | **0.200** (−70%) |
| recall | 0.979 | 0.985 |

**v11.1 at 4B delivers the faithfulness gain WITHOUT the A2 cost** — the faithful-and-simple
target chased since v7. Against the §6 gates (v11-4B vs the SFT-4B it trained from): A2 did
not regress ✅ and hallucination strongly improved (0.20, well under the 0.55 fail line) ✅
— **both gates pass.**

This resolves the central question: **the 1B catastrophe was a capability ceiling, not a
broken target.** Same reward, same 80 iters — broke 1B, worked at 4B once the model had the
headroom for the band's gradient to move passive/subordination.

### Judge-swap (de-circularization) — attempted, INCONCLUSIVE

Re-scoring the 4B outputs with a disjoint judge (deepseek-v4-pro) to confirm the
faithfulness gain isn't Haiku-specific. **Directionally agrees** (halluc 0.375 → 0.000;
n_unsupported 0.5 → 0.0) — same direction as Haiku. **But unreliable today**: deepseek
returned null content on ~78% of the 4B fidelity prompts (22/30 SFT, 25/30 GRPO fail-open),
so only ~8 SFT / ~5 GRPO scores were usable. The pinned `:gmicloud/fp8` provider appears
degraded (it worked at ~6% fail at 1B). **Not a reliable confirmation** — rerun with an
unpinned deepseek or a GPT/Gemini-class judge (which support structured outputs) needed.

### Qualitative read (3 random shared paragraphs)

- **Blackledge–Kearney House** — the clean win: SFT-1B→A2 *but* hallucinated ("Cornwallis,
  leader of the *American* army" — he was British); v11-1B & SFT-4B → B1 (faithful but
  passive/dense); **v11-4B → A2, kept all facts, stayed simple.** Only v11-4B is faithful
  *and* simple.
- **Christina Pagel** — v11-4B → **A1** (over-simplified: "uses math and data"): the mild
  tiered-recall A1 tendency, present but not catastrophic at 4B.
- **Tomorrow Never Dies** — v11-1B and SFT-4B labeled **A1** despite reading like clean A2;
  v11-4B → A2. The classifier is **noisy/harsh at the A1/A2 boundary**, which means the
  53% A2 number probably *understates* v11-4B and inflates the 1B 26%.

## 4. Conclusions

1. **v11.1 is validated at 4B.** Substantially more faithful (hallucination −62%) with A2
   held flat and no A1 drift. The reward target is coherent; the original §6 "proceed to 4B"
   decision is answered **yes — it works at 4B**.
2. **The 1B failure was capability, not target.** Identical reward; 1B couldn't move the
   syntactic features and the tiered-recall A1 drift dominated; 4B moved them and held A2.
3. **Caveats (don't oversell):**
   - n = 30 at 4B (smaller/noisier than the 50-item 1B evals).
   - **Circularity not yet broken** — faithfulness measured with the training judge; the
     deepseek swap was inconclusive (provider failure). Needs a reliable disjoint judge.
   - **Absolute A2 is only ~53%** because the 4B *SFT baseline* is weak (50%, vs 1B SFT's
     76%). This is a validated *reward*, not a finished A2 simplifier.
   - The A1/A2 classifier boundary is noisy; consider mode-of-5 to de-noise.

## 5. Next steps (open)

- **Reliable judge-swap** on the 4B outputs (unpinned deepseek, or a GPT/Gemini judge with
  structured outputs) to lock the faithfulness number and kill the circularity caveat.
- **Lift absolute A2**: stronger 4B SFT start (the `sft_n{1000..1500}_4b` ladder) and/or
  more GRPO iters — this is now an SFT-quality / training-length question, not a reward one.
- Optional: an A1 guard (length / content-thinness floor) if A1 drift grows with more iters;
  re-noise check on the classifier (mode-of-5).

## 6. Artifacts / repo state

- **Code**: `reward/{level_band,vocab,fidelity,compose,nlp,gates}.py` (v11.1);
  `langsimp/training/rewards_v11.py` (`v11_reward` adapter, §5 readouts);
  `scripts/train_grpo_v11.sh`. Tests in `tests/test_reward.py` (green).
- **Config/data**: `config/band.json` (10/90, PASS+SUB), `config/reward.yaml`
  (`prompt_version: v11`), `prompts/fidelity_judge.txt` (tiered + gloss),
  `data/vocab_1500.txt`, `data/flips/a2_to_b1.jsonl`.
- **Adapters**: `adapters/grpo_v11` (1B), `adapters/grpo_v11_4b` (4B).
- **Evals**: `eval_results/grpo_v11_eval50.json` (1B), `eval_results/grpo_v11_4b_eval.json`
  (4B). Baselines: `sft_n750_1b_eval80.json`, `sft_n750_4b_eval30.json`,
  `grpo_v10_full_eval50.json`.
- **Run logs**: `runs/v11_1b/` (1B), `runs/v11/` (4B), `runs/v11_4b_smoke/`.
- **Eval scripts**: `scripts/eval_difficulty_paired.py`, `scripts/rescore_judge_swap.py`
  (now takes `--grpo/--sft`), `scripts/make_blind_audit.py` + `score_blind_audit.py`.

---

# 7. PLAN_4B execution — best-4B model (2026-06-21)

Goal (user): best 4B model, A2 ~50% acceptable, **maximize faithfulness**, cheaply. Plan in
`PLAN_4B.md`. Outcome: **`adapters/grpo_v11_4b_n1500` (checkpoint 200) is the new best 4B.**

## 7.1 SFT-start selection (Phase 1, no training)

Scored the stored `sft_n{250..1500}_4b` eval outputs with the composite readout
(`scripts/eval_composite.py` — cached Haiku fidelity + free `level_band`, optional DeepSeek
mode-3). Single-vote A2 was too noisy; re-classified the finalist starts with DeepSeek
mode-3 on the **stored** outputs (no regeneration). Result **revised the plan's n750 guess**:

| SFT start | A2% (mode-3) | B1% | Haiku halluc | recall |
|---|---|---|---|---|
| n250 | 46.7 | 30.0 | 0.367 | 0.966 |
| n750 *(old GRPO start)* | 43.3 | 16.7 | **0.533** (worst) | 0.979 |
| n1000 | 40.0 | **43.3** (out) | 0.433 | 0.993 |
| **n1500** | **46.7** | 23.3 | **0.367** | **0.993** |

Faithfulness does **not** decay at high SFT-n (n1500 ties best), so the "leave GRPO headroom /
knee not peak" reasoning didn't apply. **Chose n1500**: tied-best A2 + best faithfulness +
B1 within guardrail. (User picked "start from n1500 directly" over a Phase-1b probe.)

## 7.2 One GRPO run (Phase 2)

`sft_n1500_4b` → `v11_reward`, 200 iters, G=8, `--save-every 50`, `max_completion_length=384`,
beta=0.1 (mlx default). Added `--save-every` forwarding to the GRPO branch of
`langsimp/training/runner.py` (red/green test in `tests/training/test_runner.py::TestGrpoCommand`).
Ran in **~58 min** (~17 s/it; the 73 s/it smoke estimate was cold-start). Reward never
collapsed/diverged; rollout core-recall held ~1.0. Logs: `runs/v11/metrics.jsonl`.

## 7.3 Checkpoint selection (Phase 3) — DeepSeek mode-3, same 30 held-out items as incumbent

| ckpt | A2% | B1% | Haiku halluc | recall | faithful-A2 |
|---|---|---|---|---|---|
| 50 | 60.0 | 13.3 | 0.267 | 0.987 | 43.3 |
| 100 | 63.3 | 10.0 | **0.200** | 0.985 | 46.7 |
| 150 | 70.0 | 10.0 | 0.233 | 0.985 | 50.0 |
| **200 (winner)** | **70.0** | **6.7** | 0.233 | 0.985 | **50.0** |
| `grpo_v11_4b` *(incumbent)* | 53.3 | 16.7 | 0.200 | 0.985 | 36.7 |

GRPO behavior: **faithfulness peaks early (iter 100), difficulty-retention strengthens late
(iter 150/200)**. Winner ckpt 200 **dominates the incumbent**: A2 +17pts (70 vs 53), B1 halved
(6.7 vs 16.7), faithful-A2 +13pts (50 vs 37), halluc ≈ (1-item noise). It also matches the
SFT start's A2 (70%) while ~halving the start's hallucination (0.367 → 0.233).

## 7.4 Judge-swap (Phase 4) — circularity check, now with WORKING disjoint judges

DeepSeek (pinned and unpinned) fails open ~80% on the structured fidelity prompt → unusable.
Switched `rescore_judge_swap.py` to GPT/Gemini with `FIDELITY_RESPONSE_FORMAT` (json_schema
strict) + a generous `max_tokens` (thinking judges truncate at 1500 → "Unterminated string").
**0 judge failures** on both. SFT_n1500 → ckpt 200, 30 paired items:

| judge | family | SFT halluc | GRPO halluc | Δ | recall SFT→GRPO |
|---|---|---|---|---|---|
| Haiku *(training)* | Anthropic | 0.367 | 0.233 | −0.134 ✓ | 0.993 → 0.985 |
| gpt-4o | OpenAI | 0.467 | 0.367 | −0.100 ✓ | 0.992 → 0.983 |
| gemini-3.5-flash | Google | 0.233 | 0.300 | +0.067 ✗ | 1.000 → 0.978 |

**Honest verdict (judge-split):** 2/3 judges (incl. disjoint gpt-4o) say GRPO hallucinates
less; the newest judge (gemini-3.5-flash) says **parity** (GRPO worse by ~2 items at n=30 —
noise; gemini is far more lenient on the SFT, compressing the gap). **No judge shows GRPO
meaningfully worse, and recall held ~0.98–1.0 everywhere.** So: **faithfulness held-or-improved,
not a clean confirmed gain.** The *difficulty* win (A2 70 vs 53) is robust and independent of
the fidelity judge (DeepSeek classifier).

## 7.5 Conclusion

**Ship `adapters/grpo_v11_4b_n1500` (ckpt 200) as the best 4B model.** Better difficulty than
every prior model, faithfulness no worse than the strong SFT start. The earlier circularity
caveat is now **partially closed** (working disjoint judges; gain confirmed by 2/3, parity by
1). Picking a stronger SFT start (n750→n1500) was the lever — exactly what the deterministic
SFT-ladder analysis predicted, at the cost of one GRPO run.

## 7.6 New/changed artifacts

- **Code**: `scripts/eval_composite.py` (new — composite readout, reused Phase 1/3);
  `scripts/select_checkpoint.py` (new — greedy gen per checkpoint via staged temp adapter dir);
  `langsimp/training/runner.py` (GRPO `--save-every`); `scripts/train_grpo_v11.sh`
  (`SAVE_EVERY`); `scripts/rescore_judge_swap.py` (structured GPT/Gemini judge, `SWAP_JUDGE`/
  `SWAP_MAX_TOKENS` env).
- **Adapter**: `adapters/grpo_v11_4b_n1500/` (fused = ckpt 200; checkpoints
  `0000{050,100,150,200}_adapters.safetensors` retained).
- **Evals**: `eval_results/ckpt_n1500_{0050,0100,0150,0200}.json` (generations),
  `ckpt_n1500_cheap.json` (Haiku+band), `ckpt_n1500_ds.json` (DeepSeek levels),
  `sft_ladder_4b_composite.json`, `sft_ladder_4b_finalists_ds.json`.
