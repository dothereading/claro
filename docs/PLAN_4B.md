# Plan: best 4B v11.1 model, cheaply

## Objective

Make the best 4B faithful-and-simple simplifier. A2 ~50% (rest split A1/B1) is acceptable,
so the optimization target is:

- **Primary: faithfulness** — minimize hallucination-flag rate, hold recall ≥ ~0.97.
- **Guardrail (not maximize): difficulty stays centered** — A2 ≳ 45%, and neither B1 nor A1
  runs away (reject a checkpoint if B1 > ~30% or A1 > ~30%). We don't chase A2; we keep it
  from collapsing while faithfulness improves.
- Headline composite to watch: **"faithful-A2 rate"** = fraction of outputs that are A2 AND
  have 0 unsupported claims (and, secondarily, "faithful & ≤A2" allowing A1).

## Guiding principles (this is where the savings are)

1. **Don't retrain SFT.** The `sft_n{250,500,750,1000,1250,1500}_4b` adapters already exist.
   Picking the start is an *eval* problem, not a *training* problem.
2. **One GRPO run, not a grid.** Pick a single SFT start, train once with checkpoints, pick
   the best checkpoint. (Optional rigor in Phase 1b if you'll spend a little more.)
3. **Cheap proxy for checkpoint selection; expensive classifier only on finalists.** Use the
   free deterministic signals + the Haiku fidelity judge to pick candidate checkpoints; run
   the DeepSeek mode-of-3 A2 classifier only on the 2–3 finalists.
4. **Early-stop on plateau.** "As many iterations as we need" = checkpoint every 50 and stop
   when faithfulness stops improving for 2 checkpoints.
5. Reuse the judge cache; smoke before long runs; lock one eval set so all numbers compare.

## Phase 0 — Lock the eval harness (once, ~$0, 10 min)

- **Eval set:** the **50-item** held-out set (`grpo_v10_full_eval50.json` titles), not the
  noisy 30. More signal, same cost ballpark.
- **Protocol:** DeepSeek classifier mode-of-3 temp-0 (A2 spread) + Haiku fidelity (recall,
  n_unsupported). Both already wired in `eval_difficulty_paired.py` and the fidelity scorer.
- **One command** to produce the composite readout per adapter (small wrapper, reuse
  existing scripts): pct_{A1,A2,B1}, halluc-flag rate, recall, faithful-A2 rate.

## Phase 1 — Pick the SFT start (no training, ~$1–2, 20 min)

The existing `sft_n*_4b` adapters already have stored eval outputs + single-vote levels.

1. **Faithfulness-score the stored SFT outputs** (Haiku, ~180 cached calls — no generation):
   gives halluc rate + recall per `n`.
2. **A2 vs n** from the stored levels (single-vote ok for *relative* comparison; the scaling
   curve `sft_scaling_curve_4b.png` already exists).
3. **Choose the start by the knee, not the peak.** GRPO *adds* faithfulness, so we want the
   SFT that best learned the A2 *format/difficulty* with the most GRPO headroom — typically
   the knee where A2 plateaus (likely n750–n1000), not the highest-n. Avoid the top of the
   ladder if A2 has already saturated (less to gain, more "locked-in" behavior).
4. If A2 is still clearly rising at n1500, note "extend SFT" as a deferred option (costs
   Opus distillation $ — only if Phase 3 disappoints).

**Output:** one chosen `sft_n{X}_4b` start (default guess: **n750_4b**, already validated).

### Phase 1b — OPTIONAL rigor (if you'll spend ~25 extra min)

Instead of trusting the SFT-eval heuristic, **short-probe GRPO** from the top 2 candidates:
40 iters each, eval faithfulness, keep the better responder. Then Phase 2 *resumes* that
checkpoint (via `--resume-adapter`) rather than starting fresh — so the probe isn't wasted.
This directly measures "which start responds best to GRPO" for ~1 extra short run's cost.

## Phase 2 — One checkpointed GRPO run (the main spend, ~1–1.5 h)

From the chosen start, `v11_reward`, **checkpoints every 50 iters, up to 200**:

```
ARM/v11: MODEL=…4b RESUME_ADAPTER=adapters/sft_n{X}_4b/adapters.safetensors \
  ADAPTER_DIR=adapters/grpo_v11_4b_final ITERS=200 GROUP_SIZE=8 bash scripts/train_grpo_v11.sh
```

- Smoke 2 iters first (already validated at 4B; just confirm memory/start).
- Keep `beta=0.1` KL-to-SFT (protects A2 from drift — directly relevant), temp 1.0, G=8.
- Cost control: `MAX_COMPLETION_LENGTH=384` (outputs are short → faster generation, ~25%
  wall-clock saving) and the judge cache makes the ~3×/step duplicate calls free.
- The runner already saves intermediate adapters (`000XXXX_adapters.safetensors`), so we get
  the 50/100/150/200 checkpoints for free.

## Phase 3 — Checkpoint selection (cheap, ~20 min)

1. **Cheap pass on every checkpoint** (50/100/150/200): generate greedy on a **20-item**
   subset, score with the free `level_band` + Haiku fidelity only (no DeepSeek). Plot
   halluc-rate + faithful proxy vs iters. **Stop early** if no improvement for 2 checkpoints.
2. **Finalists only** (the 2–3 best by the cheap pass): run the full 50-item DeepSeek mode-3
   classifier + fidelity → the real composite. Pick the checkpoint that **maximizes
   faithfulness subject to the difficulty guardrail** (A2 ≳ 45%, B1/A1 not runaway).

## Phase 4 — Final validation (~$1, 10 min)

- **Reliable judge-swap** on the winner (the one thing still open): use an unpinned
  `deepseek/deepseek-v4-pro` (drop the dead `:gmicloud/fp8` provider) OR a GPT/Gemini judge
  with structured outputs, to confirm the faithfulness gain isn't Haiku-specific. ~60 calls.
- Record final numbers in `V11_FINDINGS.md`; tag the adapter.

## Budget / stop rules

| phase | training | judge $ | wall-clock |
|---|---|---|---|
| 0 standardize | none | ~0 | 10 min |
| 1 pick SFT start | **none** | ~$1–2 | 20 min |
| 1b (optional) | 2× 40-iter probe | ~$1 | ~25 min |
| 2 GRPO ≤200 iters | 1 run | ~$2–3 | 1–1.5 h |
| 3 checkpoint select | none | ~$1 | 20 min |
| 4 judge-swap | none | ~$1 | 10 min |

**Total (without 1b): ~2–2.5 h wall-clock, ~$5–7 of judge calls, ONE GRPO run.**

Hard stops to avoid overspend:
- Phase 2: stop at the first checkpoint where faithfulness fails to improve over the prior
  two AND the guardrail still holds — don't run 200 if it plateaus at 100.
- If Phase 3's best checkpoint isn't better than the current `grpo_v11_4b` (halluc 0.20),
  **stop** — we're at diminishing returns for this SFT start; revisit the start (Phase 1b)
  or accept the current model rather than burning more iters.

## What we explicitly do NOT do (cost discipline)

- No SFT retraining unless Phase 1 shows A2 still climbing at n1500 *and* Phase 3 disappoints.
- No GRPO grid over SFT starts (Phase 1b's 2 short probes is the most we'd spend on that).
- No full DeepSeek classifier on every checkpoint — cheap proxy first, classifier on finalists.
