# v11.1 status + a finding on the §4.1 hard gate (for the reviewer)

v11.1 is implemented through CHANGE 1–3. The §4.1 "real-flip separation" gate **fails at
4/8**, but the breakdown shows it is *mis-specified* (it conflates density with
faithfulness), while the density mechanism it was meant to verify **works**. One design
call needed before I either revise the gate or proceed.

## What's implemented (all tested, full suite 368 green)

- **CHANGE 1 — syntactic band in `level_band`.** Added PASS (passive constructions/sent via
  `nsubjpass`/`auxpass`) and SUB (subordination/sent). **APPOS dropped** per the CHANGE 1.3
  detector spot-check (fired on 1/20 anchor sentences, that one a false positive). Bands
  recalibrated from the blended A2 anchors. **Calibration changed from IQR (25/75) to
  10/90 percentiles**: with four multiplicative trapezoid factors, an IQR band put only
  ~0.5⁴≈6% of anchors inside all factors, so the gold A2 anchors cratered to 0.09 median
  (failed anti-A1 check #1). 10/90 restores anchors to **0.89 median** while keeping the
  passive band tight ([0, 0.15]) so B1 density (passive ~0.30) stays well outside.
- **CHANGE 2 — vocab (Edit 1) kept, demoted.** No code change; now a rare-term guard.
- **CHANGE 3 — Edits 2+3.** Tiered core/peripheral recall (recall over CORE only) + gloss-
  is-supported. Verified live: on the Pagel source, a simplified candidate gets the two
  core facts present (recall 1.0) while CORU/methods/dates are tiered *peripheral* and
  dropping them is free, n_unsupported 0. This removes the packing pressure at its root.

## The density mechanism works (faithfulness-controlled)

`level_band` alone (faithfulness-blind) separates the 8 real flips **7–8/8** (SFT-A2 scores
higher than GRPO-B1). And a controlled hand pair — same source, **same facts preserved**,
differing only in syntax:

| candidate | total | level_band | fidelity | passive/s | sub/s |
|---|---|---|---|---|---|
| PLAIN (A2) | **1.000** | 1.00 | 1.00 | 0.0 | 0.0 |
| PACKED (B1) | **0.006** | 0.04 | 0.30 | 4.0 | 2.0 |

When fidelity is held equal, the reward crushes the packed version 1.000 → 0.006. The
target is now coherent: **faithful-and-simple is the unique maximum.**

## Why §4.1 fails anyway (4/8) — the confound

§4.1 asks: full reward scores `sft_text (A2) > grpo_text (B1)` on ≥6/8. Full-reward results:

| flip | SFT total | GRPO total | who wins | why |
|---|---|---|---|---|
| Blackledge | 0.301 | 0.134 | SFT ✓ | both faithful, SFT simpler |
| John Merritt | 0.301 | 0.058 | SFT ✓ | both faithful, SFT simpler |
| Thelma Hunt | 1.000 | 0.012 | SFT ✓ | both faithful, SFT simpler |
| Pseudomonas | 0.060 | 0.029 | SFT ✓ | similar fidelity, SFT simpler |
| Christina Pagel | 0.0006 | 0.100 | GRPO | **SFT hallucinated badly** (fid 0.006) |
| Hildegarde Howard | 0.003 | 0.012 | GRPO | **SFT fabricated 5 facts** (fid 0.027) |
| Peter Jones | 0.006 | 0.010 | GRPO | SFT partly hallucinated (fid 0.30) |
| Cayan Tower | 0.200 | 0.200 | tie | both faithful, similar density |

In all 4 GRPO-wins, **SFT badly hallucinated**, so the faithful-but-dense GRPO text
*correctly* outscores the simple-but-fabricated SFT text. The spec forbids weakening the
hallucination penalty (`halluc_term = exp(-1.2·n)`), so **no band setting can make a
hallucinated-A2 beat a faithful-B1** — and arguably it *shouldn't*. Every per-pair outcome
above is defensible on its merits.

So §4.1 measures the wrong thing: the real flips are not a clean density axis — half of
them have SFT in the *other* failure corner (fabrication). "sft > grpo" is unsatisfiable
without violating the strict-hallucination rule this spec deliberately keeps.

## The design question

1. **Do you accept "faithful-B1 > hallucinated-A2"?** I think yes — it matches the v10
   lesson (a hallucinated A2 is worse than a faithful B1) and your own "keep the
   unsupported-claim penalty strict" instruction. If so, §4.1 as written is the wrong gate.
2. **How should the offline density gate be respecified?** Options:
   - Replace §4.1 with the **faithfulness-controlled** test (the PLAIN-vs-PACKED shape
     above, generalized): for each flip, hold the GRPO text's *content* and compare a
     packed vs. a plainer rendering — but that needs hand-written simpler versions.
   - Gate on **`level_band` separating the flips ≥6/8** (faithfulness-blind — passes 7–8/8)
     plus the existing 4.2 single-clause guard. Cheap, already passes.
   - Restrict §4.1 to flips where `|fid_sft − fid_grpo|` is small (the clean density
     subset), where SFT-wins 4/4.
3. **Strategic check:** the reward now makes faithful-A2 the unique max, but in training the
   model will still prefer faithful-B1 over hallucinated-A2 when it can't reach faithful-A2.
   Is that the behavior you want (I think yes — it never rewards fabrication), or do you
   want density to dominate fidelity past some point?

My lean: accept (1); respec §4.1 to the `level_band`-separation + 4.2 guard form (option 2b);
keep everything else; proceed to training with the §5 syntactic readouts so we can watch
passive/subordination fall. But this is your gate, so I paused.

## Pre-launch check 2 (anti-A1) — does not hold, and the reason matters

You asked me to confirm the 10/90 widening didn't let A1 text sneak into band. It did —
but the deeper finding is that **the syntactic band cannot separate A1 from A2 at any
width**, because our A1-labeled outputs are not syntactically simpler than A2:

| | words | sents | FRE | MSL | passive | subord |
|---|---|---|---|---|---|---|
| A1-labeled (SFT) | 76 | 8 | **69.8** | 9.3 | 0.06 | 0.00 |
| A2 (Opus wiki refs) | 98 | 10 | **79.1** | 9.6 | 0.00 | 0.11 |

The A1 outputs have *lower* FRE (harder, not easier), similar MSL, and near-zero
passive/subordination — same corner as A2. A width sweep confirms anchors and A1 track each
other at every setting (e.g. p=15/ff=2.0: anchors 0.87, A1 0.90; p=25/ff=1.0: anchors 0.20,
A1 0.24). The original IQR "A1 = 0.18" never actually passed either — it only looked low
because IQR simultaneously cratered the A2 anchors to 0.09, i.e. **A1 scored above A2 there
too.** These features have no A1 signal.

Implications:
- The band is purely a **B1 defense** (its job — and it works, 7/8 on the respec'd gate). It
  is **A1-neutral, not A1-defending**: it neither penalizes nor rewards A1 vs A2 (flat).
- So it does not *push* the model toward A1 — but it won't *catch* A1 drift if the other
  v11.1 changes create one. The candidate for that pressure is Edit 2 (tiered recall now
  permits dropping peripheral facts); the brake is that core-recall still penalizes dropping
  core facts, and the SFT-of-Opus start sits in the **B1** gravity well (the observed failure
  direction is B1, not A1 — A1 drift has never been seen in this project).

Recommendation: proceed with the band as the B1 defense; accept there is no syntactic A1
defense (these features can't provide one); **monitor A1 directly** via the §5 probe
classifier and the §6 paired eval (which already reports `pct_too_easy`). If A1 drift
appears, the fix is a length/content-thinness feature or a recall-side guard, not a band
tweak. Band setting: I'll use **p=15 edges + falloff_frac=2.0** (anchors 0.87, B1 flip
separation 8/8, A1-neutral) unless you prefer the current 10/90.

Open question for you: ship with a B1-only band + A1 monitoring, or add an A1-separating
feature (output length / content density) before training?

## Not yet done

§4 harness wiring (assertions into `validate_reward.py`), §5 training, §6 eval. `level_band`
falloff is currently the v11 default (`falloff_frac=0.5`); if you want a stronger density
penalty I can steepen it, but the controlled test already gives a 1.000→0.006 spread.
