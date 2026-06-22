# v11 pre-flight finding — Edit 1 misses the real A2-drift mechanism

**For the reviewer who wrote the v11 spec.** Before building the §4 fixtures and
spending the 200-iter run + 4B decision, I ran the spec's own discipline ("test the
target on real data first — the lesson of v10 was that an untested target trains
confidently in the wrong direction"). Edit 1 is implemented and unit-tested. On the
**real** v10 outputs it does not do what the spec assumes. Details below; one design
question at the end.

## What the spec assumes

Edit 1 (vocab): close the source-lemma exemption so retaining **off-allow-list source
jargon** ("CORU uses operational research") becomes a hard-word penalty; gloss-or-drop
becomes reward-maximizing. The spec explicitly assigns the A2 defense to this reworked
vocab term and keeps `level_band` (FRE/MSL) as a mere hack-detector: *"the defense lives
in the reworked vocab term (Edit 1)."*

## What the data shows

Scoring the 8 real A2→B1 regression flips (GRPO-v10 vs SFT, the paragraphs that left A2)
with the implemented v11 vocab term:

- **GRPO vocab < SFT vocab on only 1 of 8 flips.** On the other 7 the two are identical.
- Reason: the allow-list is Oxford-A1/A2 **∪ wordfreq top-5000** (kept, per spec Edit
  1.1). That backstop already contains the "jargon" — `models, data, analysis,
  operational, research, mathematical` are all on-list. Genuinely off-list terms
  (`paleornithology`, `ornithologist`) do appear, but usually ≤1 per sentence (under the
  `hard ≤ 1` threshold) or are present in the SFT output too.
- Net: the vocab term barely moves between the A2 and B1 versions. Example — *Christina
  Pagel*, the spec's own motivating case: "CORU uses **mathematical models and data
  analysis**" scores **1.000** (only "mathematical" is off-list → 1 hard → no penalty).

## What the B1-drift actually is — syntax, not vocab

Averaged over the same 8 flips, SFT (A2) → GRPO (B1):

| feature | SFT (A2) | GRPO (B1) | ratio |
|---|---|---|---|
| mean sentence length (words) | 9.97 | 11.98 | 1.2× |
| subordinate clauses / sentence | 0.21 | 0.43 | 2.0× |
| **passive voice / sentence** | 0.16 | 0.59 | **3.7×** |
| appositives / sentence | 0.13 | 0.23 | 1.8× |

The model preserved facts by **packing them into longer sentences with passive voice and
subordinate clauses** ("She *was elected to*…", "eagles *that were* flightless animals
*from the Pleistocene*"). This is exactly the difficulty classifier's stated rubric
("MULTIPLE productive passive-voice constructions in the same passage → B1"; "productive,
non-fixed complex grammar"). It is invisible to **both** `vocab` (Edit 1) **and**
`level_band` (MSL 10→12 stays inside the calibrated band).

## Implication

- Edit 1 will not, on its own, reverse the A2 regression — it targets a lexical mechanism
  that accounts for ~1/8 of the real cases. The hand-written §4 assertions (5–6, off-list
  jargon pairs) would **pass** while the real data is unaffected: the same trap as v10,
  one level up.
- The spec's premise that "the A2 defense lives in the vocab term" looks wrong: the drift
  is **syntactic**, and no v11 component penalizes syntactic density.
- Edit 2 (tiered recall) is the genuinely well-aimed change — letting the model drop
  *peripheral* facts removes the packing **pressure** at its root, so it *might* carry the
  A2 recovery indirectly. But that's a hope, not a mechanism we can assert offline.

## The design question for you

Three options; which do you want, and any traps?

1. **Add a deterministic syntactic-density term** (free, no judge): penalize passive-voice
   count + subordination depth + per-sentence word/fact density, calibrated against the A2
   anchors the same way `level_band` is. This directly targets the classifier's rubric and
   becomes the real A2 defense. Keep Edits 1/2/3. Risk: over-penalizing passives that are
   natural/needed; interaction with fidelity (some facts genuinely need a relative clause).
2. **Rely on Edit 2 alone** to remove packing pressure; keep Edit 1 as a cheap rare-term
   guard; let §6 adjudicate. Cheapest, spec-faithful; risk is a wasted run.
3. **Replace/augment `level_band`** with the syntactic signal rather than adding a new
   term (the spec froze `level_band`, but it's blind to exactly this).

My lean is (1): it's deterministic, judge-free (honors v11's "no new judge calls"), and
it's the only option that encodes the *measured* mechanism rather than hoping a different
edit covers it. But it's a spec deviation, so I paused for your call.

## Status

- Edit 1 (vocab exemption + gloss matcher in `reward/nlp.py`): **implemented + tested**
  (incl. appositive/copula gloss, gloss-with-hard-definition rejection, first-occurrence
  licensing). Harmless to keep.
- Edits 2 (tiered recall) and 3 (gloss-is-supported prompt): **not yet built** — paused
  here.
- §4 fixtures / harness, §5 training, §6 eval: not started.
