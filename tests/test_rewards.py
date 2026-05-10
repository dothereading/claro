"""Tests for the GRPO reward components.

Pure-Python rewards (length, vocab, repetition) are exercised end-to-end.
The judge-based meaning reward uses a stub. Stubs (repetition, difficulty)
are required to exist and return floats so the wiring works, but assert
nothing about their numeric value yet.
"""
from __future__ import annotations

import pytest

from verifier import BaseJudge

rewards = pytest.importorskip("rewards", reason="rewards.py not implemented yet (RED)")


class StubJudge(BaseJudge):
    """Returns whatever evaluate_response was passed at construction time."""
    def __init__(self, response: dict):
        self.response = response
        self.calls: list[str] = []

    def evaluate(self, prompt: str) -> dict:
        self.calls.append(prompt)
        return self.response


# ---------- LengthVsSourceReward ----------

class TestLengthVsSourceReward:
    def setup_method(self):
        self.r = rewards.LengthVsSourceReward()

    def _ctx(self, source: str):
        return rewards.RewardContext(source=source)

    def test_full_reward_at_ratio_one(self):
        src = " ".join(["w"] * 100)
        out = " ".join(["x"] * 100)
        assert self.r.compute(out, self._ctx(src)) == pytest.approx(1.0)

    def test_full_reward_inside_target_range(self):
        # default range [0.8, 1.3] — full reward across that band
        src = " ".join(["w"] * 100)
        for n in [80, 100, 120, 130]:
            out = " ".join(["x"] * n)
            assert self.r.compute(out, self._ctx(src)) == pytest.approx(1.0), f"n={n}"

    def test_decay_below_soft_floor(self):
        src = " ".join(["w"] * 100)
        # ratio 0.5 — output much shorter than source; below 0.8 floor → decay
        out = " ".join(["x"] * 50)
        s = self.r.compute(out, self._ctx(src))
        assert 0.0 < s < 1.0

    def test_decay_above_soft_ceiling(self):
        src = " ".join(["w"] * 100)
        # ratio 1.6 — output much longer; above 1.3 ceiling → decay
        out = " ".join(["x"] * 160)
        s = self.r.compute(out, self._ctx(src))
        assert 0.0 < s < 1.0

    def test_zero_at_extreme_ratios(self):
        src = " ".join(["w"] * 100)
        # extremely short and extremely long both clamp to 0
        assert self.r.compute(" ".join(["x"] * 5), self._ctx(src)) == pytest.approx(0.0)
        assert self.r.compute(" ".join(["x"] * 400), self._ctx(src)) == pytest.approx(0.0)

    def test_empty_source_returns_zero(self):
        assert self.r.compute("anything", self._ctx("")) == 0.0


# ---------- VocabSimplicityReward ----------

class TestVocabSimplicityReward:
    def setup_method(self):
        self.r = rewards.VocabSimplicityReward()

    def _ctx(self, source: str = "irrelevant"):
        return rewards.RewardContext(source=source)

    def test_full_reward_for_all_common_words(self):
        # All top-2000 words → no penalty
        text = "The cat sat on the mat. The dog ran fast."
        assert self.r.compute(text, self._ctx()) == pytest.approx(1.0)

    def test_penalty_when_many_uncommon_words_in_one_sentence(self):
        # Multiple uncommon words in one sentence → penalty
        text = "The mausoleum housed the sarcophagus and ornate friezes."
        s = self.r.compute(text, self._ctx())
        assert s < 1.0

    def test_proper_nouns_ignored(self):
        # Proper nouns are mid-sentence capitalized words; not counted as uncommon.
        # "Albert Einstein worked in Switzerland" — Einstein/Switzerland are proper nouns.
        text = "Albert Einstein lived in Switzerland for many years."
        # Score should be roughly the same as the same sentence without the proper nouns
        # (i.e. very high, since the rest is common).
        assert self.r.compute(text, self._ctx()) == pytest.approx(1.0, abs=0.1)

    def test_sentence_initial_caps_not_treated_as_proper_noun(self):
        # Sentence-initial "Mausoleum" should count as uncommon — it's a regular
        # word that happens to be capitalized, not a name. With 3+ uncommon
        # words both versions should drop below 1.0 by the same amount.
        text1 = "Mausoleum sarcophagus friezes obelisk are old."
        text2 = "The mausoleum sarcophagus friezes obelisk are old."
        s1 = self.r.compute(text1, self._ctx())
        s2 = self.r.compute(text2, self._ctx())
        # Both should drop below 1.0 (lots of uncommon words past the threshold)
        assert s1 < 1.0 and s2 < 1.0
        # And both should drop by approximately the same amount, because the
        # only difference is "Mausoleum" vs "The mausoleum" — the proper-noun
        # heuristic must NOT skip the sentence-initial capital.
        assert abs(s1 - s2) < 0.1

    def test_ratio_in_unit_interval(self):
        # All-uncommon nightmare text — even this should clamp to [0, 1]
        text = "Mausoleum sarcophagus friezes obelisks ramparts kremlin."
        s = self.r.compute(text, self._ctx())
        assert 0.0 <= s <= 1.0

    def test_empty_text(self):
        # No sentences → vacuously full reward (nothing to penalize).
        assert self.r.compute("", self._ctx()) == 1.0


# ---------- SemanticPreservationReward ----------

class TestSemanticPreservationReward:
    def _ctx(self, source: str):
        return rewards.RewardContext(source=source)

    def _r(self):
        return rewards.SemanticPreservationReward()

    def test_full_reward_when_judge_says_perfect(self):
        judge = StubJudge({"facts_preserved": 5, "no_hallucinations": 5})
        s = self._r().compute("output", self._ctx("source"), judge=judge)
        assert s == pytest.approx(1.0)

    def test_zero_reward_when_judge_says_terrible(self):
        judge = StubJudge({"facts_preserved": 1, "no_hallucinations": 1})
        s = self._r().compute("output", self._ctx("source"), judge=judge)
        # 1/5 average → (1/5 + 1/5) / 2 = 0.2; or whichever shape we pick
        assert 0.0 <= s <= 0.3

    def test_hallucinations_hurt_more_than_omissions(self):
        # Same facts_preserved, different no_hallucinations
        omitted = StubJudge({"facts_preserved": 3, "no_hallucinations": 5})
        hallucinated = StubJudge({"facts_preserved": 5, "no_hallucinations": 3})
        # We want the reward to weight hallucination at least as much as omission.
        s_omit = self._r().compute("o", self._ctx("s"), judge=omitted)
        s_hall = self._r().compute("o", self._ctx("s"), judge=hallucinated)
        # At minimum, neither should be 1.0 and both should be in (0, 1)
        assert 0.0 < s_omit < 1.0 and 0.0 < s_hall < 1.0

    def test_judge_prompt_includes_both_source_and_output(self):
        judge = StubJudge({"facts_preserved": 4, "no_hallucinations": 4})
        self._r().compute("THE_OUTPUT_TEXT", self._ctx("THE_SOURCE_TEXT"), judge=judge)
        prompt = judge.calls[0]
        assert "THE_SOURCE_TEXT" in prompt
        assert "THE_OUTPUT_TEXT" in prompt

    def test_handles_judge_failure_gracefully(self):
        # If judge returns garbage, return a sentinel low score, not crash
        judge = StubJudge({"unknown_key": "x"})
        s = self._r().compute("o", self._ctx("s"), judge=judge)
        assert 0.0 <= s <= 1.0


# ---------- Stubs (must exist; numeric behavior is TBD) ----------

class TestRepetitionRewardStub:
    def test_class_exists_and_returns_float(self):
        r = rewards.RepetitionReward()
        out = r.compute("any text", rewards.RewardContext(source="src"))
        assert isinstance(out, float)
        assert 0.0 <= out <= 1.0


class TestSmoothDifficultyRewardStub:
    def test_class_exists_and_returns_float(self):
        # No judge needed since stub doesn't actually call one yet
        r = rewards.SmoothDifficultyReward(a1_samples=[], a2_samples=[], b1_samples=[])
        out = r.compute("any text", rewards.RewardContext(source="src"), judge=None)
        assert isinstance(out, float)
        assert 0.0 <= out <= 1.0


# ---------- CombinedReward ----------

class TestCombinedReward:
    def test_weighted_sum(self):
        # Two fixed-score components with weights 0.6, 0.4
        class Fixed(rewards.RewardComponent):
            name = "fixed"
            def __init__(self, val): self.val = val
            def compute(self, output, ctx, judge=None): return self.val

        c = rewards.CombinedReward([
            (0.6, Fixed(1.0)),
            (0.4, Fixed(0.0)),
        ])
        assert c.compute("x", rewards.RewardContext(source="s")) == pytest.approx(0.6)

    def test_meaning_gate_zeros_when_meaning_low(self):
        # If a component named 'meaning' scores below the gate threshold,
        # the combined reward must be 0.0 regardless of other components.
        class Fixed(rewards.RewardComponent):
            def __init__(self, name, val): self.name = name; self.val = val
            def compute(self, output, ctx, judge=None): return self.val

        c = rewards.CombinedReward([
            (0.5, Fixed("meaning", 0.4)),  # below default 0.5 gate
            (0.5, Fixed("length", 1.0)),
        ], meaning_gate=0.5)
        assert c.compute("x", rewards.RewardContext(source="s")) == pytest.approx(0.0)

    def test_meaning_gate_passes_when_meaning_ok(self):
        class Fixed(rewards.RewardComponent):
            def __init__(self, name, val): self.name = name; self.val = val
            def compute(self, output, ctx, judge=None): return self.val

        c = rewards.CombinedReward([
            (0.5, Fixed("meaning", 0.6)),
            (0.5, Fixed("length", 1.0)),
        ], meaning_gate=0.5)
        # 0.5*0.6 + 0.5*1.0 = 0.8
        assert c.compute("x", rewards.RewardContext(source="s")) == pytest.approx(0.8)

    def test_combined_returns_value_in_unit_interval(self):
        class Fixed(rewards.RewardComponent):
            name = "fixed"
            def __init__(self, val): self.val = val
            def compute(self, output, ctx, judge=None): return self.val

        c = rewards.CombinedReward([
            (0.5, Fixed(1.0)),
            (0.5, Fixed(1.0)),
        ])
        assert c.compute("x", rewards.RewardContext(source="s")) == pytest.approx(1.0)


# ---------- audit + variety helpers (used by the CLI) ----------

class TestAuditRecord:
    def test_returns_per_component_scores(self):
        out = rewards.audit_record(
            source="cats sleep",
            output="cats sleep here",
            judge=None,
        )
        # Should include each active component, plus 'combined'
        assert "length" in out
        assert "vocab" in out
        assert "meaning" in out
        assert "combined" in out
        for k, v in out.items():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

    def test_meaning_uses_judge_when_given(self):
        judge = StubJudge({"facts_preserved": 5, "no_hallucinations": 5})
        out = rewards.audit_record("source text", "output text", judge=judge)
        assert out["meaning"] == pytest.approx(1.0)
        # Without judge it falls back to mid score
        out2 = rewards.audit_record("source text", "output text", judge=None)
        assert out2["meaning"] == pytest.approx(0.5)


class TestRewardVariety:
    def test_returns_per_prompt_stats(self):
        # Fake "rollouts": 3 prompts × 4 completions per prompt
        prompts = ["src1", "src2", "src3"]
        rollouts_per_prompt = [
            ["a a a", "b b b b b b b b", "c c c c", "d d d d d"],   # prompt 0
            ["e e e e e e", "f f", "g g g g g g g g g g", "h h h"],  # prompt 1
            ["i i i", "j j j", "k k k", "l l l"],                   # prompt 2 — uniform-ish
        ]
        stats = rewards.compute_variety(prompts, rollouts_per_prompt, judge=None)
        assert len(stats["per_prompt"]) == 3
        assert "mean_std" in stats  # average reward std across groups
        for p in stats["per_prompt"]:
            assert "mean" in p and "std" in p and "rewards" in p
            assert len(p["rewards"]) == 4
