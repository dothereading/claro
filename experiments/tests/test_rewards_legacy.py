"""Tests for the GRPO reward components.

Pure-Python rewards (length, vocab, repetition) are exercised end-to-end.
Judge-backed rewards (meaning, difficulty) use a StubJudge that returns a
preset dict; the shared `_judge_bundle` cache is verified in
TestJudgeBundleCache so meaning + difficulty don't double-bill the API.
"""

from __future__ import annotations

import pytest

from claro.verifier import BaseJudge

rewards = pytest.importorskip(
    "experiments.rewards.rewards_legacy", reason="rewards.py not implemented yet (RED)"
)


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


class TestVocabSimplicityCalibration:
    """Calibration regression test: vocab reward must clearly separate
    A2-style outputs from B1+ academic prose.

    Targets (set after sweeping params over the real chosen/rejected/bad
    distributions in data/{sft,dpo}.jsonl; aim for bad clearly below the
    0.5 meaning gate, not at it):
      * good A2 text          → score ≥ 0.85
      * bad B1+ text          → score ≤ 0.40
      * good - bad gap        ≥ 0.45
    """

    GOOD_A2 = (
        "Washington is a city in the United States. "
        "It is the capital of the country. "
        "It is on the east coast, between Virginia and Maryland. "
        "Many people work for the government there."
    )
    # Real B1+ paragraph (a Wikipedia lead from data/sft.jsonl). Multiple
    # sentences each carrying technical vocab — closer to the median-bad
    # case observed in the calibration sweep (mean bad ≈ 0.39).
    BAD_B1 = (
        "François Magendie was a French physiologist, considered a pioneer "
        "of experimental physiology. He is known for describing the foramen "
        "of Magendie. There is also a Magendie sign, a downward and inward "
        "rotation of the eye due to a lesion in the cerebellum. Magendie "
        "was a faculty at the College of France, holding the Chair of "
        "Medicine from 1830 to 1855."
    )

    def test_good_a2_scores_high(self):
        r = rewards.VocabSimplicityReward()
        s = r.compute(self.GOOD_A2, rewards.RewardContext(source="x"))
        assert s >= 0.85, f"good A2 only scored {s:.3f}"

    def test_bad_b1_scores_low(self):
        r = rewards.VocabSimplicityReward()
        s = r.compute(self.BAD_B1, rewards.RewardContext(source="x"))
        assert s <= 0.40, f"bad B1+ scored too high: {s:.3f}"

    def test_separation_at_least_0_45(self):
        r = rewards.VocabSimplicityReward()
        ctx = rewards.RewardContext(source="x")
        good = r.compute(self.GOOD_A2, ctx)
        bad = r.compute(self.BAD_B1, ctx)
        assert good - bad >= 0.45, f"gap too small: good={good:.3f} bad={bad:.3f}"


# ---------- SemanticPreservationReward ----------


class TestSemanticPreservationReward:
    """Meaning reward reads `f` (facts) and `h` (no-hallucinations) from
    the combined judge bundle. Same per-rollout judge call also produces
    `lvl` which SmoothDifficultyReward consumes — see TestJudgeBundleCache."""

    def setup_method(self):
        # Clear the shared judge cache between tests to avoid cross-pollution.
        rewards._judge_cache.clear()

    def _ctx(self, source: str):
        return rewards.RewardContext(source=source)

    def _r(self):
        return rewards.SemanticPreservationReward()

    def test_full_reward_when_judge_says_perfect(self):
        judge = StubJudge({"f": 5, "h": 5, "lvl": "A2"})
        s = self._r().compute("output", self._ctx("source"), judge=judge)
        assert s == pytest.approx(1.0)

    def test_zero_reward_when_judge_says_terrible(self):
        judge = StubJudge({"f": 1, "h": 1, "lvl": "B2+"})
        s = self._r().compute("output", self._ctx("source"), judge=judge)
        assert 0.0 <= s <= 0.3

    def test_partial_when_one_axis_low(self):
        omitted = StubJudge({"f": 3, "h": 5, "lvl": "A2"})
        hallucinated = StubJudge({"f": 5, "h": 3, "lvl": "A2"})
        s_omit = self._r().compute("o", self._ctx("s1"), judge=omitted)
        s_hall = self._r().compute("o", self._ctx("s2"), judge=hallucinated)
        assert 0.0 < s_omit < 1.0 and 0.0 < s_hall < 1.0

    def test_judge_prompt_includes_both_source_and_output(self):
        judge = StubJudge({"f": 4, "h": 4, "lvl": "A2"})
        self._r().compute("THE_OUTPUT_TEXT", self._ctx("THE_SOURCE_TEXT"), judge=judge)
        prompt = judge.calls[0]
        assert "THE_SOURCE_TEXT" in prompt
        assert "THE_OUTPUT_TEXT" in prompt

    def test_handles_judge_failure_gracefully(self):
        judge = StubJudge({"unknown_key": "x"})
        s = self._r().compute("o", self._ctx("s"), judge=judge)
        assert 0.0 <= s <= 1.0


class TestSmoothDifficultyReward:
    """CEFR-level → smooth score: A2 is the target (1.0), drift in either
    direction costs reward. A1 is over-simplified (0.6 — not catastrophic);
    B1+ is under-simplified (0.4 → 0.0)."""

    def setup_method(self):
        rewards._judge_cache.clear()

    def _ctx(self, source: str = "src"):
        return rewards.RewardContext(source=source)

    def _r(self):
        return rewards.SmoothDifficultyReward()

    def test_a2_target_scores_one(self):
        judge = StubJudge({"f": 5, "h": 5, "lvl": "A2"})
        assert self._r().compute("o", self._ctx(), judge=judge) == pytest.approx(1.0)

    def test_a1_over_simplified_scores_partial(self):
        judge = StubJudge({"f": 5, "h": 5, "lvl": "A1"})
        s = self._r().compute("o", self._ctx(), judge=judge)
        assert 0.4 < s < 0.8

    def test_b1_under_simplified_scores_low(self):
        judge = StubJudge({"f": 5, "h": 5, "lvl": "B1"})
        s = self._r().compute("o", self._ctx(), judge=judge)
        assert 0.2 < s < 0.5

    def test_b2plus_scores_zero(self):
        judge = StubJudge({"f": 5, "h": 5, "lvl": "B2+"})
        s = self._r().compute("o", self._ctx(), judge=judge)
        assert s <= 0.1

    def test_no_judge_returns_neutral(self):
        # When no judge configured, contribute a constant mid-score (no
        # signal but no crash), matching the meaning-reward fallback pattern.
        assert self._r().compute("o", self._ctx(), judge=None) == 0.5

    def test_unrecognized_level_returns_neutral(self):
        judge = StubJudge({"f": 5, "h": 5, "lvl": "QQQ"})
        s = self._r().compute("o", self._ctx(), judge=judge)
        assert s == 0.5

    def test_a2_ordering(self):
        # Monotonicity check: B2+ ≤ B1 ≤ A1 < A2; A2 is the apex.
        def s(level):
            return self._r().compute(
                "o", self._ctx(level), judge=StubJudge({"f": 5, "h": 5, "lvl": level})
            )

        assert s("A2") > s("A1") > s("B1") >= s("B2+")


class TestJudgeBundleCache:
    """Meaning + difficulty rewards must share a single judge call per
    (source, output) pair — saves a round trip on every rollout during
    training."""

    def setup_method(self):
        rewards._judge_cache.clear()

    def test_second_reward_hits_cache(self):
        judge = StubJudge({"f": 5, "h": 5, "lvl": "A2"})
        ctx = rewards.RewardContext(source="src")
        rewards.SemanticPreservationReward().compute("out", ctx, judge=judge)
        rewards.SmoothDifficultyReward().compute("out", ctx, judge=judge)
        # One judge call serves both rewards.
        assert len(judge.calls) == 1, f"expected 1 judge call, got {len(judge.calls)}"

    def test_different_outputs_each_call_judge(self):
        judge = StubJudge({"f": 5, "h": 5, "lvl": "A2"})
        ctx = rewards.RewardContext(source="src")
        rewards.SemanticPreservationReward().compute("out_A", ctx, judge=judge)
        rewards.SemanticPreservationReward().compute("out_B", ctx, judge=judge)
        # Different completions → no cache reuse → two calls.
        assert len(judge.calls) == 2


# ---------- Stubs (must exist; numeric behavior is TBD) ----------


class TestNoMarkdownReward:
    """Format-adherence guardrail. The DPO data already steers against
    markdown, but a small reward signal keeps the policy from drifting
    back into bullets/headings/bold during GRPO.

    Binary by design: any markdown marker → 0.0, none → 1.0. Subtle
    "is it kind of markdown" cases aren't worth the false-positive risk.
    """

    def test_plain_prose_scores_one(self):
        r = rewards.NoMarkdownReward()
        text = (
            "Washington, D.C. is the capital of the United States. "
            "The city is on the Potomac River. Many people work for "
            "the government there."
        )
        assert r.compute(text, rewards.RewardContext(source="x")) == 1.0

    def test_empty_text_scores_one(self):
        r = rewards.NoMarkdownReward()
        assert r.compute("", rewards.RewardContext(source="x")) == 1.0

    def test_bold_caught(self):
        r = rewards.NoMarkdownReward()
        s = r.compute("Plain prose with **bold word** in it.", rewards.RewardContext(source="x"))
        assert s == 0.0

    def test_heading_caught(self):
        r = rewards.NoMarkdownReward()
        for text in ("# Title\nText.", "## Section\nText.", "### Sub\nText."):
            s = r.compute(text, rewards.RewardContext(source="x"))
            assert s == 0.0, f"heading not caught: {text!r}"

    def test_bullet_list_caught(self):
        r = rewards.NoMarkdownReward()
        text = "Three items:\n- first\n- second\n- third"
        assert r.compute(text, rewards.RewardContext(source="x")) == 0.0

    def test_numbered_list_caught(self):
        r = rewards.NoMarkdownReward()
        text = "Steps:\n1. Open the door.\n2. Walk inside."
        assert r.compute(text, rewards.RewardContext(source="x")) == 0.0

    def test_inline_code_caught(self):
        r = rewards.NoMarkdownReward()
        s = r.compute("The variable is `x` in the code.", rewards.RewardContext(source="x"))
        assert s == 0.0

    def test_link_caught(self):
        r = rewards.NoMarkdownReward()
        s = r.compute(
            "Visit [the site](https://example.com) for more.", rewards.RewardContext(source="x")
        )
        assert s == 0.0

    def test_em_dash_not_treated_as_bullet(self):
        # Opus loves em-dashes; they must not trip the bullet detector.
        r = rewards.NoMarkdownReward()
        text = "Washington, D.C. — the capital — is on the Potomac River."
        assert r.compute(text, rewards.RewardContext(source="x")) == 1.0

    def test_sentence_initial_number_period_not_caught(self):
        # "Section 1. Introduction" should NOT count as a numbered list.
        r = rewards.NoMarkdownReward()
        text = "In 1985 the building opened. By 1992 it was rebuilt."
        assert r.compute(text, rewards.RewardContext(source="x")) == 1.0


class TestRepetitionReward:
    """Catches the GRPO failure mode seen in the smoke run: policy degenerates
    into 4-gram loops that max out the completion budget. Without this signal
    length/vocab can stay artificially high on degenerate outputs.

    Targets (calibrated against good A2 prose vs synthetic loops):
      * normal A2 prose          → ≥ 0.85
      * 4-gram loop ("of the …") → ≤ 0.20
      * repeated whole sentence  → ≤ 0.30
      * very short text          → 1.0  (can't penalize what we can't measure)
    """

    NORMAL_A2 = (
        "Washington is a city in the United States. "
        "It is the capital of the country. "
        "It is on the east coast, between Virginia and Maryland. "
        "Many people work for the government there."
    )

    LOOP_4GRAM = (
        "The president of the country of the country of the country of the "
        "country of the country of the country of the country of the country "
        "of the country of the country of the country of the country."
    )

    REPEATED_SENTENCE = (
        "Cats sleep all day. Cats sleep all day. Cats sleep all day. "
        "Cats sleep all day. Cats sleep all day."
    )

    def test_normal_a2_scores_high(self):
        r = rewards.RepetitionReward()
        s = r.compute(self.NORMAL_A2, rewards.RewardContext(source="x"))
        assert s >= 0.85, f"normal A2 scored {s:.3f}"

    def test_4gram_loop_scores_low(self):
        r = rewards.RepetitionReward()
        s = r.compute(self.LOOP_4GRAM, rewards.RewardContext(source="x"))
        assert s <= 0.20, f"4-gram loop scored {s:.3f}"

    def test_repeated_sentence_scores_low(self):
        r = rewards.RepetitionReward()
        s = r.compute(self.REPEATED_SENTENCE, rewards.RewardContext(source="x"))
        assert s <= 0.30, f"repeated sentence scored {s:.3f}"

    def test_short_text_returns_one(self):
        # Too few tokens to measure repetition meaningfully; return 1.0
        # rather than incidentally penalize short outputs.
        r = rewards.RepetitionReward()
        s = r.compute("Hello world.", rewards.RewardContext(source="x"))
        assert s == 1.0

    def test_empty_text_returns_one(self):
        r = rewards.RepetitionReward()
        assert r.compute("", rewards.RewardContext(source="x")) == 1.0

    def test_score_in_unit_interval(self):
        r = rewards.RepetitionReward()
        for text in (self.NORMAL_A2, self.LOOP_4GRAM, self.REPEATED_SENTENCE):
            s = r.compute(text, rewards.RewardContext(source="x"))
            assert 0.0 <= s <= 1.0


# ---------- CombinedReward ----------


class TestCombinedReward:
    def test_weighted_sum(self):
        # Two fixed-score components with weights 0.6, 0.4
        class Fixed(rewards.RewardComponent):
            name = "fixed"

            def __init__(self, val):
                self.val = val

            def compute(self, output, ctx, judge=None):
                return self.val

        c = rewards.CombinedReward(
            [
                (0.6, Fixed(1.0)),
                (0.4, Fixed(0.0)),
            ]
        )
        assert c.compute("x", rewards.RewardContext(source="s")) == pytest.approx(0.6)

    def test_meaning_gate_zeros_when_meaning_low(self):
        # If a component named 'meaning' scores below the gate threshold,
        # the combined reward must be 0.0 regardless of other components.
        class Fixed(rewards.RewardComponent):
            def __init__(self, name, val):
                self.name = name
                self.val = val

            def compute(self, output, ctx, judge=None):
                return self.val

        c = rewards.CombinedReward(
            [
                (0.5, Fixed("meaning", 0.4)),  # below default 0.5 gate
                (0.5, Fixed("length", 1.0)),
            ],
            meaning_gate=0.5,
        )
        assert c.compute("x", rewards.RewardContext(source="s")) == pytest.approx(0.0)

    def test_meaning_gate_passes_when_meaning_ok(self):
        class Fixed(rewards.RewardComponent):
            def __init__(self, name, val):
                self.name = name
                self.val = val

            def compute(self, output, ctx, judge=None):
                return self.val

        c = rewards.CombinedReward(
            [
                (0.5, Fixed("meaning", 0.6)),
                (0.5, Fixed("length", 1.0)),
            ],
            meaning_gate=0.5,
        )
        # 0.5*0.6 + 0.5*1.0 = 0.8
        assert c.compute("x", rewards.RewardContext(source="s")) == pytest.approx(0.8)

    def test_combined_returns_value_in_unit_interval(self):
        class Fixed(rewards.RewardComponent):
            name = "fixed"

            def __init__(self, val):
                self.val = val

            def compute(self, output, ctx, judge=None):
                return self.val

        c = rewards.CombinedReward(
            [
                (0.5, Fixed(1.0)),
                (0.5, Fixed(1.0)),
            ]
        )
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
        judge = StubJudge({"f": 5, "h": 5, "lvl": "A2"})
        out = rewards.audit_record("source text", "output text", judge=judge)
        assert out["meaning"] == pytest.approx(1.0)
        # Without judge it falls back to mid score
        out2 = rewards.audit_record("source text", "output text", judge=None)
        assert out2["meaning"] == pytest.approx(0.5)


class TestRewardFunctionKwargs:
    """mlx-lm-lora invokes registered reward functions with the call shape
    `reward_func(prompts=..., completions=..., answer=..., types=...)`.
    The kwarg is `answer` (singular). Our wrappers must accept that exact
    keyword or training crashes immediately."""

    def test_length_reward_accepts_answer_kwarg(self):
        out = rewards.length_reward(
            prompts=["a b c d e"],
            completions=["a b c d"],
            answer=["x y z"],
            types=None,
        )
        assert isinstance(out, list) and len(out) == 1
        assert isinstance(out[0], float)

    def test_vocab_reward_accepts_answer_kwarg(self):
        out = rewards.vocab_reward(
            prompts=["src"],
            completions=["The cat sat on the mat."],
            answer=["ref"],
            types=None,
        )
        assert isinstance(out, list) and len(out) == 1

    def test_meaning_reward_accepts_answer_kwarg(self):
        # No judge configured → fallback 0.5 per item, no HTTP.
        out = rewards.meaning_reward(
            prompts=["src"],
            completions=["out"],
            answer=["ref"],
            types=None,
        )
        assert out == [0.5]


class TestGetJudgeFactory:
    """`rewards._get_judge` picks a backend from env vars:
    * MEANING_JUDGE_BACKEND=openrouter  → OpenRouter (requires OPENROUTER_API_KEY)
    * MEANING_JUDGE_URL set             → local LM Studio (back-compat path)
    * neither                           → None (meaning reward returns 0.5)
    """

    def setup_method(self):
        # Drop the function-attribute cache between tests.
        if hasattr(rewards._get_judge, "_cached"):
            del rewards._get_judge._cached

    def test_returns_none_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv("MEANING_JUDGE_BACKEND", raising=False)
        monkeypatch.delenv("MEANING_JUDGE_URL", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert rewards._get_judge() is None

    def test_local_path_when_only_url_set(self, monkeypatch):
        monkeypatch.delenv("MEANING_JUDGE_BACKEND", raising=False)
        monkeypatch.setenv("MEANING_JUDGE_URL", "http://127.0.0.1:1234/v1")
        monkeypatch.setenv("MEANING_JUDGE_MODEL", "google/gemma-4-26b-a4b")
        j = rewards._get_judge()
        assert j is not None
        assert j.api_key is None
        assert "127.0.0.1" in j.endpoint

    def test_openrouter_path_uses_haiku_default_and_key(self, monkeypatch):
        monkeypatch.setenv("MEANING_JUDGE_BACKEND", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.delenv("MEANING_JUDGE_MODEL", raising=False)
        monkeypatch.delenv("MEANING_JUDGE_URL", raising=False)
        j = rewards._get_judge()
        assert j is not None
        assert j.api_key == "sk-or-test"
        assert j.model == "anthropic/claude-haiku-4-5"
        assert "openrouter.ai" in j.endpoint

    def test_openrouter_backend_without_key_raises(self, monkeypatch):
        monkeypatch.setenv("MEANING_JUDGE_BACKEND", "openrouter")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            rewards._get_judge()


class TestRewardVariety:
    def test_returns_per_prompt_stats(self):
        # Fake "rollouts": 3 prompts × 4 completions per prompt
        prompts = ["src1", "src2", "src3"]
        rollouts_per_prompt = [
            ["a a a", "b b b b b b b b", "c c c c", "d d d d d"],  # prompt 0
            ["e e e e e e", "f f", "g g g g g g g g g g", "h h h"],  # prompt 1
            ["i i i", "j j j", "k k k", "l l l"],  # prompt 2 — uniform-ish
        ]
        stats = rewards.compute_variety(prompts, rollouts_per_prompt, judge=None)
        assert len(stats["per_prompt"]) == 3
        assert "mean_std" in stats  # average reward std across groups
        for p in stats["per_prompt"]:
            assert "mean" in p and "std" in p and "rewards" in p
            assert len(p["rewards"]) == 4


# ---------- v5: Entity preservation + combined v5 ----------


class TestEntityPreservationReward:
    """v5 component: % of named entities from source that appear in output.
    Entities = capitalized multi-word phrases, all-caps acronyms (UCL, MI6),
    and 2-4 digit numbers (1750, 2013)."""

    def _r(self):
        return rewards.EntityPreservationReward()

    def test_full_credit_when_all_entities_preserved(self):
        src = "Christina Pagel works at University College London (UCL) since 2017."
        out = "Christina Pagel works at University College London (UCL). She joined in 2017."
        s = self._r().compute(out, rewards.RewardContext(source=src))
        assert s == pytest.approx(1.0)

    def test_zero_credit_when_no_entities_preserved(self):
        src = "Christina Pagel works at University College London since 2017."
        out = "A scientist works at a university."
        s = self._r().compute(out, rewards.RewardContext(source=src))
        assert s == pytest.approx(0.0)

    def test_partial_credit_proportional(self):
        # source has 3 entities (Christina Pagel, University College London, 2017)
        # output keeps 2/3
        src = "Christina Pagel works at University College London since 2017."
        out = "Christina Pagel works at University College London."
        s = self._r().compute(out, rewards.RewardContext(source=src))
        assert s == pytest.approx(2 / 3, abs=0.05)

    def test_acronyms_detected(self):
        src = "She works at the GMC and UCL."
        out = "She works at the GMC and UCL."
        s = self._r().compute(out, rewards.RewardContext(source=src))
        assert s == pytest.approx(1.0)

    def test_case_insensitive_match(self):
        src = "Cayan Tower is in Dubai."
        out = "The cayan tower is in dubai."
        s = self._r().compute(out, rewards.RewardContext(source=src))
        assert s == pytest.approx(1.0)

    def test_no_entities_in_source_returns_one(self):
        src = "the cat sat on the mat"
        out = "a cat is on a mat"
        s = self._r().compute(out, rewards.RewardContext(source=src))
        assert s == pytest.approx(1.0)


class TestCombinedRewardV5:
    """v5: base = 0.6*meaning + 0.4*entity, multiplied by difficulty_factor
    and length_factor, with hard gates for meaning<0.3, markdown, and loops.
    """

    def setup_method(self):
        rewards._judge_cache.clear()

    def _ctx(self, source: str):
        return rewards.RewardContext(source=source)

    A2_CLEAN = (
        "Christina Pagel is a German-British mathematician. "
        "She works at University College London. UCL is a famous university. "
        "She started there in 2017 and helps doctors do research."
    )

    # Picked so A2_CLEAN (32 words) / SRC (25 words) ≈ 1.28, inside [0.8, 1.4]
    SRC_SHORT = (
        "Christina Pagel is a German-British mathematician at University "
        "College London. She has worked at UCL since 2017 doing applied "
        "medical research."
    )

    def test_clean_a2_full_credit(self):
        judge = StubJudge({"f": 5, "h": 5, "lvl": "A2"})
        r = rewards._default_combined_v5()
        s = r.compute(self.A2_CLEAN, self._ctx(self.SRC_SHORT), judge=judge)
        # meaning=1, diff=A2(1.0), entity≈1, length OK → ~1.0
        assert s >= 0.9

    def test_meaning_gate_zeroes_reward(self):
        # meaning < 0.3 → 0
        judge = StubJudge({"f": 1, "h": 1, "lvl": "A2"})
        r = rewards._default_combined_v5()
        s = r.compute(self.A2_CLEAN, self._ctx(self.SRC_SHORT), judge=judge)
        assert s == pytest.approx(0.0)

    def test_markdown_gate_zeroes_reward(self):
        judge = StubJudge({"f": 5, "h": 5, "lvl": "A2"})
        with_markdown = "**Christina Pagel** works at UCL since 2017."
        r = rewards._default_combined_v5()
        s = r.compute(with_markdown, self._ctx(self.SRC_SHORT), judge=judge)
        assert s == pytest.approx(0.0)

    def test_loop_gate_zeroes_reward(self):
        judge = StubJudge({"f": 5, "h": 5, "lvl": "A2"})
        loop = "Pagel works at UCL. Pagel works at UCL. Pagel works at UCL. Pagel works at UCL."
        r = rewards._default_combined_v5()
        s = r.compute(loop, self._ctx(self.SRC_SHORT), judge=judge)
        assert s == pytest.approx(0.0)

    def test_a1_difficulty_softens_reward(self):
        judge = StubJudge({"f": 5, "h": 5, "lvl": "A1"})
        r = rewards._default_combined_v5()
        s = r.compute(self.A2_CLEAN, self._ctx(self.SRC_SHORT), judge=judge)
        # base ≈ 1.0, A1 factor = 0.85, length_factor=1.0 → ~0.85
        assert 0.75 <= s <= 0.95

    def test_b1_difficulty_more_harshly_penalized(self):
        judge = StubJudge({"f": 5, "h": 5, "lvl": "B1"})
        r = rewards._default_combined_v5()
        s = r.compute(self.A2_CLEAN, self._ctx(self.SRC_SHORT), judge=judge)
        # B1 factor = 0.6 — strictly less than A1's 0.85
        assert 0.45 <= s <= 0.7

    def test_b2plus_zeroes_reward(self):
        judge = StubJudge({"f": 5, "h": 5, "lvl": "B2+"})
        r = rewards._default_combined_v5()
        s = r.compute(self.A2_CLEAN, self._ctx(self.SRC_SHORT), judge=judge)
        assert s == pytest.approx(0.0)

    def test_too_short_length_penalized(self):
        judge = StubJudge({"f": 5, "h": 5, "lvl": "A2"})
        very_short = "She works."  # 2 words; SRC is 25 → ratio 0.08 < 0.5 floor → 0
        r = rewards._default_combined_v5()
        s = r.compute(very_short, self._ctx(self.SRC_SHORT), judge=judge)
        assert s == pytest.approx(0.0)

    def test_slightly_long_not_penalized(self):
        # Asymmetric: ratio 1.3 should still be full credit (window [0.8, 1.4])
        judge = StubJudge({"f": 5, "h": 5, "lvl": "A2"})
        src = "X Y Z A B C D E F G."  # 10 words
        long_out = "X Y Z A B C D E F G H I J."  # 13 words, ratio 1.3
        r = rewards._default_combined_v5()
        s = r.compute(long_out, self._ctx(src), judge=judge)
        # Length should be full credit; only the meaning + entity drives
        assert s >= 0.5  # broad floor; depends on entity match on token-soup


# ---------- v6: 8-axis fidelity + per-group A2-ranking ----------


rewards_v6 = pytest.importorskip(
    "experiments.rewards.rewards_v6", reason="rewards_v6.py not implemented yet (RED)"
)


class V6StubJudge(BaseJudge):
    """Dispatches on prompt header. The fidelity prompt starts with
    "You are auditing"; the rank prompt starts with "You are ranking".
    Each branch returns the preset response and records the call so
    tests can assert call counts."""

    FIDELITY_MARKER = "auditing a text simplification"
    RANK_MARKER = "ranking text simplifications"

    def __init__(self, fidelity_response: dict | None = None, rank_response: dict | None = None):
        self.fidelity_response = fidelity_response if fidelity_response is not None else {
            "n_source_claims": 1,
            "missing_full": 0,
        }
        self.rank_response = rank_response if rank_response is not None else {"order": [0]}
        self.fidelity_calls: list[str] = []
        self.rank_calls: list[str] = []

    def evaluate(self, prompt: str) -> dict:
        if self.FIDELITY_MARKER in prompt:
            self.fidelity_calls.append(prompt)
            return self.fidelity_response
        if self.RANK_MARKER in prompt:
            self.rank_calls.append(prompt)
            return self.rank_response
        raise AssertionError(f"V6StubJudge: unrecognized prompt header: {prompt[:80]!r}")


@pytest.fixture(autouse=True)
def _clear_v6_fidelity_cache():
    """Each test starts with an empty fidelity cache so cache state doesn't
    leak across tests (the cache lives at module scope)."""
    rewards_v6._fidelity_cache.clear()
    yield
    rewards_v6._fidelity_cache.clear()


class TestFidelityReward:
    def _ctx(self, source: str = "src"):
        return rewards.RewardContext(source=source)

    def test_zero_errors_gives_full_credit(self):
        judge = V6StubJudge(
            fidelity_response={
                "n_source_claims": 5,
                "missing_full": 0, "missing_specificity": 0, "missing_nuance": 0,
                "hallucinated": 0, "off_topic": 0,
                "factuality_distorted": 0, "fidelity_major": 0, "fidelity_minor": 0,
            }
        )
        r = rewards_v6.FidelityReward()
        assert r.compute("out", self._ctx(), judge=judge) == pytest.approx(1.0)

    def test_all_claims_missing_gives_zero(self):
        # 3 claims × weight 4 = 12; max = 4*3 = 12 → 1 - 12/12 = 0
        judge = V6StubJudge(
            fidelity_response={"n_source_claims": 3, "hallucinated": 3}
        )
        r = rewards_v6.FidelityReward()
        assert r.compute("out", self._ctx(), judge=judge) == pytest.approx(0.0)

    def test_each_error_type_contributes_its_weight(self):
        # n_claims=10 → max_errors=40. Each axis below contributes 1*weight.
        cases = {
            "missing_full": 2,
            "missing_specificity": 1,
            "missing_nuance": 2,
            "hallucinated": 4,
            "off_topic": 1,
            "factuality_distorted": 4,
            "fidelity_major": 3,
            "fidelity_minor": 1,
        }
        for axis, weight in cases.items():
            judge = V6StubJudge(
                fidelity_response={"n_source_claims": 10, axis: 1}
            )
            r = rewards_v6.FidelityReward()
            expected = 1.0 - weight / 40
            got = r.compute("out", self._ctx(source=f"src-{axis}"), judge=judge)
            assert got == pytest.approx(expected), f"{axis}: got {got}, expected {expected}"

    def test_malformed_judge_response_returns_neutral(self):
        judge = V6StubJudge(fidelity_response={"unrelated": "junk"})
        r = rewards_v6.FidelityReward()
        # Missing n_source_claims → neutral 0.5
        assert r.compute("out", self._ctx(), judge=judge) == pytest.approx(0.5)

    def test_zero_source_claims_gives_full_credit(self):
        judge = V6StubJudge(fidelity_response={"n_source_claims": 0})
        r = rewards_v6.FidelityReward()
        assert r.compute("out", self._ctx(), judge=judge) == pytest.approx(1.0)

    def test_caches_per_source_output_pair(self):
        judge = V6StubJudge(fidelity_response={"n_source_claims": 1, "missing_full": 0})
        r = rewards_v6.FidelityReward()
        r.compute("output A", self._ctx(source="src"), judge=judge)
        r.compute("output A", self._ctx(source="src"), judge=judge)
        # Second compute on same (source, output) hits cache, no second call
        assert len(judge.fidelity_calls) == 1
        # Different output → new call
        r.compute("output B", self._ctx(source="src"), judge=judge)
        assert len(judge.fidelity_calls) == 2

    def test_no_judge_returns_neutral(self):
        r = rewards_v6.FidelityReward()
        assert r.compute("out", self._ctx(), judge=None) == pytest.approx(0.5)

    def test_judge_exception_returns_neutral(self):
        class BoomJudge(BaseJudge):
            def evaluate(self, prompt):
                raise RuntimeError("transport boom")

        r = rewards_v6.FidelityReward()
        assert r.compute("out", self._ctx(), judge=BoomJudge()) == pytest.approx(0.5)


class TestGroupRankReward:
    def test_valid_ranking_returns_linear_spread(self):
        # order [0,1,2,3] → rollout 0 is best, rollout 3 is worst
        judge = V6StubJudge(rank_response={"order": [0, 1, 2, 3]})
        r = rewards_v6.GroupRankReward()
        scores = r.compute_group("src", ["a", "b", "c", "d"], judge=judge)
        assert scores == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0])

    def test_permuted_ranking_assigns_scores_by_id(self):
        # rollout id 3 ranked best, id 0 ranked worst
        judge = V6StubJudge(rank_response={"order": [3, 1, 2, 0]})
        r = rewards_v6.GroupRankReward()
        scores = r.compute_group("src", ["w", "x", "y", "z"], judge=judge)
        # By position: id 0 has rank 3 → 0.0; id 3 has rank 0 → 1.0
        assert scores[0] == pytest.approx(0.0)
        assert scores[3] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(2 / 3)
        assert scores[2] == pytest.approx(1 / 3)

    def test_malformed_response_falls_back_to_neutral(self):
        judge = V6StubJudge(rank_response={"garbage": True})
        r = rewards_v6.GroupRankReward()
        scores = r.compute_group("src", ["a", "b", "c"], judge=judge)
        assert scores == [0.5, 0.5, 0.5]

    def test_missing_ids_in_order_falls_back(self):
        # Length matches G=3 but has dup + missing id 2
        judge = V6StubJudge(rank_response={"order": [0, 1, 1]})
        r = rewards_v6.GroupRankReward()
        scores = r.compute_group("src", ["a", "b", "c"], judge=judge)
        assert scores == [0.5, 0.5, 0.5]

    def test_single_rollout_returns_one(self):
        judge = V6StubJudge(rank_response={"order": [0]})
        r = rewards_v6.GroupRankReward()
        # G=1: no judge call needed; we short-circuit to [1.0]
        assert r.compute_group("src", ["only"], judge=judge) == [1.0]
        assert judge.rank_calls == []

    def test_group_of_eight_gives_linear_spread(self):
        judge = V6StubJudge(rank_response={"order": [3, 0, 7, 1, 5, 2, 6, 4]})
        r = rewards_v6.GroupRankReward()
        outputs = [f"o{i}" for i in range(8)]
        scores = r.compute_group("src", outputs, judge=judge)
        # Verify scores span [0, 1] linearly and are unique
        assert min(scores) == pytest.approx(0.0)
        assert max(scores) == pytest.approx(1.0)
        assert len(set(scores)) == 8
        # Spot-check: id 3 ranked first → score 1.0; id 4 ranked last → 0.0
        assert scores[3] == pytest.approx(1.0)
        assert scores[4] == pytest.approx(0.0)

    def test_no_judge_returns_all_neutral(self):
        r = rewards_v6.GroupRankReward()
        assert r.compute_group("src", ["a", "b"], judge=None) == [0.5, 0.5]


class TestCombinedRewardV6:
    """End-to-end behavior of the multiplicative combined reward.

    Source paragraphs are sized so length_factor is ~1.0 by default
    (output ≈ source word count). We tweak length and content for the
    specific edge cases."""

    # 20-word source with enough lexical variety not to trigger the loop gate
    # (which fires when the distinct-4gram ratio drops below 0.55).
    SRC = (
        "The library opened in 1934 and stood near the river for many years. "
        "Many people visited it during quiet afternoons throughout autumn."
    )
    GOOD_OUT = (
        "A library opened in 1934 and stood near the river for many years. "
        "Many readers visited it on quiet afternoons each autumn season."
    )

    HIGH_FID = {
        "n_source_claims": 5,
        "missing_full": 0, "missing_specificity": 0, "missing_nuance": 0,
        "hallucinated": 0, "off_topic": 0,
        "factuality_distorted": 0, "fidelity_major": 0, "fidelity_minor": 0,
    }

    def test_clean_output_high_fidelity_top_rank_gives_near_one(self):
        judge = V6StubJudge(
            fidelity_response=self.HIGH_FID,
            rank_response={"order": [0, 1]},
        )
        r = rewards_v6._default_combined_v6()
        scores = r.compute_group(self.SRC, [self.GOOD_OUT, self.GOOD_OUT], judge=judge)
        # rollout 0 ranked best: base = 0.5*1.0 + 0.5*1.0 = 1.0
        # length_factor ≈ 1.0 (output is 23 / source 22 words), gate = 1.0
        assert scores[0] == pytest.approx(1.0, abs=0.02)

    def test_markdown_detected_zeros_reward(self):
        markdown_out = "**bold word** " + self.GOOD_OUT
        judge = V6StubJudge(
            fidelity_response=self.HIGH_FID,
            rank_response={"order": [0]},
        )
        r = rewards_v6._default_combined_v6()
        scores = r.compute_group(self.SRC, [markdown_out], judge=judge)
        assert scores == [0.0]

    def test_loop_detected_zeros_reward(self):
        # Repeated sentence pattern fires _v5_has_loop
        looped = "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat."
        judge = V6StubJudge(
            fidelity_response=self.HIGH_FID,
            rank_response={"order": [0]},
        )
        r = rewards_v6._default_combined_v6()
        # Use a source matching length so the length factor doesn't already kill it
        src = " ".join(["w"] * len(looped.split()))
        scores = r.compute_group(src, [looped], judge=judge)
        assert scores == [0.0]

    def test_very_short_output_zeroes_length_factor(self):
        # ratio ~ 1/20 = 0.05 → exp(-(0.05-1)^2 / 0.32) ≈ exp(-2.82) ≈ 0.06
        # Plenty small once multiplied through. Tight ceiling here.
        short = "tiny"
        judge = V6StubJudge(
            fidelity_response=self.HIGH_FID,
            rank_response={"order": [0]},
        )
        r = rewards_v6._default_combined_v6()
        scores = r.compute_group(self.SRC, [short], judge=judge)
        assert scores[0] < 0.1

    def test_slightly_long_output_retains_most_reward(self):
        # ratio ≈ 1.3 with sigma=0.4: exp(-0.09/0.32) ≈ 0.755
        # NOTE: V6 spec text claims this should be "≈ same as ratio=1.0",
        # but with the spec's σ=0.4 the gaussian is sharper than that.
        # This test pins the actual behavior; raise σ if the comment-vs-code
        # mismatch is resolved later.
        # SRC is 22 words; GOOD_OUT is 23; appended phrase is 6 → 29 → ratio ≈ 1.32.
        slightly_long = self.GOOD_OUT + " A garden bloomed each spring."
        judge = V6StubJudge(
            fidelity_response=self.HIGH_FID,
            rank_response={"order": [0]},
        )
        r = rewards_v6._default_combined_v6()
        scores = r.compute_group(self.SRC, [slightly_long], judge=judge)
        # Full base = 1.0, length factor drops with ratio; should retain >0.5
        assert 0.5 < scores[0] < 0.9

    def test_soft_fidelity_floor_attenuates_high_rank_low_fidelity(self):
        # fidelity ≈ 0.15: with n_claims=10, weighted_errors ≈ 34
        # Try n_claims=10, factuality_distorted=8 → 32/40 → fidelity = 0.2 (boundary)
        # Use n_claims=5, hallucinated=4 → 16/20 → fidelity = 0.2 — at the floor.
        # We want STRICTLY < 0.2: hallucinated=4, off_topic=1 with n=5 → 17/20 → 0.15
        low_fid = {
            "n_source_claims": 5,
            "hallucinated": 4,
            "off_topic": 1,
        }
        judge = V6StubJudge(
            fidelity_response=low_fid,
            rank_response={"order": [0]},
        )
        r = rewards_v6._default_combined_v6()
        scores = r.compute_group(self.SRC, [self.GOOD_OUT], judge=judge)
        # Unattenuated base = 0.5*0.15 + 0.5*1.0 = 0.575; length ≈ 1.0
        # Attenuated: 0.575 * 0.2 = 0.115
        assert scores[0] == pytest.approx(0.115, abs=0.01)

    def test_output_order_preserved_in_compute_group(self):
        judge = V6StubJudge(
            fidelity_response=self.HIGH_FID,
            rank_response={"order": [2, 0, 1]},
        )
        r = rewards_v6._default_combined_v6()
        outputs = [self.GOOD_OUT + " a", self.GOOD_OUT + " b", self.GOOD_OUT + " c"]
        scores = r.compute_group(self.SRC, outputs, judge=judge)
        assert len(scores) == 3
        # id 2 ranked best → highest score; id 1 ranked last → lowest
        assert scores[2] > scores[0] > scores[1]

    def test_compute_group_returns_g_floats_for_g_outputs(self):
        judge = V6StubJudge(
            fidelity_response=self.HIGH_FID,
            rank_response={"order": list(range(5))},
        )
        r = rewards_v6._default_combined_v6()
        outputs = [self.GOOD_OUT for _ in range(5)]
        scores = r.compute_group(self.SRC, outputs, judge=judge)
        assert len(scores) == 5
        assert all(isinstance(s, float) for s in scores)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_compute_single_uses_rank_one(self):
        # Offline-audit path: rank_score is implicitly 1.0
        judge = V6StubJudge(fidelity_response=self.HIGH_FID)
        r = rewards_v6._default_combined_v6()
        s = r.compute(self.GOOD_OUT, rewards.RewardContext(source=self.SRC), judge=judge)
        # base = 0.5*1.0 + 0.5*1.0 = 1.0, length ≈ 1.0
        assert s == pytest.approx(1.0, abs=0.02)


class TestV6RegisteredFunction:
    """The mlx-lm-lora entry point. Group detection walks contiguous runs
    of identical prompts. We monkeypatch `_get_judge` so the function
    talks to our stub instead of needing OPENROUTER env."""

    HIGH_FID = TestCombinedRewardV6.HIGH_FID

    def test_batch_two_group_two_runs_two_rank_calls(self, monkeypatch):
        judge = V6StubJudge(
            fidelity_response=self.HIGH_FID,
            rank_response={"order": [0, 1]},
        )
        monkeypatch.setattr(rewards_v6, "_get_judge", lambda: judge)
        out = rewards_v6.v6_combined_reward(
            prompts=["src1", "src1", "src2", "src2"],
            completions=["a", "b", "c", "d"],
            answer=["", "", "", ""],
            types=None,
        )
        assert len(out) == 4
        # 2 groups → 2 rank calls; 4 unique (source, output) pairs → 4 fidelity calls
        assert len(judge.rank_calls) == 2
        assert len(judge.fidelity_calls) == 4

    def test_batch_one_group_four_runs_one_rank_call(self, monkeypatch):
        judge = V6StubJudge(
            fidelity_response=self.HIGH_FID,
            rank_response={"order": [0, 1, 2, 3]},
        )
        monkeypatch.setattr(rewards_v6, "_get_judge", lambda: judge)
        out = rewards_v6.v6_combined_reward(
            prompts=["src1"] * 4,
            completions=["a", "b", "c", "d"],
            answer=[""] * 4,
            types=None,
        )
        assert len(out) == 4
        assert len(judge.rank_calls) == 1
        assert len(judge.fidelity_calls) == 4

    def test_result_length_matches_completions(self, monkeypatch):
        judge = V6StubJudge(
            fidelity_response=self.HIGH_FID,
            rank_response={"order": [0, 1, 2]},
        )
        monkeypatch.setattr(rewards_v6, "_get_judge", lambda: judge)
        # Batch=2 with G=3
        out = rewards_v6.v6_combined_reward(
            prompts=["s1", "s1", "s1", "s2", "s2", "s2"],
            completions=["a", "b", "c", "d", "e", "f"],
            answer=[""] * 6,
            types=None,
        )
        assert len(out) == 6
        assert all(isinstance(x, float) for x in out)

    def test_group_detection_runs(self):
        # Pure helper test, no judge needed
        runs = rewards_v6._group_runs(["a", "a", "b", "b", "c"])
        assert runs == [(0, 2), (2, 4), (4, 5)]
        assert rewards_v6._group_runs([]) == []
        assert rewards_v6._group_runs(["only"]) == [(0, 1)]


# ---------- v7: sparse-geometric ranking, no fidelity ----------


rewards_v7 = pytest.importorskip(
    "experiments.rewards.rewards_v7", reason="rewards_v7.py not implemented yet (RED)"
)


class V7StubJudge(BaseJudge):
    """One prompt type only. Asserts the prompt is the v7 ranking prompt
    and returns whatever `reply` was configured (list, dict, or string)."""

    def __init__(self, reply: Any = None):
        self.reply = reply
        self.calls: list[str] = []

    def evaluate(self, prompt: str) -> Any:
        assert rewards_v7._RANK_PROMPT_MARKER in prompt, (
            f"V7StubJudge: prompt missing v7 marker: {prompt[:80]!r}"
        )
        self.calls.append(prompt)
        return self.reply


class TestScoreRanksSparse:
    def test_g8_default_top4_geometric(self):
        # rank 0 → 1.0, rank 1 → 0.5, rank 2 → 0.25, rank 3 → 0.125, rest 0
        scores = rewards_v7._score_ranks_sparse([0, 1, 2, 3, 4, 5, 6, 7], 8)
        assert scores == pytest.approx([1.0, 0.5, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0])

    def test_permuted_order_scores_by_id(self):
        # rollout id 3 ranked best, id 0 ranked second, ...
        scores = rewards_v7._score_ranks_sparse([3, 0, 5, 1, 7, 2, 4, 6], 8)
        assert scores[3] == pytest.approx(1.0)
        assert scores[0] == pytest.approx(0.5)
        assert scores[5] == pytest.approx(0.25)
        assert scores[1] == pytest.approx(0.125)
        # The four rollouts ranked 4-7 are all zero
        for rid in (7, 2, 4, 6):
            assert scores[rid] == 0.0

    def test_g4_default_top2(self):
        scores = rewards_v7._score_ranks_sparse([0, 1, 2, 3], 4)
        assert scores == pytest.approx([1.0, 0.5, 0.0, 0.0])

    def test_g1_returns_single_one(self):
        assert rewards_v7._score_ranks_sparse([0], 1) == [1.0]

    def test_custom_base(self):
        # base=0.25 → 1.0, 0.25, 0.0625, 0.015625, 0, ...
        scores = rewards_v7._score_ranks_sparse([0, 1, 2, 3, 4, 5, 6, 7], 8, base=0.25)
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.25)
        assert scores[2] == pytest.approx(0.0625)
        assert scores[3] == pytest.approx(0.015625)

    def test_custom_k(self):
        # k=1 = winner-take-all
        scores = rewards_v7._score_ranks_sparse([3, 0, 1, 2], 4, k=1)
        assert scores[3] == 1.0
        for rid in (0, 1, 2):
            assert scores[rid] == 0.0


class TestParseRankList:
    def test_valid_list(self):
        assert rewards_v7._parse_rank_list([3, 0, 5, 1, 7, 2, 4, 6], 8) == [3, 0, 5, 1, 7, 2, 4, 6]

    def test_v6_style_dict_with_order_key(self):
        assert rewards_v7._parse_rank_list({"order": [0, 1, 2, 3]}, 4) == [0, 1, 2, 3]

    def test_string_with_digits(self):
        # Pulled from a dict's "error" payload (recovery path)
        assert rewards_v7._parse_rank_list({"error": "garbled... 3 0 5 1 7 2 4 6 ..."}, 8) == [
            3, 0, 5, 1, 7, 2, 4, 6
        ]

    def test_wrong_length(self):
        assert rewards_v7._parse_rank_list([0, 1, 2], 4) is None

    def test_duplicate_ids(self):
        assert rewards_v7._parse_rank_list([0, 1, 1, 2], 4) is None

    def test_out_of_range_ids(self):
        assert rewards_v7._parse_rank_list([0, 1, 2, 9], 4) is None

    def test_non_list_non_dict(self):
        assert rewards_v7._parse_rank_list(42, 4) is None

    def test_empty_dict(self):
        assert rewards_v7._parse_rank_list({}, 4) is None


class TestSparseRankReward:
    def test_valid_ranking_gives_geometric(self):
        judge = V7StubJudge(reply=[3, 0, 5, 1, 7, 2, 4, 6])
        r = rewards_v7.SparseRankReward()
        outputs = [f"o{i}" for i in range(8)]
        scores = r.compute_group("src", outputs, judge=judge)
        assert scores[3] == pytest.approx(1.0)
        assert scores[0] == pytest.approx(0.5)
        assert scores[5] == pytest.approx(0.25)
        assert scores[1] == pytest.approx(0.125)
        # Bottom four all zero
        assert sum(1 for s in scores if s == 0.0) == 4

    def test_one_judge_call_per_group(self):
        judge = V7StubJudge(reply=[0, 1, 2, 3])
        r = rewards_v7.SparseRankReward()
        r.compute_group("src", ["a", "b", "c", "d"], judge=judge)
        assert len(judge.calls) == 1

    def test_malformed_falls_back_to_zero(self):
        judge = V7StubJudge(reply={"garbage": True})
        r = rewards_v7.SparseRankReward()
        scores = r.compute_group("src", ["a", "b", "c", "d"], judge=judge)
        assert scores == [0.0, 0.0, 0.0, 0.0]

    def test_single_rollout_short_circuits(self):
        judge = V7StubJudge(reply=[0])
        r = rewards_v7.SparseRankReward()
        assert r.compute_group("src", ["only"], judge=judge) == [1.0]
        assert judge.calls == []  # no call needed

    def test_no_judge_returns_all_zero(self):
        r = rewards_v7.SparseRankReward()
        assert r.compute_group("src", ["a", "b"], judge=None) == [0.0, 0.0]

    def test_judge_exception_returns_zero(self):
        class BoomJudge(BaseJudge):
            def evaluate(self, prompt):
                raise RuntimeError("transport boom")

        r = rewards_v7.SparseRankReward()
        assert r.compute_group("src", ["a", "b"], judge=BoomJudge()) == [0.0, 0.0]

    def test_prompt_includes_source_and_candidates(self):
        judge = V7StubJudge(reply=[0, 1])
        r = rewards_v7.SparseRankReward()
        r.compute_group("my source paragraph", ["candidate one", "candidate two"], judge=judge)
        assert "my source paragraph" in judge.calls[0]
        assert "[0] candidate one" in judge.calls[0]
        assert "[1] candidate two" in judge.calls[0]


class TestCombinedRewardV7:
    # 20-word source (same as v6 tests so we know length_factor is ~1.0)
    SRC = (
        "The library opened in 1934 and stood near the river for many years. "
        "Many people visited it during quiet afternoons throughout autumn."
    )
    GOOD_OUT = (
        "A library opened in 1934 and stood near the river for many years. "
        "Many readers visited it on quiet afternoons each autumn season."
    )

    def test_winner_gets_full_reward(self):
        # G=4, default k=2: rollout 0 ranked best, rollout 1 second
        judge = V7StubJudge(reply=[0, 1, 2, 3])
        c = rewards_v7.CombinedRewardV7()
        scores = c.compute_group(self.SRC, [self.GOOD_OUT] * 4, judge=judge)
        # rank 0 → 1.0, rank 1 → 0.5; length ≈ 1.0, gate = 1.0
        assert scores[0] == pytest.approx(1.0, abs=0.05)
        assert scores[1] == pytest.approx(0.5, abs=0.05)
        # Bottom half zero
        assert scores[2] == 0.0
        assert scores[3] == 0.0

    def test_markdown_gate_zeros_reward(self):
        # G=4 so the 2nd-ranked rollout also has a nonzero rank score
        judge = V7StubJudge(reply=[0, 1, 2, 3])
        c = rewards_v7.CombinedRewardV7()
        # Rollout 0 has markdown → gate=0, reward=0 even though ranked best
        outs = ["**bold** " + self.GOOD_OUT, self.GOOD_OUT, self.GOOD_OUT, self.GOOD_OUT]
        scores = c.compute_group(self.SRC, outs, judge=judge)
        assert scores[0] == 0.0
        assert scores[1] > 0.0  # rank 1 → 0.5 base

    def test_length_factor_attenuates(self):
        judge = V7StubJudge(reply=[0, 1, 2, 3])
        c = rewards_v7.CombinedRewardV7()
        # Rollout 0 is extremely short — length factor pulls it way down
        outs = ["hi", self.GOOD_OUT, self.GOOD_OUT, self.GOOD_OUT]
        scores = c.compute_group(self.SRC, outs, judge=judge)
        # Even ranked best, the length penalty crushes it
        assert scores[0] < 0.2

    def test_bottom_half_zero(self):
        judge = V7StubJudge(reply=[0, 1, 2, 3, 4, 5, 6, 7])
        c = rewards_v7.CombinedRewardV7()
        outs = [self.GOOD_OUT] * 8
        scores = c.compute_group(self.SRC, outs, judge=judge)
        # Top 4 nonzero, bottom 4 zero
        assert sum(1 for s in scores if s > 0) == 4
        assert sum(1 for s in scores if s == 0) == 4


class TestV7RegisteredFunction:
    def test_batch_two_group_two_runs_two_judge_calls(self, monkeypatch):
        judge = V7StubJudge(reply=[0, 1])
        monkeypatch.setattr(rewards_v7, "_get_judge", lambda: judge)
        out = rewards_v7.v7_sparse_rank_reward(
            prompts=["src1", "src1", "src2", "src2"],
            completions=["a", "b", "c", "d"],
            answer=["", "", "", ""],
            types=None,
        )
        assert len(out) == 4
        # 2 groups → 2 judge calls (no per-rollout calls, unlike v6)
        assert len(judge.calls) == 2

    def test_g8_single_group_one_call(self, monkeypatch):
        judge = V7StubJudge(reply=[0, 1, 2, 3, 4, 5, 6, 7])
        monkeypatch.setattr(rewards_v7, "_get_judge", lambda: judge)
        out = rewards_v7.v7_sparse_rank_reward(
            prompts=["src1"] * 8,
            completions=[f"o{i}" for i in range(8)],
            answer=[""] * 8,
            types=None,
        )
        assert len(out) == 8
        assert len(judge.calls) == 1

    def test_result_length_matches_completions(self, monkeypatch):
        judge = V7StubJudge(reply=[0, 1, 2])
        monkeypatch.setattr(rewards_v7, "_get_judge", lambda: judge)
        out = rewards_v7.v7_sparse_rank_reward(
            prompts=["s1", "s1", "s1", "s2", "s2", "s2"],
            completions=["a", "b", "c", "d", "e", "f"],
            answer=[""] * 6,
            types=None,
        )
        assert len(out) == 6
        assert all(isinstance(x, float) for x in out)

