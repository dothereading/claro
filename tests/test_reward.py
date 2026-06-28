"""Unit tests for the v10 cardinal reward (offline components 1, 2, 4).

Component 3 (fidelity / judge) lives in TestFidelity at the bottom and uses
a mocked judge — no network. The band and vocab tests use the real shared
spaCy pipeline (en_core_web_sm), so they exercise the actual lemmatizer /
tagger that scoring uses.
"""

from __future__ import annotations

import json

import pytest

level_band = pytest.importorskip("claro.reward.c1_level_band")
vocab = pytest.importorskip("claro.reward.c2_vocab")
gates = pytest.importorskip("claro.reward.c4_gates")
fidelity = pytest.importorskip("claro.reward.c3_fidelity")
compose = pytest.importorskip("claro.reward.compose")

from claro.verifier import BaseJudge  # noqa: E402


class StubJudge(BaseJudge):
    """Returns a queue of preset replies; counts calls. A reply may be a dict
    (already-parsed verdict) — mirrors LocalJudge.evaluate's contract."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def evaluate(self, prompt: str) -> dict:
        self.calls += 1
        return self.replies.pop(0) if self.replies else self.replies_default()

    def replies_default(self):
        return {"error": "exhausted"}


def _scorer(replies, **kw):
    return fidelity.FidelityScorer(
        judge=StubJudge(replies),
        model_id="test-model",
        prompt_template="SOURCE:\n{source}\n\nCANDIDATE:\n{candidate}\n",
        prompt_version="test",
        **kw,
    )


# ---------- Component 1: trapezoid ----------


class TestTrapezoid:
    # q1=10, q3=20 -> width 10, margin 0.5*10 = 5
    def test_inside_band_is_full(self):
        assert level_band.trapezoid(15, 10, 20) == 1.0

    def test_boundaries_are_full(self):
        assert level_band.trapezoid(10, 10, 20) == 1.0
        assert level_band.trapezoid(20, 10, 20) == 1.0

    def test_linear_falloff_below(self):
        # x = q1 - margin/2 = 10 - 2.5 = 7.5 -> 1 - 2.5/5 = 0.5
        assert level_band.trapezoid(7.5, 10, 20) == pytest.approx(0.5)

    def test_linear_falloff_above(self):
        # x = q3 + margin/2 = 22.5 -> 0.5
        assert level_band.trapezoid(22.5, 10, 20) == pytest.approx(0.5)

    def test_floor_far_below(self):
        assert level_band.trapezoid(-100, 10, 20) == pytest.approx(0.2)

    def test_floor_far_above(self):
        assert level_band.trapezoid(1000, 10, 20) == pytest.approx(0.2)

    def test_never_below_floor(self):
        # Just past the linear zero crossing, clamps to floor not negative.
        assert level_band.trapezoid(5, 10, 20) == pytest.approx(0.2)


# ---------- Component 1: FRE / MSL extraction ----------


class TestReadabilityMetrics:
    def test_mean_sentence_length_counts_words(self):
        # Two sentences, two words each (punctuation excluded).
        assert level_band.mean_sentence_length("I run. You jump.") == pytest.approx(2.0)

    def test_flesch_returns_float(self):
        score = level_band.flesch_reading_ease("The cat sat on the mat. It was a good day.")
        assert isinstance(score, float)


# ---------- Component 1 (v11.1): syntactic features ----------


class TestSyntacticFeatures:
    def test_passive_detected(self):
        p, _, _ = level_band.syntactic_features(
            "The house was built in 1750. It was damaged by soldiers.")
        assert p == pytest.approx(1.0)  # one passive construction per sentence

    def test_active_voice_has_no_passive(self):
        p, _, _ = level_band.syntactic_features("The dog runs fast. The boy plays outside.")
        assert p == pytest.approx(0.0)

    def test_subordination_detected(self):
        _, s, _ = level_band.syntactic_features("She won the prize because she worked very hard.")
        assert s >= 1.0

    def test_simple_sentences_low_subordination(self):
        _, s, _ = level_band.syntactic_features("The dog is brown. It runs. The boy is happy.")
        assert s == pytest.approx(0.0)

    def test_appositive_detected(self):
        _, _, a = level_band.syntactic_features(
            "The Taj Mahal, a famous building, stands in India.")
        assert a >= 1.0

    def test_packed_b1_scores_below_simple_a2(self):
        # The measured mechanism: a passive/subordinated rewrite must score
        # lower than a plain one, holding a wide FRE/MSL band fixed.
        band = level_band.Band(fre_q1=10.0, fre_q3=130.0, msl_q1=3.0, msl_q3=20.0,
                               pass_q1=0.0, pass_q3=0.4, sub_q1=0.0, sub_q3=0.6)
        a2 = "The house is old. People built it long ago. Soldiers later broke it."
        b1 = ("The old house, which was built long ago, was later damaged by soldiers "
              "who were sent to the area.")
        assert level_band.level_band(b1, band) < level_band.level_band(a2, band)


# ---------- Component 1: level_band scoring ----------


class TestLevelBand:
    def _band(self):
        # Wide bands chosen so a simple multi-sentence text lands inside
        # both metrics (very short sentences push FRE above 100).
        return level_band.Band(fre_q1=10.0, fre_q3=130.0, msl_q1=3.0, msl_q3=20.0,
                               pass_q1=0.0, pass_q3=2.0, sub_q1=0.0, sub_q3=3.0)

    def test_degenerate_too_few_sentences(self):
        # One sentence -> floor*floor = 0.04
        assert level_band.level_band("Just one short clean sentence here.", self._band()) == pytest.approx(0.04)

    def test_degenerate_too_few_words(self):
        assert level_band.level_band("Short. Text.", self._band()) == pytest.approx(0.04)

    def test_in_band_scores_high(self):
        text = (
            "The dog is brown. It likes to run in the park. "
            "Every day the boy takes it out. They play with a ball."
        )
        assert level_band.level_band(text, self._band()) == pytest.approx(1.0)


# ---------- Component 2: sentence_score math ----------


class TestSentenceScore:
    def test_zero_or_one_hard_is_full(self):
        assert vocab.sentence_score(0) == 1.0
        assert vocab.sentence_score(1) == 1.0

    def test_geometric_penalty(self):
        assert vocab.sentence_score(2) == pytest.approx(0.5)
        assert vocab.sentence_score(3) == pytest.approx(0.25)


# ---------- Component 2: vocab exemptions ----------


class TestVocabExemptions:
    # A deliberately tiny vocab so only the word under test is ever off-list.
    BASE = frozenset({"be", "a", "the", "he", "she", "it", "have", "write",
                      "play", "cat", "run", "and", "of", "to", "in", "was",
                      "were", "is", "are", "rich", "person", "kind", "group",
                      "people", "with", "use", "data", "help", "doctor", "that",
                      "check", "study", "old", "bird"})

    def test_proper_noun_exempt(self):
        # "Shakespeare" is off-list but PROPN -> not hard.
        score, flagged, _ = vocab.vocab_term("", "Shakespeare wrote plays.", vocab=self.BASE | {"wrote"})
        assert score == pytest.approx(1.0)
        assert flagged == [[]]

    def test_number_exempt(self):
        score, _, _ = vocab.vocab_term("", "He has 1000 cats.", vocab=self.BASE | {"has"})
        assert score == pytest.approx(1.0)

    def test_repeated_hard_word_counted_twice(self):
        # Two occurrences of an off-list non-exempt word -> hard_count 2 -> 0.5.
        score, flagged, _ = vocab.vocab_term("", "The mogul met a mogul.", vocab=self.BASE | {"meet"})
        assert score == pytest.approx(0.5)
        assert flagged[0].count("mogul") == 2

    def test_floor_applied(self):
        # Many hard words in one sentence -> product would be tiny -> floor 0.2.
        text = "The zorble florbed the quaxil with a snicket and a brizzle."
        score, _, _ = vocab.vocab_term("", text, vocab=self.BASE)
        assert score == pytest.approx(0.2)


# ---------- Component 2 (v11): density penalty — source jargon is no longer free ----------


class TestVocabDensityPenalty(TestVocabExemptions):
    def test_bare_source_term_is_counted(self):
        # v11: an off-list term from the source, kept BARE, now counts hard
        # (old behavior exempted it unconditionally). Single hard -> still 1.0,
        # but it must be flagged as bare_source.
        score, flagged, dbg = vocab.vocab_term(
            "She married a mogul.", "He was a mogul.", vocab=self.BASE)
        assert flagged == [["mogul"]]
        assert dbg.bare_source == ["mogul"]
        assert dbg.glossed == []

    def test_two_bare_source_terms_penalized(self):
        # Two retained bare source terms in one sentence -> hard 2 -> 0.5.
        score, flagged, dbg = vocab.vocab_term(
            "The mogul met the tycoon.", "The mogul met the tycoon.", vocab=self.BASE | {"meet"})
        assert score == pytest.approx(0.5)
        assert set(dbg.bare_source) == {"mogul", "tycoon"}

    def test_copula_gloss_exempts_first_occurrence(self):
        # "A mogul is a rich person" glosses mogul -> exempt; 0 hard -> 1.0.
        score, flagged, dbg = vocab.vocab_term(
            "She married a mogul.", "A mogul is a rich person.", vocab=self.BASE)
        assert score == pytest.approx(1.0)
        assert "mogul" in [w.lower() for w in dbg.glossed]
        assert dbg.bare_source == []

    def test_appositive_gloss_exempts(self):
        score, flagged, dbg = vocab.vocab_term(
            "She married a mogul.", "The mogul, a rich person, was kind.", vocab=self.BASE)
        assert "mogul" in [w.lower() for w in dbg.glossed]

    def test_gloss_with_hard_definition_does_not_count(self):
        # Definition itself contains an off-list word ("tycoon") -> not a clean
        # gloss -> mogul is NOT exempted, counts bare.
        score, flagged, dbg = vocab.vocab_term(
            "She married a mogul.", "A mogul is a tycoon.", vocab=self.BASE)
        assert dbg.glossed == []
        assert "mogul" in dbg.bare_source

    def test_gloss_licenses_first_occurrence_only(self):
        # Glossed once, then reused bare -> the reuse counts.
        score, flagged, dbg = vocab.vocab_term(
            "She married a mogul.",
            "A mogul is a rich person. The mogul was kind.", vocab=self.BASE)
        assert "mogul" in [w.lower() for w in dbg.glossed]
        assert "mogul" in [w.lower() for w in dbg.bare_source]

    def test_invented_offlist_word_counts_even_if_glossed(self):
        # A term NOT in the source is "invented" for vocab purposes and counts
        # hard regardless of gloss (only source terms can be licensed).
        score, flagged, dbg = vocab.vocab_term(
            "", "A mogul is a rich person.", vocab=self.BASE)
        assert "mogul" in dbg.invented
        assert dbg.glossed == []

    def test_retained_jargon_scores_below_simplified_away(self):
        # The exact v10 failure, reversed: keeping the bare source term must
        # score strictly lower than dropping it.
        src = "The mogul ran the tycoon empire firm."
        keep = vocab.vocab_term(src, "The mogul ran the tycoon firm.", vocab=self.BASE)[0]
        drop = vocab.vocab_term(src, "He ran the big company.", vocab=self.BASE | {"big", "company"})[0]
        assert keep < drop


# ---------- Component 4: format gates ----------


class TestGates:
    def test_clean_prose_passes(self):
        assert gates.format_gates("The dog is brown. It likes to run.") == 1.0

    def test_markdown_fails(self):
        assert gates.format_gates("**Here** is the rewrite.") == 0.0

    def test_loop_fails(self):
        looped = "the cat is here " * 12
        assert gates.format_gates(looped) == 0.0


# ---------- Component 3: fidelity scoring math (no network) ----------


class TestFidelityScoring:
    def test_perfect(self):
        v = {"source_facts": [{"fact": "a", "status": "present"},
                              {"fact": "b", "status": "present"}],
             "unsupported_claims": []}
        term, recall_term, halluc, recall, n = fidelity.score_verdict(v)
        assert recall == pytest.approx(1.0)
        assert halluc == pytest.approx(1.0)
        assert term == pytest.approx(1.0)

    def test_partial_recall(self):
        v = {"source_facts": [{"fact": "a", "status": "present"},
                              {"fact": "b", "status": "present"},
                              {"fact": "c", "status": "present"},
                              {"fact": "d", "status": "absent"}],
             "unsupported_claims": []}
        term, recall_term, _, recall, _ = fidelity.score_verdict(v)
        assert recall == pytest.approx(0.75)
        assert term == pytest.approx(0.75)

    def test_recall_floor(self):
        v = {"source_facts": [{"fact": "a", "status": "absent"}],
             "unsupported_claims": []}
        _, recall_term, _, recall, _ = fidelity.score_verdict(v)
        assert recall == pytest.approx(0.0)
        assert recall_term == pytest.approx(0.2)

    def test_no_facts_full_recall(self):
        v = {"source_facts": [], "unsupported_claims": []}
        _, recall_term, _, recall, _ = fidelity.score_verdict(v)
        assert recall == pytest.approx(1.0)
        assert recall_term == pytest.approx(1.0)

    def test_one_unsupported(self):
        import math
        v = {"source_facts": [{"fact": "a", "status": "present"}],
             "unsupported_claims": ["invented"]}
        term, _, halluc, _, n = fidelity.score_verdict(v)
        assert n == 1
        assert halluc == pytest.approx(math.exp(-fidelity._HALLUC_ALPHA))  # ~0.30
        assert term == pytest.approx(math.exp(-fidelity._HALLUC_ALPHA))

    def test_recall_over_core_only(self):
        # 2 core (1 present), 3 peripheral (all absent). Recall = 1/2 = 0.5,
        # peripheral drops are free.
        v = {"source_facts": [
                {"fact": "a", "tier": "core", "status": "present"},
                {"fact": "b", "tier": "core", "status": "absent"},
                {"fact": "c", "tier": "peripheral", "status": "absent"},
                {"fact": "d", "tier": "peripheral", "status": "absent"},
                {"fact": "e", "tier": "peripheral", "status": "absent"}],
             "unsupported_claims": []}
        _, recall_term, _, recall, _ = fidelity.score_verdict(v)
        assert recall == pytest.approx(0.5)

    def test_dropping_only_peripheral_is_full_recall(self):
        v = {"source_facts": [
                {"fact": "a", "tier": "core", "status": "present"},
                {"fact": "b", "tier": "peripheral", "status": "absent"}],
             "unsupported_claims": []}
        _, recall_term, _, recall, _ = fidelity.score_verdict(v)
        assert recall == pytest.approx(1.0)

    def test_no_core_facts_full_recall(self):
        v = {"source_facts": [{"fact": "a", "tier": "peripheral", "status": "absent"}],
             "unsupported_claims": []}
        _, recall_term, _, recall, _ = fidelity.score_verdict(v)
        assert recall == pytest.approx(1.0)
        assert fidelity.has_core_facts(v) is False

    def test_missing_tier_treated_as_core(self):
        # Backward-compat / conservative: untiered facts count toward recall.
        v = {"source_facts": [{"fact": "a", "status": "absent"}],
             "unsupported_claims": []}
        _, _, _, recall, _ = fidelity.score_verdict(v)
        assert recall == pytest.approx(0.0)

    def test_unsupported_penalty_is_uncapped_and_monotonic(self):
        # The whole point of the reshape: n=3 and n=5 must differ (old cap made
        # them identical), so the hardest groups keep within-group gradient.
        def halluc(n):
            v = {"source_facts": [{"fact": "a", "status": "present"}],
                 "unsupported_claims": ["x"] * n}
            return fidelity.score_verdict(v)[2]
        assert halluc(3) > halluc(5) > 0.0
        assert halluc(1) > halluc(2) > halluc(3)


# ---------- Component 3: judge call, retry, fail-open, cache ----------


class TestFidelityScorer:
    VALID = {"source_facts": [{"fact": "a", "status": "present"}], "unsupported_claims": []}

    def test_valid_verdict_scored(self):
        s = _scorer([self.VALID])
        r = s.score("src", "cand")
        assert r.failed is False
        assert r.term == pytest.approx(1.0)
        assert s.judge.calls == 1

    def test_retry_then_success(self):
        # First reply is a parse failure (LocalJudge-style error dict), retry succeeds.
        s = _scorer([{"error": "bad json"}, self.VALID])
        r = s.score("src", "cand")
        assert r.failed is False
        assert s.judge.calls == 2

    def test_fail_open_after_two_failures(self):
        s = _scorer([{"error": "bad"}, {"error": "still bad"}])
        r = s.score("src", "cand")
        assert r.failed is True
        assert r.term == 1.0  # fail OPEN — a missing verdict must not punish
        assert s.failures == 1
        assert s.judge.calls == 2

    def test_cache_hit_skips_second_call(self):
        s = _scorer([self.VALID], cache=fidelity.JudgeCache(":memory:"))
        s.score("src", "cand")
        # Second call: no replies left, but cache should answer without calling.
        r2 = s.score("src", "cand")
        assert r2.from_cache is True
        assert s.judge.calls == 1
        assert s.cache_hits == 1

    def test_build_scorer_wires_yaml_prompt(self):
        # build_scorer sources its template from the `fidelity_judge` prompt in
        # prompts.yaml (no more standalone .txt), and its config from reward.yaml.
        from claro.prompts import FIDELITY_JUDGE_PROMPT

        s = fidelity.build_scorer(judge=StubJudge([]), use_cache=False)
        assert s.prompt_template == FIDELITY_JUDGE_PROMPT
        assert "{source}" in s.prompt_template and "{candidate}" in s.prompt_template
        assert s.prompt_version == fidelity.load_fidelity_config()["prompt_version"]


# ---------- Composition: gated rollout short-circuits the judge ----------


class TestCompose:
    def _band(self):
        return level_band.Band(fre_q1=10.0, fre_q3=130.0, msl_q1=3.0, msl_q3=20.0,
                               pass_q1=0.0, pass_q3=2.0, sub_q1=0.0, sub_q3=3.0)

    def test_gated_skips_judge(self):
        s = _scorer([])  # no replies; must never be called
        res = compose.reward("source text here", "**markdown** is not allowed.",
                             band=self._band(), scorer=s)
        assert res.total == 0.0
        assert res.skipped_judge is True
        assert s.judge.calls == 0

    def test_no_fidelity_arm_b(self):
        text = ("The dog is brown. It likes to run in the park. "
                "Every day the boy takes it out. They play with a ball.")
        res = compose.reward("The brown canine enjoys the park.", text,
                             band=self._band(), use_fidelity=False)
        assert res.components["fidelity"] == 1.0
        assert res.total == pytest.approx(res.components["level_band"] * res.components["vocab"])

    def test_full_composition_multiplies(self):
        s = _scorer([{"source_facts": [{"fact": "a", "status": "present"}],
                      "unsupported_claims": ["invented fact"]}])
        text = ("The dog is brown. It likes to run in the park. "
                "Every day the boy takes it out. They play with a ball.")
        res = compose.reward("A brown dog.", text, band=self._band(), scorer=s)
        c = res.components
        import math
        assert res.total == pytest.approx(c["level_band"] * c["vocab"] * c["fidelity"])
        # one unsupported claim, recall 1.0 -> fidelity = exp(-alpha)
        assert c["fidelity"] == pytest.approx(math.exp(-fidelity._HALLUC_ALPHA))


# ---------- Trainer adapter (Arm A / Arm B) ----------


class _StubScorer:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = 0
        self.failures = 0
        self.cache_hits = 0

    def score(self, source, candidate):
        self.calls += 1
        term, recall_term, halluc, recall, n = fidelity.score_verdict(self.verdict)
        return fidelity.FidelityResult(term, recall_term, halluc, recall, n,
                                       self.verdict, failed=False, from_cache=False)


class TestTrainerAdapter:
    SRC = "A brown dog runs in the park every day with the boy."
    OUTS = [
        "The dog is brown. It likes to run in the park. The boy takes it out. They play.",
        "The dog is brown. It runs in the park. Every day the boy goes too. They have fun.",
    ]

    def test_shipped_reward_runs_offline(self, tmp_path, monkeypatch):
        rewards = pytest.importorskip("claro.training.rewards")
        stub = _StubScorer({"source_facts": [{"fact": "a", "status": "present"}],
                            "unsupported_claims": []})
        monkeypatch.setattr(rewards, "_run_dir", lambda: tmp_path)
        monkeypatch.setattr(rewards, "default_scorer", lambda: stub)
        out = rewards.cefr_a2_reward([self.SRC] * 2, self.OUTS, [None, None])
        assert len(out) == 2
        assert all(0.0 <= r <= 1.0 for r in out)
        metrics = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
        assert metrics  # at least one iteration summary written
        summary = json.loads(metrics[-1])
        assert summary["n"] == 2
        assert "fidelity_mean" in summary

    def test_shipped_reward_calls_scorer_per_rollout(self, tmp_path, monkeypatch):
        rewards = pytest.importorskip("claro.training.rewards")
        stub = _StubScorer({"source_facts": [{"fact": "a", "status": "present"}],
                            "unsupported_claims": []})
        monkeypatch.setattr(rewards, "_run_dir", lambda: tmp_path)
        monkeypatch.setattr(rewards, "default_scorer", lambda: stub)
        rewards.cefr_a2_reward([self.SRC] * 2, self.OUTS, [None, None])
        assert stub.calls == 2  # one judge call per rollout
        summary = json.loads((tmp_path / "metrics.jsonl").read_text().strip().splitlines()[-1])
        assert "halluc_flag_rate" in summary and "judge_failures" in summary
