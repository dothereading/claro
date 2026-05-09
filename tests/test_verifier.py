"""Tests for verifier.py — judges, tests, and reward aggregation.

The judge is mocked so no LM Studio process is required.
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from verifier import (
    BaseJudge,
    DifficultyRankingTest,
    LocalJudge,
    RewardVerifier,
    _clean,
    truncate_to_words,
    windowed_excerpts,
)


class StubJudge(BaseJudge):
    """In-memory judge that returns a pre-set response."""
    def __init__(self, response: Dict[str, Any]):
        self.response = response
        self.calls: list[str] = []

    def evaluate(self, prompt: str) -> Dict[str, Any]:
        self.calls.append(prompt)
        return self.response


class TestClean:
    def test_strips_urls(self):
        assert _clean("see https://example.com/foo for more") == "see  for more"

    def test_strips_bracketed_refs(self):
        assert _clean("citation [1] needed [42] here") == "citation  needed  here"


class TestTruncateToWords:
    def test_short_text_returned_whole(self):
        assert truncate_to_words("a b c", n_words=10) == "a b c"

    def test_long_text_truncated_to_n_words(self):
        text = " ".join(str(i) for i in range(50))
        out = truncate_to_words(text, n_words=10)
        assert out.split() == [str(i) for i in range(10)]

    def test_cleans_before_truncation(self):
        text = "see https://example.com/x for context [1] one two three four five"
        out = truncate_to_words(text, n_words=4)
        # URL and ref removed by _clean before split
        assert out == "see for context one"


class TestWindowedExcerpts:
    def test_single_window_when_short(self):
        assert windowed_excerpts("a b c", n_words=10) == ["a b c"]

    def test_three_evenly_spaced_windows(self):
        text = " ".join(str(i) for i in range(30))
        out = windowed_excerpts(text, n_words=10, n_windows=3)
        assert len(out) == 3
        assert out[0].split() == [str(i) for i in range(10)]
        assert out[-1].split() == [str(i) for i in range(20, 30)]

    def test_dedups_when_text_too_short_for_distinct_windows(self):
        text = " ".join(str(i) for i in range(11))
        out = windowed_excerpts(text, n_words=10, n_windows=3)
        # Only two distinct starting positions exist (0 and 1)
        assert len(out) <= 3
        assert len(set(out)) == len(out)


class TestLocalJudgeJsonParse:
    def setup_method(self):
        self.j = LocalJudge(base_url="http://x", model_name="m")

    def test_parses_plain_json(self):
        assert self.j._parse_json('{"level": "A2"}') == {"level": "A2"}

    def test_parses_json_in_fenced_block(self):
        raw = 'preamble\n```json\n{"level": "A1"}\n```\ntrailing'
        assert self.j._parse_json(raw) == {"level": "A1"}

    def test_parses_json_in_unlabeled_fence(self):
        raw = '```\n{"level": "B1"}\n```'
        assert self.j._parse_json(raw) == {"level": "B1"}

    def test_falls_back_to_first_brace_block(self):
        raw = 'noise before {"level": "A2", "x": 1} noise after'
        assert self.j._parse_json(raw) == {"level": "A2", "x": 1}


class TestDifficultyRanking:
    def _test_obj(self, **kwargs):
        return DifficultyRankingTest(
            a1_samples=["short"], b1_samples=["long"], a2_samples=["med"], **kwargs,
        )

    def test_returns_1_when_judge_says_a2(self):
        t = self._test_obj()
        score = t.run("some text", StubJudge({"level": "A2"}))
        assert score == 1.0

    def test_returns_0_when_judge_says_b1(self):
        t = self._test_obj()
        assert t.run("some text", StubJudge({"level": "B1"})) == 0.0

    def test_normalizes_b2_variants_to_b2plus(self):
        t = self._test_obj()
        for label in ["B2", "C1", "C2"]:
            assert t.run("x", StubJudge({"level": label})) == 0.0

    def test_normalizes_below_a1_variants(self):
        t = self._test_obj()
        for label in ["BELOW A1", "PRE-A1", "<<A1"]:
            assert t.run("x", StubJudge({"level": label})) == 0.0

    def test_returns_0_for_unrecognized_level(self):
        t = self._test_obj()
        assert t.run("x", StubJudge({"level": "QQQ"})) == 0.0

    def test_prompt_includes_candidate_text(self):
        t = self._test_obj()
        judge = StubJudge({"level": "A2"})
        t.run("CANDIDATE_MARKER text here", judge)
        assert "CANDIDATE_MARKER" in judge.calls[0]


class TestRewardVerifier:
    def test_no_tests_returns_zero(self):
        v = RewardVerifier(judge=StubJudge({}))
        assert v.verify("anything") == 0.0

    def test_averages_test_scores(self):
        class FixedScore:
            def __init__(self, s): self.s = s
            def run(self, text, judge): return self.s

        v = RewardVerifier(judge=StubJudge({}))
        v.add_test(FixedScore(1.0))
        v.add_test(FixedScore(0.0))
        assert v.verify("x") == 0.5
