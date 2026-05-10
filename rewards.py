"""GRPO reward components for language simplification.

Three active components in v1:
  * LengthVsSourceReward      — output/source word ratio in target band
  * VocabSimplicityReward     — penalty for too many uncommon words/sentence
  * SemanticPreservationReward — judge call comparing source vs output

CombinedReward aggregates them as a weighted sum, with a *meaning gate*:
if SemanticPreservation < gate threshold, the whole reward is zeroed.
This prevents the model from learning to game length/vocab while
silently shedding source content.

Two stubs (RepetitionReward, SmoothDifficultyReward) exist so the wiring
is in place; their numeric behavior is TODO.

The bottom of the file contains thin `@register_reward_function`-style
adapters that mlx_lm_lora.train can discover via --reward-functions-file.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from wordfreq import top_n_list

from verifier import BaseJudge, split_sentences

# Top-2000 most common English words. CEFR-A2 vocabulary roughly tracks
# the 1500-2000-most-common floor. Loaded once at import time.
COMMON_WORDS: frozenset[str] = frozenset(top_n_list("en", 2000))


@dataclass
class RewardContext:
    """Per-rollout context. Source = the complex paragraph the model is
    rewriting. Answer = optional reference simplification (Opus chosen)
    that some rewards may compare against; not used in v1."""
    source: str
    answer: Optional[str] = None


class RewardComponent(ABC):
    """Single reward component. Returns a float in [0, 1]."""
    name: str = "reward"

    @abstractmethod
    def compute(
        self, output: str, ctx: RewardContext, judge: Optional[BaseJudge] = None,
    ) -> float: ...


# ---------- LengthVsSourceReward ----------

class LengthVsSourceReward(RewardComponent):
    """Reward 1.0 when output_words/source_words is within [floor, ceiling],
    decaying linearly outside. Penalizes both excessive condensation
    (model dropped content) and excessive padding (model expanded source)."""
    name = "length"

    def __init__(
        self,
        soft_floor: float = 0.8,
        soft_ceiling: float = 1.3,
        hard_floor: float = 0.3,
        hard_ceiling: float = 2.5,
    ):
        self.soft_floor = soft_floor
        self.soft_ceiling = soft_ceiling
        self.hard_floor = hard_floor
        self.hard_ceiling = hard_ceiling

    def compute(self, output: str, ctx: RewardContext, judge=None) -> float:
        src_n = len(ctx.source.split())
        if src_n == 0:
            return 0.0
        ratio = len(output.split()) / src_n
        if self.soft_floor <= ratio <= self.soft_ceiling:
            return 1.0
        if ratio < self.soft_floor:
            if ratio <= self.hard_floor:
                return 0.0
            return (ratio - self.hard_floor) / (self.soft_floor - self.hard_floor)
        # ratio > soft_ceiling
        if ratio >= self.hard_ceiling:
            return 0.0
        return 1.0 - (ratio - self.soft_ceiling) / (self.hard_ceiling - self.soft_ceiling)


# ---------- VocabSimplicityReward ----------

_PROPER_NOUN_RE = re.compile(r"^[A-Z][a-z]+$")


def _tokenize_words(sentence: str) -> list[str]:
    """Split into word-only tokens (drops punctuation)."""
    return re.findall(r"[A-Za-z'']+", sentence)


def _is_likely_proper_noun(word: str, position_in_sentence: int) -> bool:
    """Heuristic: capitalized AND not the first word of the sentence."""
    return position_in_sentence > 0 and bool(_PROPER_NOUN_RE.match(word))


class VocabSimplicityReward(RewardComponent):
    """Per-sentence: count words not in COMMON_WORDS (skipping proper nouns).
    Penalty kicks in once that count exceeds `allowed_uncommon` per sentence.
    Reward = 1 - (mean penalty across sentences), clipped to [0, 1]."""
    name = "vocab"

    def __init__(self, allowed_uncommon: int = 2, severity: float = 0.25):
        self.allowed_uncommon = allowed_uncommon
        self.severity = severity  # how much each excess uncommon word costs

    def compute(self, output: str, ctx: RewardContext, judge=None) -> float:
        sentences = split_sentences(output)
        if not sentences:
            return 1.0
        penalties: list[float] = []
        for s in sentences:
            tokens = _tokenize_words(s)
            uncommon = 0
            for i, tok in enumerate(tokens):
                if _is_likely_proper_noun(tok, i):
                    continue
                if tok.lower() not in COMMON_WORDS:
                    uncommon += 1
            excess = max(0, uncommon - self.allowed_uncommon)
            penalties.append(min(1.0, excess * self.severity))
        return max(0.0, 1.0 - sum(penalties) / len(penalties))


# ---------- SemanticPreservationReward ----------

_MEANING_PROMPT_TEMPLATE = """You will compare a SOURCE text and a SIMPLIFICATION of it. Rate the simplification on two axes, each 1-5.

1. **facts_preserved** — Did the simplification keep the important facts of the source? 5 = all important facts present; 1 = most important facts dropped.

2. **no_hallucinations** — Did the simplification avoid adding facts NOT in the source? 5 = nothing invented; 1 = many facts invented.

Note: dropping minor / decorative detail is fine and should NOT lower facts_preserved. Only score down when *important* facts are missing.

SOURCE:
{source}

SIMPLIFICATION:
{output}

Respond with ONLY a JSON object, no prose, no markdown fences:
{{"facts_preserved": int, "no_hallucinations": int}}"""


class SemanticPreservationReward(RewardComponent):
    """Asks a judge whether the simplification preserves source meaning
    without hallucinating. Average of the two scores, normalized to [0, 1]."""
    name = "meaning"

    def compute(self, output: str, ctx: RewardContext, judge: Optional[BaseJudge] = None) -> float:
        if judge is None:
            return 0.5  # no judge → unknown; mid score so we don't crash GRPO
        prompt = _MEANING_PROMPT_TEMPLATE.format(source=ctx.source, output=output)
        try:
            result = judge.evaluate(prompt)
        except Exception:
            return 0.5
        try:
            facts = float(result.get("facts_preserved", 0))
            halluc = float(result.get("no_hallucinations", 0))
        except (TypeError, ValueError):
            return 0.5
        # Both axes contribute equally; convert from 1-5 → 0-1
        # (a score of 1 is the worst, 5 the best, so subtract 1 and /4)
        score = ((facts - 1) / 4 + (halluc - 1) / 4) / 2
        return max(0.0, min(1.0, score))


# ---------- Stubs (TODO) ----------

class RepetitionReward(RewardComponent):
    """TODO: penalize repetitive outputs (low unique-words/total ratio).
    Stubbed at 1.0 for v1 — included so the wiring works when we activate it."""
    name = "repetition"

    def compute(self, output: str, ctx: RewardContext, judge=None) -> float:
        return 1.0  # TODO: implement actual repetition detection


class SmoothDifficultyReward(RewardComponent):
    """TODO: judge-based CEFR level → smooth score (A2=1.0, A1=0.6, B1=0.4, ...).
    Stubbed at 1.0 for v1. Will reuse few-shot samples like
    verifier.DifficultyRankingTest does."""
    name = "difficulty"

    def __init__(self, a1_samples=None, a2_samples=None, b1_samples=None):
        self.a1_samples = a1_samples or []
        self.a2_samples = a2_samples or []
        self.b1_samples = b1_samples or []

    def compute(self, output: str, ctx: RewardContext, judge=None) -> float:
        return 1.0  # TODO: judge call + smooth A2/A1/B1/B2+/<A1 → score


# ---------- CombinedReward ----------

class CombinedReward(RewardComponent):
    """Weighted sum of components. If a 'meaning' component is present and
    scores below `meaning_gate`, the whole reward is zeroed.

    The meaning gate is the safety belt against the model gaming the
    cheaper rewards (length, vocab) by silently dropping content."""
    name = "combined"

    def __init__(
        self,
        components: list[tuple[float, RewardComponent]],
        meaning_gate: float = 0.5,
    ):
        self.components = components
        self.meaning_gate = meaning_gate

    def compute(self, output: str, ctx: RewardContext, judge=None) -> float:
        scores: dict[str, float] = {}
        total = 0.0
        for w, comp in self.components:
            s = comp.compute(output, ctx, judge=judge)
            scores[comp.name] = s
            total += w * s
        meaning = scores.get("meaning")
        if meaning is not None and meaning < self.meaning_gate:
            return 0.0
        return max(0.0, min(1.0, total))


# ---------- mlx_lm_lora @register_reward_function adapters ----------
#
# Defined as plain functions matching the (prompts, completions, **kwargs)
# signature the framework calls. Each wraps the corresponding
# RewardComponent with a per-record loop so we always return one float
# per (prompt, completion) pair. The wrappers will be activated by the
# `--reward-functions-file rewards.py` flag in `train.py grpo`.
#
# The actual @register_reward_function decorator is added in train.py
# import context to keep this module importable without mlx_lm_lora.

_LENGTH = LengthVsSourceReward()
_VOCAB = VocabSimplicityReward()
_MEANING = SemanticPreservationReward()


def length_reward_fn(prompts, completions, **_) -> list[float]:
    return [
        _LENGTH.compute(c, RewardContext(source=p))
        for p, c in zip(prompts, completions)
    ]


def vocab_reward_fn(prompts, completions, **_) -> list[float]:
    return [
        _VOCAB.compute(c, RewardContext(source=p))
        for p, c in zip(prompts, completions)
    ]


def meaning_reward_fn(prompts, completions, judge=None, **_) -> list[float]:
    return [
        _MEANING.compute(c, RewardContext(source=p), judge=judge)
        for p, c in zip(prompts, completions)
    ]
