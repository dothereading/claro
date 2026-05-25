"""GRPO reward components for language simplification.

Six active components in v3, weighted by importance (sum = 1.0):

  Pure-Python (no judge):
    * LengthVsSourceReward    — output/source word ratio in [0.8, 1.3]      w=0.05
    * VocabSimplicityReward   — uncommon-word penalty per sentence          w=0.10
                                (top-3000 wordfreq, calibrated against
                                 chosen/rejected/bad in data/{sft,dpo}.jsonl)
    * RepetitionReward        — distinct-4-gram ratio + repeated-sentence   w=0.15
                                detection (catches the GRPO loop failure)
    * NoMarkdownReward        — binary: 0 if any markdown marker present    w=0.05

  Judge-backed (share one combined HTTP call per rollout):
    * SemanticPreservationReward — facts kept + no hallucinations (`f`, `h`)  w=0.40
    * SmoothDifficultyReward     — CEFR level → smooth score (`lvl`)           w=0.25

The two judge rewards read from a shared `_judge_bundle()` cache keyed
on (source, output), so adding difficulty costs zero extra round trips
vs. meaning alone. Benchmarked: combined call is 2.1× faster than two
independent calls; anchored prompt (samples.jsonl A1/A2/B1 examples)
brings independent/combined CEFR agreement from 0/9 to 6/9.

Weighting rationale: meaning (0.40) dominates because a hallucinated
output is worse than no simplification — faithfulness is the
non-negotiable axis. Difficulty (0.25) is the explicit task target.
Repetition (0.15) is a guardrail with hard penalty when it fires.
Vocab/length/markdown (0.10/0.05/0.05) are finer guardrails; mostly
pinned at 1.0 in normal training, contribute zero advantage gradient
when the policy is well-behaved.

CombinedReward aggregates with a *meaning gate*: if the meaning score
is below the gate threshold (0.3 — intentionally low to avoid starving
early training), the entire reward is zeroed. Hard floor against
catastrophic faithfulness failures; soft enough that mediocre-but-
recoverable rollouts can contribute gradient through the weighted sum.

The bottom of the file contains thin `@register_reward_function`-style
adapters that mlx_lm_lora.train discovers via --reward-functions-file.
Register order matters: meaning_reward should come before
difficulty_reward so the judge cache is warm when difficulty reads it.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from wordfreq import top_n_list

from langsimp.verifier import BaseJudge, split_sentences

# Default common-word list size. Top-3000 strikes the right balance: the
# bare top-2000 misses common A2 concrete vocabulary (e.g. "tower",
# "destroyed", "stands"), while top-5000+ stops penalizing real B1+
# academic prose. Calibrated against data/{sft,dpo}.jsonl — see
# TestVocabSimplicityCalibration. Cached per-N at module level so the
# wordfreq import cost is paid once.
_DEFAULT_TOP_N = 3000
_COMMON_WORDS_CACHE: dict[int, frozenset[str]] = {}


def _common_words(top_n: int) -> frozenset[str]:
    if top_n not in _COMMON_WORDS_CACHE:
        _COMMON_WORDS_CACHE[top_n] = frozenset(top_n_list("en", top_n))
    return _COMMON_WORDS_CACHE[top_n]


@dataclass
class RewardContext:
    """Per-rollout context. Source = the complex paragraph the model is
    rewriting. Answer = optional reference simplification (Opus chosen)
    that some rewards may compare against; not used in v1."""

    source: str
    answer: str | None = None


class RewardComponent(ABC):
    """Single reward component. Returns a float in [0, 1]."""

    name: str = "reward"

    @abstractmethod
    def compute(
        self,
        output: str,
        ctx: RewardContext,
        judge: BaseJudge | None = None,
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
    """Per-sentence: count words not in the top-N most common English words
    (skipping proper nouns). Penalty kicks in once that count exceeds
    `allowed_uncommon` per sentence. Reward = 1 - (mean penalty across
    sentences), clipped to [0, 1].

    Defaults (top_n=3000, allowed=1, severity=0.5) calibrated against
    chosen/rejected/bad distributions in data/{sft,dpo}.jsonl. See
    TestVocabSimplicityCalibration for the regression targets."""

    name = "vocab"

    def __init__(
        self, top_n: int = _DEFAULT_TOP_N, allowed_uncommon: int = 1, severity: float = 0.5
    ):
        self.common_words = _common_words(top_n)
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
                if tok.lower() not in self.common_words:
                    uncommon += 1
            excess = max(0, uncommon - self.allowed_uncommon)
            penalties.append(min(1.0, excess * self.severity))
        return max(0.0, 1.0 - sum(penalties) / len(penalties))


# ---------- Combined judge bundle (meaning + difficulty in one call) ----------
#
# Two iterations of benchmarking, both 3 quality tiers × 3 runs against
# anthropic/claude-haiku-4-5:
#
#   round 1 (no CEFR anchors): combined is 2.1× faster than two independent
#     calls (0.97s vs 2.04s), 22% fewer completion tokens, 27% fewer prompt
#     tokens. Meaning scores agree 18/18, but CEFR levels disagree 0/9 in a
#     systematic way (combined rates everything one notch harder).
#
#   round 2 (with A1/A2/B1 anchors from samples.jsonl, the same set the
#     verifier already uses for CEFR classification): combined still wins
#     on latency (1.16s vs 1.92s, ~1.65×) and decode (28 vs 36 tokens).
#     Anchors lift CEFR agreement from 0/9 to 6/9. Spot-check on 6 random
#     real Opus simplifications confirms the anchored judge classifies
#     5/6 as A2 (the LEVEL_SCORES apex) — anchors keep the reward signal
#     pointed in the right direction.
#
# Field names are 1-char to keep decode cheap on hosted APIs.

# CEFR few-shot anchors. Loaded once at import; the file is small and
# read-only here. Three explicit fields rather than a dict so the template
# substitution stays readable.
_SAMPLES_PATH = Path(__file__).resolve().parents[2] / "samples.jsonl"


def _load_cefr_anchor(level: str) -> str:
    """First sample matching `level` from samples.jsonl. Raises if the
    file is missing or the level has no sample — the judge prompt assumes
    these exist."""
    with open(_SAMPLES_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["level"] == level:
                return r["text"]
    raise KeyError(f"no {level} sample in {_SAMPLES_PATH}")


_A1_ANCHOR = _load_cefr_anchor("A1")
_A2_ANCHOR = _load_cefr_anchor("A2")
_B1_ANCHOR = _load_cefr_anchor("B1")


_JUDGE_PROMPT_TEMPLATE = f"""Rate this simplification on three axes.

f (facts kept): 5 = all key facts present; 1 = most key facts dropped. Dropping decorative detail is fine.
h (no hallucinations): 5 = nothing invented; 1 = many invented facts.
lvl (CEFR level of the simplification — calibrate against the reference texts below).

Reference CEFR levels:

A1 example:
{_A1_ANCHOR}

A2 example:
{_A2_ANCHOR}

B1 example:
{_B1_ANCHOR}

B2+ is any text noticeably harder than the B1 example above.

SOURCE:
{{source}}

SIMPLIFICATION TO RATE:
{{output}}

Respond with ONLY this JSON, no prose, no markdown:
{{{{"f": int, "h": int, "lvl": "A1"|"A2"|"B1"|"B2+"}}}}"""

# Shared per-rollout cache so meaning + difficulty rewards split one judge
# call. Bounded to ~1024 entries (FIFO eviction) since GRPO will see many
# unique (source, output) pairs across a training run; we just want intra-
# batch sharing, not unbounded growth.
_judge_cache: dict[tuple[str, str], dict] = {}
_JUDGE_CACHE_MAX = 1024


def _judge_bundle(judge: BaseJudge | None, source: str, output: str) -> dict:
    """Single combined judge call returning {f, h, lvl}. Cached on
    (source, output) so the second reward component shares the first's
    HTTP round-trip. Returns {} on missing judge or failure — callers
    handle their own neutral fallback."""
    key = (source, output)
    cached = _judge_cache.get(key)
    if cached is not None:
        return cached
    if judge is None:
        return {}
    prompt = _JUDGE_PROMPT_TEMPLATE.format(source=source, output=output)
    try:
        result = judge.evaluate(prompt)
    except Exception:
        return {}
    if not isinstance(result, dict):
        return {}
    _judge_cache[key] = result
    if len(_judge_cache) > _JUDGE_CACHE_MAX:
        # Insertion-order FIFO eviction (dicts preserve insertion order ≥ 3.7).
        _judge_cache.pop(next(iter(_judge_cache)))
    return result


# ---------- SemanticPreservationReward ----------


class SemanticPreservationReward(RewardComponent):
    """Reads `f` (facts) and `h` (no-hallucinations) from the combined
    judge bundle and averages them. Both axes 1-5, normalized to [0, 1]."""

    name = "meaning"

    def compute(self, output: str, ctx: RewardContext, judge: BaseJudge | None = None) -> float:
        if judge is None:
            return 0.5  # no judge → unknown; mid score so we don't crash GRPO
        result = _judge_bundle(judge, ctx.source, output)
        try:
            facts = float(result.get("f", 0))
            halluc = float(result.get("h", 0))
        except (TypeError, ValueError):
            return 0.5
        if facts == 0 or halluc == 0:
            return 0.5  # garbage/missing → neutral, not catastrophic
        # 1-5 → 0-1 per axis, then average.
        score = ((facts - 1) / 4 + (halluc - 1) / 4) / 2
        return max(0.0, min(1.0, score))


# ---------- Stubs (TODO) ----------


class RepetitionReward(RewardComponent):
    """Penalize degenerate outputs that loop on short n-grams or repeat
    whole sentences. This catches the GRPO failure mode where the policy
    learns to stuff the completion budget with high-vocab-score gibberish.

    Two independent signals, the lower of the two wins:

      1. Distinct-4-gram ratio: unique 4-grams / total 4-grams. Looping
         text reuses the same 4-gram many times → low ratio. A natural
         A2 paragraph sits at ~0.95+ even with normal repetition of
         function words like "the cat is".
      2. Repeated-sentence detection: if any sentence appears more than
         once after normalization, multiply reward by 1/k where k is the
         repetition count. Two copies → 0.5; three copies → 0.33; etc.

    For outputs shorter than `min_tokens` (default 8) the signal isn't
    measurable, so we return 1.0 rather than incidentally penalize
    short-but-fine outputs."""

    name = "repetition"

    def __init__(
        self,
        n: int = 4,
        min_tokens: int = 8,
        soft_floor: float = 0.55,  # distinct-4gram ratio at which reward = 0
        soft_ceiling: float = 0.95,  # distinct-4gram ratio at which reward = 1
    ):
        self.n = n
        self.min_tokens = min_tokens
        self.soft_floor = soft_floor
        self.soft_ceiling = soft_ceiling

    def compute(self, output: str, ctx: RewardContext, judge=None) -> float:
        toks = output.split()
        if len(toks) < self.min_tokens:
            return 1.0

        # Signal 1: distinct-n-gram ratio
        ngrams = [tuple(toks[i : i + self.n]) for i in range(len(toks) - self.n + 1)]
        if not ngrams:
            return 1.0
        ratio = len(set(ngrams)) / len(ngrams)
        # Map [soft_floor, soft_ceiling] → [0, 1] linearly
        ngram_reward = (ratio - self.soft_floor) / (self.soft_ceiling - self.soft_floor)
        ngram_reward = max(0.0, min(1.0, ngram_reward))

        # Signal 2: repeated whole sentences (after light normalization)
        sentences = [s.strip().lower() for s in split_sentences(output) if s.strip()]
        sent_reward = 1.0
        if len(sentences) >= 2:
            from collections import Counter

            counts = Counter(sentences)
            max_rep = max(counts.values())
            if max_rep > 1:
                sent_reward = 1.0 / max_rep

        return min(ngram_reward, sent_reward)


_MARKDOWN_PATTERNS = [
    re.compile(r"\*\*[^*\n]+\*\*"),  # **bold**
    re.compile(r"__[^_\n]+__"),  # __bold__
    re.compile(r"^#{1,6}\s", re.MULTILINE),  # # heading at line start
    re.compile(r"^\s*[-*+]\s", re.MULTILINE),  # bullet at line start ("- ", "* ", "+ ")
    re.compile(r"^\s*\d+\.\s\S", re.MULTILINE),  # numbered list at line start
    re.compile(r"`[^`\n]+`"),  # `inline code` or ```block```
    re.compile(r"^>\s", re.MULTILINE),  # > blockquote
    re.compile(r"\[[^\]\n]+\]\([^)\n]+\)"),  # [link](url)
]


class NoMarkdownReward(RewardComponent):
    """Format-adherence guardrail. Returns 1.0 if the output is plain
    prose, 0.0 if any markdown marker is detected.

    Binary by design — half-credit for "kind of markdown" would invite
    false positives on legitimate prose punctuation (em-dashes, mid-
    sentence numbers, parenthetical refs). The patterns are anchored
    where possible so things like "In 1985 the building opened" or
    "Washington — the capital" don't trigger."""

    name = "markdown"

    def compute(self, output: str, ctx: RewardContext, judge=None) -> float:
        if not output:
            return 1.0
        for pat in _MARKDOWN_PATTERNS:
            if pat.search(output):
                return 0.0
        return 1.0


class SmoothDifficultyReward(RewardComponent):
    """Reads `lvl` (CEFR classification) from the combined judge bundle and
    maps to a smooth score. A2 is the target; over-simplification (A1) is
    a soft failure, under-simplification (B1, B2+) is increasingly bad.

    Sharing the bundle with SemanticPreservationReward means this reward
    adds **zero extra judge calls** per rollout — both components read from
    the same cached result."""

    name = "difficulty"

    # Smooth score per CEFR level. A2 is the apex; the mapping was chosen
    # so the gradient pushes A1 → A2 (small lift) and B1 → A2 (bigger lift),
    # which matches the failure-mode asymmetry: over-simplifying is mostly
    # an aesthetic flaw, under-simplifying defeats the whole task.
    LEVEL_SCORES: dict[str, float] = {
        "A2": 1.0,
        "A1": 0.6,
        "<A1": 0.2,
        "B1": 0.4,
        "B2+": 0.0,
    }

    def compute(self, output: str, ctx: RewardContext, judge: BaseJudge | None = None) -> float:
        if judge is None:
            return 0.5
        result = _judge_bundle(judge, ctx.source, output)
        lvl = result.get("lvl")
        if not isinstance(lvl, str):
            return 0.5
        return self.LEVEL_SCORES.get(lvl, 0.5)


# ---------- CombinedReward ----------


class CombinedReward(RewardComponent):
    """Weighted sum of components. If a 'meaning' component is present and
    scores below `meaning_gate`, the whole reward is zeroed.

    The meaning gate is the safety belt against the model gaming the
    cheaper rewards (length, vocab) by silently dropping content.

    The default gate is intentionally low (0.3) — early in GRPO the policy
    is weak and meaning scores cluster around 0.5; a higher gate would zero
    the reward on most rollouts and starve the training signal. 0.3 still
    blocks catastrophic faithfulness failures (judge says 1-2 / 5 on
    both axes → meaning ≈ 0.0-0.25) while letting mediocre-but-recoverable
    rollouts contribute gradient through the weighted sum."""

    name = "combined"

    def __init__(
        self,
        components: list[tuple[float, RewardComponent]],
        meaning_gate: float = 0.3,
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


# ---------- v5 (2026-05-23): entity preservation + multiplicative combined ----------
#
# v3 (above) was an additive weighted sum of 6 components. Auditing 25 model
# outputs against subjective quality revealed two problems:
#   1. vocab_reward had *negative* Spearman correlation with quality (−0.57).
#      It punishes outputs that gloss technical terms (a *good* simplification
#      move) because the gloss adds out-of-top-3000 tokens.
#   2. repetition_reward and markdown_reward stayed at ~1.0 across all 25
#      outputs — they never discriminated, just inflated the baseline.
#
# v5 changes:
#   * Drop vocab entirely.
#   * Convert repetition/markdown to *hard gates* (multiply by 0 when fired).
#   * Add EntityPreservationReward — % of named entities from source that
#     appear in output. Strongest new signal we found (+0.57 ρ on held-out).
#   * base = 0.6 * meaning + 0.4 * entity_preservation
#   * Difficulty becomes a soft multiplier: A2=1.0, A1=0.85, B1=0.6, B2+=0.0.
#   * Length stays as an asymmetric multiplier — short is bad (content lost),
#     slightly long is fine (content preserved with gloss).


_ENTITY_RES = [
    re.compile(r"(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)"),  # CamelCase multi-word
    re.compile(r"\b[A-Z]{2,}\b"),  # ALL-CAPS acronyms (UCL, MI6, UAV)
    re.compile(r"\b\d{2,4}\b"),  # years and small counts
]


class EntityPreservationReward(RewardComponent):
    """Fraction of named entities from the source that survive in the output.

    Entity definition is heuristic: capitalized multi-word phrases, all-caps
    acronyms ≥2 letters, and 2-4 digit numbers. Single capitalized words are
    intentionally excluded (sentence-initial false positives). Matching is
    case-insensitive substring against the output.

    Returns 1.0 when the source has no detected entities (nothing to compare).
    """

    name = "entity"

    def compute(self, output: str, ctx: RewardContext, judge=None) -> float:
        ents: set[str] = set()
        for r in _ENTITY_RES:
            for m in r.findall(ctx.source):
                ents.add(m.strip())
        if not ents:
            return 1.0
        out_lower = output.lower()
        kept = sum(1 for e in ents if e.lower() in out_lower)
        return kept / len(ents)


_V5_DIFF_FACTOR = {"A2": 1.0, "A1": 0.85, "B1": 0.6, "B2+": 0.0, "<A1": 0.5, "NA": 0.0}


def _v5_length_factor(source: str, output: str) -> float:
    """Asymmetric window: [0.8, 1.4] = 1.0, decays to 0 at [0.5, 1.8]."""
    sw = len(source.split())
    ow = len(output.split())
    if sw == 0:
        return 0.0
    r = ow / sw
    if 0.8 <= r <= 1.4:
        return 1.0
    if r < 0.5 or r > 1.8:
        return 0.0
    if r < 0.8:
        return (r - 0.5) / 0.3
    return (1.8 - r) / 0.4


def _v5_has_markdown(output: str) -> bool:
    return any(p.search(output) for p in _MARKDOWN_PATTERNS)


def _v5_has_loop(output: str) -> bool:
    """Fire the gate when RepetitionReward would score < 0.3 (loop or
    repeated-sentence pattern). Reuses the existing detection."""
    rep_score = RepetitionReward().compute(output, RewardContext(source=""))
    return rep_score < 0.3


class CombinedRewardV5(RewardComponent):
    """v5 combined reward: multiplicative gating + asymmetric length factor.

    reward = base * difficulty_factor * length_factor
    where base = meaning_weight * meaning + entity_weight * entity
    and the reward is zeroed if:
      - meaning < meaning_gate (catastrophic faithfulness failure)
      - output contains markdown
      - output exhibits loop / repeated-sentence pattern
    """

    name = "combined_v5"

    def __init__(
        self,
        meaning_weight: float = 0.6,
        entity_weight: float = 0.4,
        meaning_gate: float = 0.3,
    ):
        self.meaning = SemanticPreservationReward()
        self.entity = EntityPreservationReward()
        self.meaning_weight = meaning_weight
        self.entity_weight = entity_weight
        self.meaning_gate = meaning_gate

    def compute(self, output: str, ctx: RewardContext, judge=None) -> float:
        meaning = self.meaning.compute(output, ctx, judge=judge)
        if meaning < self.meaning_gate:
            return 0.0
        if _v5_has_markdown(output):
            return 0.0
        if _v5_has_loop(output):
            return 0.0
        entity = self.entity.compute(output, ctx)
        base = self.meaning_weight * meaning + self.entity_weight * entity
        bundle = _judge_bundle(judge, ctx.source, output) if judge else {}
        lvl = bundle.get("lvl", "NA")
        diff_factor = _V5_DIFF_FACTOR.get(lvl, 0.5)
        len_factor = _v5_length_factor(ctx.source, output)
        return max(0.0, min(1.0, base * diff_factor * len_factor))


def _default_combined_v5() -> CombinedRewardV5:
    return CombinedRewardV5(meaning_weight=0.6, entity_weight=0.4, meaning_gate=0.3)


# ---------- mlx_lm_lora @register_reward_function adapters ----------
#
# mlx_lm_lora calls reward functions with the signature
#   (prompts: list[str], completions: list[str], answer: list[str],
#    types: list[str] | None) -> list[float]
# Note: the framework passes `answer` as a *singular* kwarg even though it
# is a list — the param name is part of the contract. Returning one float
# in [0, 1] per (prompt, completion). The framework picks the functions up
# by name via --reward-functions and --reward-functions-file.

_LENGTH = LengthVsSourceReward()
_VOCAB = VocabSimplicityReward()
_REPETITION = RepetitionReward()
_MARKDOWN = NoMarkdownReward()
_MEANING = SemanticPreservationReward()
_DIFFICULTY = SmoothDifficultyReward()


_OPENROUTER_DEFAULT_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_DEFAULT_MODEL = "anthropic/claude-haiku-4-5"


def _get_judge():
    """Lazy-load a judge from env. Backend selection:

    * MEANING_JUDGE_BACKEND=openrouter → OpenRouter (needs OPENROUTER_API_KEY).
      Defaults: model=anthropic/claude-haiku-4-5, url=https://openrouter.ai/api/v1.
      Override via MEANING_JUDGE_MODEL / MEANING_JUDGE_URL.
    * MEANING_JUDGE_URL set            → local LM Studio (no auth).
    * neither                          → None; meaning_reward returns 0.5
      (constant contribution → no signal but no crash).
    """
    import os

    if hasattr(_get_judge, "_cached"):
        return _get_judge._cached

    backend = os.environ.get("MEANING_JUDGE_BACKEND", "").lower()
    from langsimp.verifier import LocalJudge

    if backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("MEANING_JUDGE_BACKEND=openrouter but OPENROUTER_API_KEY is not set")
        url = os.environ.get("MEANING_JUDGE_URL", _OPENROUTER_DEFAULT_URL)
        model = os.environ.get("MEANING_JUDGE_MODEL", _OPENROUTER_DEFAULT_MODEL)
        _get_judge._cached = LocalJudge(base_url=url, model_name=model, api_key=api_key)
        return _get_judge._cached

    url = os.environ.get("MEANING_JUDGE_URL")
    if not url:
        return None
    model = os.environ.get("MEANING_JUDGE_MODEL", "google/gemma-4-26b-a4b")
    _get_judge._cached = LocalJudge(base_url=url, model_name=model)
    return _get_judge._cached


try:
    from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function
except ImportError:
    # Tests don't need mlx_lm_lora — fall back to a no-op decorator so
    # this module is still importable.
    def register_reward_function(name=None):
        def deco(fn):
            return fn

        return deco


@register_reward_function()
def length_reward(prompts, completions, answer, types=None) -> list[float]:
    return [
        _LENGTH.compute(c, RewardContext(source=p, answer=a))
        for p, c, a in zip(prompts, completions, answer, strict=True)
    ]


@register_reward_function()
def vocab_reward(prompts, completions, answer, types=None) -> list[float]:
    return [
        _VOCAB.compute(c, RewardContext(source=p, answer=a))
        for p, c, a in zip(prompts, completions, answer, strict=True)
    ]


@register_reward_function()
def meaning_reward(prompts, completions, answer, types=None) -> list[float]:
    judge = _get_judge()
    return [
        _MEANING.compute(c, RewardContext(source=p, answer=a), judge=judge)
        for p, c, a in zip(prompts, completions, answer, strict=True)
    ]


@register_reward_function()
def repetition_reward(prompts, completions, answer, types=None) -> list[float]:
    return [
        _REPETITION.compute(c, RewardContext(source=p, answer=a))
        for p, c, a in zip(prompts, completions, answer, strict=True)
    ]


@register_reward_function()
def difficulty_reward(prompts, completions, answer, types=None) -> list[float]:
    """Reads the CEFR `lvl` field from the same combined judge bundle as
    meaning_reward. Calling order matters: meaning_reward first warms the
    cache, then difficulty_reward reads the cached result for free. mlx-
    lm-lora calls reward functions in the order listed in --reward-
    functions, so register them in that order in the shell scripts."""
    judge = _get_judge()
    return [
        _DIFFICULTY.compute(c, RewardContext(source=p, answer=a), judge=judge)
        for p, c, a in zip(prompts, completions, answer, strict=True)
    ]


@register_reward_function()
def markdown_reward(prompts, completions, answer, types=None) -> list[float]:
    return [
        _MARKDOWN.compute(c, RewardContext(source=p, answer=a))
        for p, c, a in zip(prompts, completions, answer, strict=True)
    ]


_COMBINED_V5 = _default_combined_v5()


@register_reward_function()
def v5_combined_reward(prompts, completions, answer, types=None) -> list[float]:
    """v5 stack as a single reward function. Use with
    `--reward-functions v5_combined_reward --reward-weights [1.0]`.

    Internally:
      base = 0.6 * meaning + 0.4 * entity_preservation
      then multiplied by difficulty_factor and length_factor.
    Gated to 0 on meaning<0.3, markdown present, or repetition/loop.
    """
    judge = _get_judge()
    return [
        _COMBINED_V5.compute(c, RewardContext(source=p, answer=a), judge=judge)
        for p, c, a in zip(prompts, completions, answer, strict=True)
    ]


# ---------- audit + variety (offline diagnostics) ----------
#
# Used to verify rewards make sense before training, and to monitor reward
# variance per group during/after training. Reward variance ≈ 0 inside a
# GRPO group means the advantage signal is dead.


def _default_combined() -> CombinedReward:
    """v3 weights — opinionated, not uniform. Reasoning:

      * meaning (0.40): faithfulness is the highest-stakes axis. A
        hallucinated A2 is worse than a faithful B1. Plus the meaning
        gate is the safety belt.
      * difficulty (0.25): "make this A2" is the explicit task target.
      * repetition (0.15): guardrail. Usually pinned at 1.0, but when it
        fires the failure is total — we want a real penalty.
      * vocab (0.10): per-sentence granularity that difficulty (4 buckets)
        can't see; correlated with difficulty so it shouldn't dominate.
      * length (0.05): mostly redundant with meaning + difficulty
        (dropping content lowers both). Guardrail only.
      * markdown (0.05): narrow binary failure; cheap reinforcement of
        the DPO format constraint.

    Meaning + difficulty = 0.65 of the signal, share one judge call (free).
    Guardrails sum to 0.25 and contribute zero advantage gradient when the
    policy is well-behaved — that's by design."""
    return CombinedReward(
        components=[
            (0.40, _MEANING),
            (0.25, _DIFFICULTY),
            (0.15, _REPETITION),
            (0.10, _VOCAB),
            (0.05, _LENGTH),
            (0.05, _MARKDOWN),
        ],
        meaning_gate=0.3,
    )


def audit_record(source: str, output: str, judge: BaseJudge | None = None) -> dict[str, float]:
    """Per-component scores for one (source, output) pair, plus combined."""
    ctx = RewardContext(source=source)
    out = {
        "length": _LENGTH.compute(output, ctx),
        "vocab": _VOCAB.compute(output, ctx),
        "repetition": _REPETITION.compute(output, ctx),
        "markdown": _MARKDOWN.compute(output, ctx),
        "meaning": _MEANING.compute(output, ctx, judge=judge),
        "difficulty": _DIFFICULTY.compute(output, ctx, judge=judge),
    }
    out["combined"] = _default_combined().compute(output, ctx, judge=judge)
    return out


def compute_variety(
    prompts: list[str],
    rollouts_per_prompt: list[list[str]],
    judge: BaseJudge | None = None,
) -> dict:
    """For each prompt, score its rollouts and report mean/std.

    GRPO advantage = (reward - mean) / std within each group; if std ≈ 0
    the gradient is zero and no learning happens. This function tells us
    whether our rewards are *discriminating* between rollouts.
    """
    import statistics

    combined = _default_combined()
    per_prompt: list[dict] = []
    stds: list[float] = []
    for p, rollouts in zip(prompts, rollouts_per_prompt, strict=True):
        scores = [combined.compute(r, RewardContext(source=p), judge=judge) for r in rollouts]
        mean_s = statistics.mean(scores) if scores else 0.0
        std_s = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        per_prompt.append({"mean": mean_s, "std": std_s, "rewards": scores})
        stds.append(std_s)
    return {
        "per_prompt": per_prompt,
        "mean_std": statistics.mean(stds) if stds else 0.0,
        "min_std": min(stds) if stds else 0.0,
        "max_std": max(stds) if stds else 0.0,
    }


# ---------- CLI ----------


def _variety_cli(args) -> None:
    """Sample G rollouts per prompt from a real adapter; report reward
    std per group. GRPO advantage = (reward - mean) / std within a group;
    if std ≈ 0 across most groups, GRPO can't learn — this catches that
    BEFORE we burn training compute."""
    from langsimp.inference.engine import load_model_with_adapter, make_generate_fn

    judge = None
    if args.with_judge:
        from verifier import LocalJudge

        judge = LocalJudge(base_url=args.lm_studio_url, model_name=args.judge_model)

    # Load prompts. Accept either GRPO-shape (`prompt`) or SFT-shape (`complex`).
    prompts: list[str] = []
    with open(args.prompts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prompts.append(r.get("prompt") or r.get("complex"))
            if len(prompts) >= args.n_prompts:
                break
    print(
        f"[variety] {len(prompts)} prompts × {args.group_size} rollouts at temp={args.temperature}",
        flush=True,
    )

    adapter_path = None if args.adapter == "base" else args.adapter
    model, tokenizer = load_model_with_adapter(args.model, adapter_path)
    gen = make_generate_fn(model, tokenizer, max_tokens=args.max_tokens, temp=args.temperature)

    rollouts_per_prompt: list[list[str]] = []
    for i, p in enumerate(prompts):
        rollouts: list[str] = []
        for j in range(args.group_size):
            out = gen(p)
            rollouts.append(out)
            print(
                f"  [{i + 1}/{len(prompts)}, rollout {j + 1}/{args.group_size}] {len(out.split())}w",
                flush=True,
            )
        rollouts_per_prompt.append(rollouts)

    stats = compute_variety(prompts, rollouts_per_prompt, judge=judge)
    print(f"\n=== REWARD VARIETY ({len(prompts)} prompts, G={args.group_size}) ===")
    print(f"  mean across-group std : {stats['mean_std']:.4f}")
    print(f"  min  across-group std : {stats['min_std']:.4f}")
    print(f"  max  across-group std : {stats['max_std']:.4f}")
    if stats["mean_std"] < 0.05:
        print("  ⚠️  mean std < 0.05 — GRPO advantage signal will be weak!")
    print("\n=== PER-PROMPT BREAKDOWN ===")
    for i, (p, prompt_text, rollouts) in enumerate(
        zip(stats["per_prompt"], prompts, rollouts_per_prompt, strict=True)
    ):
        print(
            f"\n[{i + 1}] mean={p['mean']:.3f} std={p['std']:.4f}  rewards={[round(r, 3) for r in p['rewards']]}"
        )
        if args.show_rollouts:
            print(f"    SOURCE ({len(prompt_text.split())}w): {prompt_text[:140]}…")
            for j, (r, score) in enumerate(zip(rollouts, p["rewards"], strict=True)):
                # Per-component scores for this rollout
                comp = audit_record(prompt_text, r, judge=judge)
                print(
                    f"    [rollout {j + 1} | combined={score:.3f} L={comp['length']:.2f} V={comp['vocab']:.2f} M={comp['meaning']:.2f}] {len(r.split())}w"
                )
                print(f"      {r[:200]}…" if len(r) > 200 else f"      {r}")


def _audit_cli(args) -> None:
    """Score a JSONL of {complex, simple} (or {complex, output}) records."""
    judge = None
    if args.with_judge:
        from verifier import LocalJudge

        judge = LocalJudge(base_url=args.lm_studio_url, model_name=args.judge_model)

    records: list[dict] = []
    with open(args.path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if args.limit:
        records = records[: args.limit]

    output_field = args.output_field
    rows: list[dict] = []
    for rec in records:
        out = rec.get(output_field) or rec.get("simple") or rec.get("output", "")
        scores = audit_record(rec["complex"], out, judge=judge)
        rows.append({"title": rec.get("title", ""), **scores})

    if not rows:
        print("no records")
        return

    keys = ["length", "vocab", "meaning", "combined"]
    means = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    print(f"\n=== REWARD AUDIT ({len(rows)} records) ===")
    for k in keys:
        print(f"  mean {k:>9}: {means[k]:.3f}")

    if args.show_worst:
        worst = sorted(rows, key=lambda r: r["combined"])[: args.show_worst]
        print(f"\n=== {args.show_worst} WORST records by combined score ===")
        for r in worst:
            print(
                f"  {r['combined']:.3f}  L={r['length']:.2f} V={r['vocab']:.2f} M={r['meaning']:.2f}  {r['title']}"
            )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    audit = sub.add_parser("audit", help="score per-component rewards on a JSONL")
    audit.add_argument("path")
    audit.add_argument("--with-judge", action="store_true")
    audit.add_argument("--lm-studio-url", default="http://127.0.0.1:1234/v1")
    audit.add_argument("--judge-model", default="google/gemma-4-26b-a4b")
    audit.add_argument(
        "--output-field",
        default="simple",
        help="JSON field that holds the model output (simple|output|...)",
    )
    audit.add_argument("--limit", type=int, default=0)
    audit.add_argument("--show-worst", type=int, default=5)

    variety = sub.add_parser(
        "variety", help="sample rollouts from an adapter and report reward std per group"
    )
    variety.add_argument("--adapter", required=True, help="adapter dir or 'base' for no adapter")
    variety.add_argument("--prompts-path", default="data/grpo/train.jsonl")
    variety.add_argument("--n-prompts", type=int, default=5)
    variety.add_argument("--group-size", type=int, default=4)
    variety.add_argument("--temperature", type=float, default=0.8)
    variety.add_argument("--max-tokens", type=int, default=512)
    variety.add_argument("--model", default="mlx-community/gemma-3-1b-it-bf16")
    variety.add_argument("--with-judge", action="store_true")
    variety.add_argument(
        "--show-rollouts",
        action="store_true",
        help="print each rollout's text and per-component scores",
    )
    variety.add_argument("--lm-studio-url", default="http://127.0.0.1:1234/v1")
    variety.add_argument("--judge-model", default="google/gemma-4-26b-a4b")

    args = p.parse_args()
    if args.cmd == "audit":
        _audit_cli(args)
    elif args.cmd == "variety":
        _variety_cli(args)


if __name__ == "__main__":
    main()
