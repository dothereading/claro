"""Component 4: format_gates (hard 0/1 mask).

Self-contained markdown and degenerate-loop detection. A gated rollout already
scores 0, so the trainer can short-circuit and skip the (paid) judge call.

No length factor and no preamble gate: Component 3's recall term covers what the
old Gaussian length factor proxied, and preamble was only ever a ranking signal.
"""

from __future__ import annotations

import re
from collections import Counter

from claro.verifier import split_sentences

_MARKDOWN_PATTERNS = [
    re.compile(r"\*\*[^*\n]+\*\*"),  # **bold**
    re.compile(r"__[^_\n]+__"),  # __bold__
    re.compile(r"^#{1,6}\s", re.MULTILINE),  # # heading at line start
    re.compile(r"^\s*[-*+]\s", re.MULTILINE),  # bullet at line start
    re.compile(r"^\s*\d+\.\s\S", re.MULTILINE),  # numbered list at line start
    re.compile(r"`[^`\n]+`"),  # `inline code`
    re.compile(r"^>\s", re.MULTILINE),  # > blockquote
    re.compile(r"\[[^\]\n]+\]\([^)\n]+\)"),  # [link](url)
]

# distinct-4gram ratio mapped linearly from [floor, ceiling] -> [0, 1]
_NGRAM, _SOFT_FLOOR, _SOFT_CEILING, _MIN_TOKENS = 4, 0.55, 0.95, 8
_LOOP_THRESHOLD = 0.3  # repetition score below this trips the gate


def has_markdown(candidate: str) -> bool:
    return any(p.search(candidate) for p in _MARKDOWN_PATTERNS)


def _repetition_score(candidate: str) -> float:
    """Lower of two signals: distinct-4gram ratio and repeated-sentence penalty.
    1.0 = clean; toward 0 = looping / repeated text. Mirrors the v9 detector."""
    toks = candidate.split()
    if len(toks) < _MIN_TOKENS:
        return 1.0

    ngrams = [tuple(toks[i : i + _NGRAM]) for i in range(len(toks) - _NGRAM + 1)]
    if not ngrams:
        return 1.0
    ratio = len(set(ngrams)) / len(ngrams)
    ngram_reward = (ratio - _SOFT_FLOOR) / (_SOFT_CEILING - _SOFT_FLOOR)
    ngram_reward = max(0.0, min(1.0, ngram_reward))

    sentences = [s.strip().lower() for s in split_sentences(candidate) if s.strip()]
    sent_reward = 1.0
    if len(sentences) >= 2:
        max_rep = max(Counter(sentences).values())
        if max_rep > 1:
            sent_reward = 1.0 / max_rep

    return min(ngram_reward, sent_reward)


def has_loop(candidate: str) -> bool:
    return _repetition_score(candidate) < _LOOP_THRESHOLD


def format_gates(candidate: str) -> float:
    """0.0 if the candidate trips any hard gate, else 1.0."""
    return 0.0 if (has_markdown(candidate) or has_loop(candidate)) else 1.0
