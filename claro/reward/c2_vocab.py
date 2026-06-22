"""Component 2: vocab_term (deterministic) — the A2 density penalty.

Penalizes sentences containing more than one "hard" word, where hard = a
content-word lemma outside the A2 allow-list (data/vocab_1500.txt) with
exemptions for proper nouns and numbers.

v11 change (the density penalty): a SOURCE term that is off the allow-list is
no longer automatically exempt. It is exempt only if it is ALSO in the
allow-list (trivially not, for jargon) OR it is *glossed* in the candidate
(reward.nlp.find_glossed_lemmas), on its first occurrence only. A bare,
unglossed source term ("CORU uses operational research") is counted hard like
any other off-list word — which is what makes retaining jargon cost something.
The reward-maximizing move becomes: gloss the core term once, or drop it.

The allow-list is built by scripts/build_vocab_list.py through the same spaCy
lemmatizer used here, so membership and token lemmas are consistent.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from math import prod
from pathlib import Path

from claro.reward.nlp import doc, find_glossed_lemmas, is_exempt_token

_VOCAB_PATH = Path(__file__).resolve().parents[2] / "data" / "vocab_1500.txt"

_FLOOR = 0.2


@functools.lru_cache(maxsize=1)
def load_vocab(path: str | Path = _VOCAB_PATH) -> frozenset[str]:
    """The A2 allow-list as a frozenset of lowercase lemmas."""
    text = Path(path).read_text()
    return frozenset(line.strip().lower() for line in text.splitlines() if line.strip())


def source_lemmas(source: str) -> frozenset[str]:
    """Lowercase content-word lemmas of the source. Used to tell apart a term
    the candidate *echoed from the source* (off-list but legitimate to gloss)
    from one it *invented* (off-list and not in source — always hard)."""
    out: set[str] = set()
    for tok in doc(source):
        if is_exempt_token(tok) or not tok.lemma_.strip():
            continue
        out.add(tok.lemma_.lower())
    return frozenset(out)


def sentence_score(hard_count: int) -> float:
    """1.0 for 0 or 1 hard words; halves for each additional hard word."""
    return 1.0 if hard_count <= 1 else 0.5 ** (hard_count - 1)


@dataclass
class VocabDebug:
    """Per-rollout vocab breakdown for probe audits / training logs (Edit 1.4)."""

    glossed: list[str] = field(default_factory=list)       # source terms exempted via a gloss
    bare_source: list[str] = field(default_factory=list)   # off-list source terms kept bare (counted)
    invented: list[str] = field(default_factory=list)      # off-list non-source words (counted)


def _classify_token(tok, src_lemmas, allow, glossed, used_gloss) -> str:
    """One of: 'exempt', 'gloss_exempt', 'hard_bare_source', 'hard_invented'."""
    if is_exempt_token(tok):
        return "exempt"
    lemma = tok.lemma_.lower()
    if not lemma.strip() or lemma in allow:
        return "exempt"
    # off the allow-list:
    if lemma in src_lemmas:
        if lemma in glossed and lemma not in used_gloss:
            return "gloss_exempt"   # licensed once by its gloss
        return "hard_bare_source"   # bare jargon retained from source -> hard
    return "hard_invented"          # off-list and not in source -> hard


def vocab_term(
    source: str,
    candidate: str,
    vocab: frozenset[str] | None = None,
) -> tuple[float, list[list[str]], VocabDebug]:
    """Score the candidate's vocabulary against the A2 allow-list.

    Returns `(score, flagged, debug)`:
      * `score`   — product of per-sentence scores, floored at 0.2;
      * `flagged` — hard words per sentence (surfaces, repeats included);
      * `debug`   — VocabDebug (glossed / bare_source / invented), for logs.
    """
    allow = vocab if vocab is not None else load_vocab()
    src_lemmas = source_lemmas(source) if source else frozenset()

    cand = doc(candidate)
    glossed = find_glossed_lemmas(cand, allow)
    used_gloss: set[str] = set()

    flagged: list[list[str]] = []
    scores: list[float] = []
    dbg = VocabDebug()

    for sent in cand.sents:
        if not any(not t.is_space for t in sent):
            continue
        hard: list[str] = []
        for tok in sent:
            cls = _classify_token(tok, src_lemmas, allow, glossed, used_gloss)
            if cls == "gloss_exempt":
                used_gloss.add(tok.lemma_.lower())
                dbg.glossed.append(tok.text)
            elif cls == "hard_bare_source":
                hard.append(tok.text)
                dbg.bare_source.append(tok.text)
            elif cls == "hard_invented":
                hard.append(tok.text)
                dbg.invented.append(tok.text)
        flagged.append(hard)
        scores.append(sentence_score(len(hard)))

    if not scores:
        return _FLOOR, [], dbg
    return max(_FLOOR, prod(scores)), flagged, dbg
