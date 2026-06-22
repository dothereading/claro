"""Shared spaCy pipeline for the v10 reward.

One singleton `en_core_web_sm` pipeline, loaded lazily and reused by the
band component (sentence segmentation), the vocab component (lemmas, POS,
`like_num`), and `scripts/build_vocab_list.py` (offline list construction).

Loading it once in one place is what keeps the vocab list and the scoring
path definitionally consistent: a word is "in the list" iff the lemma the
*same* pipeline produces for it is in the list. NER is disabled — we never
use entity spans and it is the most expensive component.
"""

from __future__ import annotations

import functools

import spacy
from spacy.language import Language
from spacy.tokens import Doc

_MODEL = "en_core_web_sm"


@functools.lru_cache(maxsize=1)
def nlp() -> Language:
    """The shared pipeline. Parser kept (it sets sentence boundaries);
    NER disabled (unused, and the slowest pipe)."""
    return spacy.load(_MODEL, disable=["ner"])


def doc(text: str) -> Doc:
    """Process `text` through the shared pipeline."""
    return nlp()(text)


def sentences(text: str) -> list[str]:
    """Sentence strings via the shared pipeline's dependency boundaries."""
    return [s.text.strip() for s in doc(text).sents if s.text.strip()]


def word_lemmas(text: str) -> list[str]:
    """Lowercased lemmas of the alphabetic tokens in `text`.

    Used by the offline vocab builder to round-trip a word list through the
    exact lemmatizer that scoring uses.
    """
    return [t.lemma_.lower() for t in doc(text) if t.is_alpha]


# ---------- shared hard-word predicate (used by vocab + gloss matcher) ----------

# POS tags / token kinds that never count as hard vocabulary regardless of
# allow-list membership.
EXEMPT_POS = frozenset({"PROPN", "NUM", "PUNCT", "SYM", "SPACE"})


def is_exempt_token(tok) -> bool:
    """True if `tok` is never a 'hard' word on type grounds alone (before any
    allow-list / source / gloss check): punctuation, space, proper nouns,
    numbers, possessive/contraction parts, single-character tokens."""
    if tok.is_space or tok.is_punct:
        return True
    if tok.pos_ in EXEMPT_POS or tok.like_num:
        return True
    return tok.pos_ == "PART" or len(tok.text.strip()) <= 1


# ---------- gloss detection (Edit 1.2) ----------

_GLOSS_MAX_DEF_WORDS = 12
_COPULA_LEMMAS = frozenset({"be", "mean", "refer"})
# dependency labels for the definition (complement) side of a copula.
_DEF_DEPS = frozenset({"attr", "acomp", "oprd", "dobj", "advcl", "relcl", "acl"})
_SUBJ_DEPS = frozenset({"nsubj", "nsubjpass"})


def _definition_is_clean(def_tokens: list, allow) -> bool:
    """A gloss's definition must be short and itself vocab-clean: every
    content token's lemma is in the allow-list (so we don't 'gloss' a hard
    term with more hard terms). Conservative by design — under-detecting a
    gloss only nudges the model to gloss more clearly, the desired direction.
    """
    words = [t for t in def_tokens if not t.is_punct and not t.is_space]
    if not words or len(words) > _GLOSS_MAX_DEF_WORDS:
        return False
    for t in words:
        if is_exempt_token(t):
            continue
        if t.lemma_.lower() not in allow:
            return False
    return True


def find_glossed_lemmas(doc_obj, allow) -> set[str]:
    """Lowercase lemmas of terms that are introduced WITH a plain-language
    gloss in the candidate. Two conservative patterns:

      * copula/definition: "T is/means/refers to <clean short definition>"
        (subject T is the glossed term);
      * appositive: "T, <clean short definition>, ..." (head of the appos).

    `allow` is the A2 allow-list, used to verify the definition is itself
    vocab-clean. Returns the set of glossed term lemmas.
    """
    glossed: set[str] = set()
    for sent in doc_obj.sents:
        for tok in sent:
            # copula / "means" / "refers to"
            if tok.lemma_.lower() in _COPULA_LEMMAS and tok.pos_ in {"AUX", "VERB"}:
                subjs = [c for c in tok.children if c.dep_ in _SUBJ_DEPS]
                defs = [c for c in tok.children if c.dep_ in _DEF_DEPS or c.dep_ == "prep"]
                if not subjs or not defs:
                    continue
                def_tokens: list = []
                for d in defs:
                    def_tokens.extend(d.subtree)
                if not _definition_is_clean(def_tokens, allow):
                    continue
                for s in subjs:
                    for t in s.subtree:
                        if t.pos_ in {"NOUN", "PROPN"}:
                            glossed.add(t.lemma_.lower())
            # appositive: "T, a kind of Y, ..."
            if tok.dep_ == "appos" and _definition_is_clean(list(tok.subtree), allow):
                glossed.add(tok.head.lemma_.lower())
    return glossed
