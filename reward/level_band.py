"""Component 1: level_band (deterministic).

Targets the readability *band* of the A2 anchors instead of maximizing
simplicity. Two metrics — Flesch Reading Ease (textstat) and mean sentence
length in words (shared spaCy pipeline) — are each scored with a trapezoid
bump that is flat inside the anchors' [q1, q3] inter-quartile range and
falls off linearly outside it, never below a floor. The two scores
multiply.

Bands are calibrated offline (scripts/calibrate_band.py -> config/band.json)
and loaded once at startup; nothing here recomputes during training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import textstat

from reward.nlp import doc

_BAND_PATH = Path(__file__).resolve().parents[1] / "config" / "band.json"

_FLOOR = 0.2


def trapezoid(
    x: float,
    q1: float,
    q3: float,
    falloff_frac: float = 0.5,
    floor: float = _FLOOR,
) -> float:
    """Full score inside [q1, q3]; linear falloff over a margin of
    `falloff_frac * (q3 - q1)` on each side; never below `floor`.

    The floor (not 0) preserves gradient direction for off-band rollouts.
    """
    width = q3 - q1
    margin = max(falloff_frac * width, 1e-6)
    if q1 <= x <= q3:
        return 1.0
    if x < q1:
        return max(floor, 1.0 - (q1 - x) / margin)
    return max(floor, 1.0 - (x - q3) / margin)


def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease via textstat (higher = easier)."""
    return float(textstat.flesch_reading_ease(text))


def _word_tokens(span) -> list:
    """Word-like tokens of a spaCy span/doc (drops punctuation and spaces)."""
    return [t for t in span if not t.is_punct and not t.is_space]


# Dependency labels for the v11.1 syntactic features. spaCy's en_core_web_sm
# emits the ClearNLP-style passive labels (nsubjpass / auxpass), not UD's
# nsubj:pass — verified on real outputs in the v11 pre-flight.
_PASSIVE_DEPS = frozenset({"nsubjpass", "auxpass"})
_SUBORD_DEPS = frozenset({"advcl", "ccomp", "xcomp", "acl", "relcl", "csubj"})


def _sent_passive(sent) -> int:
    """Passive *constructions* in a sentence (count once per passive verb, not
    per nsubjpass/auxpass token)."""
    return len({t.head.i for t in sent if t.dep_ in _PASSIVE_DEPS})


def _sent_subord(sent) -> int:
    return sum(1 for t in sent if t.dep_ in _SUBORD_DEPS)


def _sent_appos(sent) -> int:
    return sum(1 for t in sent if t.dep_ == "appos")


def _doc_stats(d) -> dict:
    """All band features from a single parse: counts + MSL + per-sentence means
    of passive / subordination / appositive."""
    sents = [s for s in d.sents if _word_tokens(s)]
    n_words = len(_word_tokens(d))
    if not sents:
        return {"n_sents": 0, "n_words": n_words, "msl": 0.0,
                "passive": 0.0, "subord": 0.0, "appos": 0.0}
    n = len(sents)
    return {
        "n_sents": n,
        "n_words": n_words,
        "msl": sum(len(_word_tokens(s)) for s in sents) / n,
        "passive": sum(_sent_passive(s) for s in sents) / n,
        "subord": sum(_sent_subord(s) for s in sents) / n,
        "appos": sum(_sent_appos(s) for s in sents) / n,
    }


def mean_sentence_length(text: str) -> float:
    """Mean words per sentence. 0.0 for text with no sentences."""
    return _doc_stats(doc(text))["msl"]


def syntactic_features(text: str) -> tuple[float, float, float]:
    """(passive, subordination, appositive) per-sentence means — the v11.1
    syntactic-density features that capture B1 drift (FRE/MSL are blind to it)."""
    s = _doc_stats(doc(text))
    return s["passive"], s["subord"], s["appos"]


@dataclass(frozen=True)
class Band:
    """Calibrated inter-quartile bands for the A2 anchor distribution. PASS/SUB
    are required (v11.1); APPOS is optional (dropped if its detector is too
    noisy — CHANGE 1.3)."""

    fre_q1: float
    fre_q3: float
    msl_q1: float
    msl_q3: float
    pass_q1: float
    pass_q3: float
    sub_q1: float
    sub_q3: float
    appos_q1: float | None = None
    appos_q3: float | None = None

    @classmethod
    def load(cls, path: Path | str = _BAND_PATH) -> Band:
        d = json.loads(Path(path).read_text())
        return cls(
            fre_q1=float(d["fre_q1"]), fre_q3=float(d["fre_q3"]),
            msl_q1=float(d["msl_q1"]), msl_q3=float(d["msl_q3"]),
            pass_q1=float(d["pass_q1"]), pass_q3=float(d["pass_q3"]),
            sub_q1=float(d["sub_q1"]), sub_q3=float(d["sub_q3"]),
            appos_q1=float(d["appos_q1"]) if "appos_q1" in d else None,
            appos_q3=float(d["appos_q3"]) if "appos_q3" in d else None,
        )


def level_band(candidate: str, band: Band) -> float:
    """Trapezoid bands on FRE × MSL × passive × subordination (× appositive,
    if calibrated). Two-sided by design: too-high density reads B1, too-low
    reads choppy A1 — both fall outside band. Targets the anchors, not the
    minimum.

    Degenerate outputs (<2 sentences or <10 words) get floor**2.
    """
    s = _doc_stats(doc(candidate))
    if s["n_sents"] < 2 or s["n_words"] < 10:
        return _FLOOR * _FLOOR
    score = (
        trapezoid(flesch_reading_ease(candidate), band.fre_q1, band.fre_q3)
        * trapezoid(s["msl"], band.msl_q1, band.msl_q3)
        * trapezoid(s["passive"], band.pass_q1, band.pass_q3)
        * trapezoid(s["subord"], band.sub_q1, band.sub_q3)
    )
    if band.appos_q1 is not None:
        score *= trapezoid(s["appos"], band.appos_q1, band.appos_q3)
    return score
