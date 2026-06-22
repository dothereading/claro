"""Reward composition.

    reward(source, candidate) = level_band x vocab x fidelity x format_gates

Multiplicative by design: every component is in [0, 1], so any one can veto
(drive toward 0) but none can rescue a rollout that fails another. Gated
rollouts (format_gates == 0) short-circuit before the judge call — they are
already 0 and must not cost money.

`reward()` uses module-level singletons for the band and the fidelity
scorer; tests and the validation harness inject their own. Set
`use_fidelity=False` for Arm B (the free, no-judge arm), where the fidelity
term is identically 1.0.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

from reward.fidelity import FidelityScorer, build_scorer
from reward.gates import format_gates
from reward.level_band import Band, level_band
from reward.vocab import vocab_term


@dataclass
class RewardResult:
    total: float
    components: dict[str, float]
    debug: dict = field(default_factory=dict)
    skipped_judge: bool = False


@functools.lru_cache(maxsize=1)
def default_band() -> Band:
    return Band.load()


@functools.lru_cache(maxsize=1)
def default_scorer() -> FidelityScorer:
    return build_scorer()


def reward(
    source: str,
    candidate: str,
    *,
    band: Band | None = None,
    scorer: FidelityScorer | None = None,
    use_fidelity: bool = True,
) -> RewardResult:
    candidate = candidate.strip()

    gate = format_gates(candidate)
    if gate == 0.0:
        return RewardResult(
            total=0.0,
            components={"level_band": 0.0, "vocab": 0.0, "fidelity": 0.0, "gates": 0.0},
            debug={"gated": True},
            skipped_judge=True,
        )

    band = band or default_band()
    lb = level_band(candidate, band)
    vt, hard_words, vocab_dbg = vocab_term(source, candidate)

    if use_fidelity:
        scorer = scorer or default_scorer()
        fr = scorer.score(source, candidate)
        ft = fr.term
        fidelity_debug = {
            "recall": fr.recall,
            "recall_term": fr.recall_term,
            "halluc_term": fr.halluc_term,
            "n_unsupported": fr.n_unsupported,
            "failed": fr.failed,
            "from_cache": fr.from_cache,
            "judge": fr.judge_json,
        }
    else:
        ft = 1.0
        fidelity_debug = {"disabled": True}

    return RewardResult(
        total=lb * vt * ft,
        components={"level_band": lb, "vocab": vt, "fidelity": ft, "gates": 1.0},
        debug={
            "hard_words": hard_words,
            "vocab": {"glossed": vocab_dbg.glossed, "bare_source": vocab_dbg.bare_source,
                      "invented": vocab_dbg.invented},
            "fidelity": fidelity_debug,
        },
        skipped_judge=not use_fidelity,
    )
