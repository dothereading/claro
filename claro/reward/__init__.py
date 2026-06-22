"""Claro's cardinal reward for the CEFR-A2 simplifier.

reward(source, candidate) = level_band x vocab x fidelity x format_gates

Each component lives in its own module and returns a value in [0, 1]:
  * level_band  (reward.c1_level_band)  — deterministic readability-band match
  * vocab       (reward.c2_vocab)       — deterministic hard-word penalty
  * fidelity    (reward.c3_fidelity)    — one LLM judge call per rollout
  * gates       (reward.c4_gates)       — hard 0/1 format mask (lifted from v9)

The shared spaCy pipeline (reward.nlp) is loaded once and used identically
by the band and vocab components and by scripts/build_vocab_list.py, so
list-membership and token-lemma stay definitionally consistent.
"""

__all__ = ["RewardResult", "reward"]


def __getattr__(name: str):
    # Lazy so that importing a single component (e.g. reward.nlp from the
    # offline vocab builder) doesn't pull in the judge / compose stack.
    if name in __all__:
        from claro.reward import compose

        return getattr(compose, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
