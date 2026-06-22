"""v10 cardinal reward as mlx-lm-lora reward functions.

Two arms, identical except the fidelity term (§6):

  * Arm A (full):     level_band x vocab x fidelity x gates
      --reward-functions v10_full_reward --reward-weights [1.0]
  * Arm B (no judge): level_band x vocab x gates  (fidelity ≡ 1.0, free)
      --reward-functions v10_nojudge_reward --reward-weights [1.0]

The actual reward math lives in the top-level `reward` package; this module
is the thin trainer adapter: it maps mlx-lm-lora's
(prompts, completions, answer, types) batch signature onto per-rollout
`reward.compose.reward`, runs the 8 judge calls of a group concurrently
(small semaphore), and writes the per-rollout RewardResult and a
per-iteration summary to runs/<arm>/ for the §6 readouts.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.rewards.rewards_legacy import register_reward_function
from reward.compose import RewardResult, default_band, default_scorer
from reward.compose import reward as compute_reward
from reward.c3_fidelity import load_fidelity_config

_log = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[2]

try:
    _CONCURRENCY = int(load_fidelity_config().get("concurrency", 4))
except Exception:  # pragma: no cover - config optional for Arm B
    _CONCURRENCY = 4

# One batch call == one GRPO iteration (per-arm counter).
_iter_count: dict[str, int] = {"full": 0, "nojudge": 0}


def _run_dir(arm: str) -> Path:
    d = _ROOT / "runs" / arm
    d.mkdir(parents=True, exist_ok=True)
    return d


def _word_ratio(source: str, candidate: str) -> float:
    s = len(source.split())
    return len(candidate.split()) / s if s else 0.0


def _hard_rate(res: RewardResult) -> float:
    """Off-list hard words per sentence for one rollout."""
    hw = res.debug.get("hard_words") or []
    if not hw:
        return 0.0
    return sum(len(s) for s in hw) / len(hw)


def _log_iteration(
    arm: str,
    it: int,
    prompts: list[str],
    completions: list[str],
    results: list[RewardResult],
    use_fidelity: bool,
) -> None:
    run = _run_dir(arm)
    # Per-rollout lines (one JSON object per rollout).
    with (run / "rollouts.jsonl").open("a") as f:
        for p, c, r in zip(prompts, completions, results, strict=True):
            f.write(json.dumps({
                "iter": it,
                "ts": time.time(),
                "total": r.total,
                "components": r.components,
                "word_ratio": _word_ratio(p, c),
                "hard_rate": _hard_rate(r),
                "debug": r.debug,
            }) + "\n")

    totals = [r.total for r in results]
    summary = {
        "iter": it,
        "ts": time.time(),
        "n": len(results),
        "reward_mean": statistics.mean(totals) if totals else 0.0,
        "reward_std": statistics.pstdev(totals) if len(totals) > 1 else 0.0,
        "word_ratio_mean": statistics.mean(
            _word_ratio(p, c) for p, c in zip(prompts, completions, strict=True)
        ) if completions else 0.0,
        "hard_rate_mean": statistics.mean(_hard_rate(r) for r in results) if results else 0.0,
    }
    for comp in ("level_band", "vocab", "fidelity", "gates"):
        vals = [r.components.get(comp, 0.0) for r in results]
        summary[f"{comp}_mean"] = statistics.mean(vals) if vals else 0.0
        summary[f"{comp}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    if use_fidelity:
        n_flag = sum(1 for r in results if (r.debug.get("fidelity") or {}).get("n_unsupported", 0) >= 1)
        summary["halluc_flag_rate"] = n_flag / len(results) if results else 0.0
        scorer = default_scorer()
        summary["judge_calls"] = scorer.calls
        summary["judge_failures"] = scorer.failures
        summary["judge_cache_hits"] = scorer.cache_hits
        summary["judge_failure_rate"] = scorer.failures / max(scorer.calls, 1)

    with (run / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")

    # Headline halt check (§6): mean AND std flat at ~0 from iter 1 means the
    # cardinal design failed its core purpose — surface it loudly.
    if it == 0 and summary["reward_std"] < 1e-6 and summary["reward_mean"] < 1e-6:
        _log.warning(
            "v10 %s: reward mean AND std ~0 at iter 0 — group advantage is dead; "
            "inspect runs/%s/rollouts.jsonl before continuing.", arm, arm
        )


def _score_batch(prompts, completions, *, use_fidelity: bool, arm: str) -> list[float]:
    band = default_band()
    scorer = default_scorer() if use_fidelity else None
    n = len(completions)
    results: list[RewardResult | None] = [None] * n

    def work(i: int):
        return i, compute_reward(
            prompts[i], completions[i], band=band, scorer=scorer, use_fidelity=use_fidelity
        )

    if use_fidelity and _CONCURRENCY > 1 and n > 1:
        with ThreadPoolExecutor(max_workers=_CONCURRENCY) as ex:
            for i, res in ex.map(work, range(n)):
                results[i] = res
    else:
        for i in range(n):
            _, results[i] = work(i)

    it = _iter_count[arm]
    _iter_count[arm] = it + 1
    _log_iteration(arm, it, list(prompts), list(completions), results, use_fidelity)
    return [r.total for r in results]


@register_reward_function()
def v10_full_reward(prompts, completions, answer, types=None) -> list[float]:
    """Arm A: level_band x vocab x fidelity x gates. One judge call/rollout."""
    return _score_batch(prompts, completions, use_fidelity=True, arm="full")


@register_reward_function()
def v10_nojudge_reward(prompts, completions, answer, types=None) -> list[float]:
    """Arm B: level_band x vocab x gates. No judge call (free)."""
    return _score_batch(prompts, completions, use_fidelity=False, arm="nojudge")
