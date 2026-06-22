"""Cardinal CEFR-A2 reward as an mlx-lm-lora reward function (the shipped reward).

Same composition as v10 (`level_band × vocab × fidelity × gates`) but with the
v11.1 changes baked into the components: a syntactic band (passive +
subordination) inside level_band, the demoted vocab density penalty + gloss
matcher, and tiered core/peripheral fidelity recall. This module is the thin
trainer adapter; the maths live in the `reward` package.

It adds the §5/CHANGE-5 readouts to runs/gspo/metrics.jsonl: per-iteration mean
passive / subordination (must trend toward the A2 band, NOT to zero), gloss
rate (glossed vs kept-bare source terms — the central behavioral signal),
core-fact recall, and the level_band component std (must be > 0 now that it
carries the active A2 defense).

    --reward-functions cefr_a2_reward --reward-functions-file langsimp/training/rewards.py
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langsimp.training.registry import register_reward_function
from reward.compose import RewardResult, default_band, default_scorer
from reward.compose import reward as compute_reward
from reward.fidelity import load_fidelity_config
from reward.level_band import syntactic_features

_log = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[2]

try:
    _CONCURRENCY = int(load_fidelity_config().get("concurrency", 4))
except Exception:  # pragma: no cover
    _CONCURRENCY = 4

_iter_count = {"a2": 0}


def _run_dir() -> Path:
    d = _ROOT / "runs" / "gspo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _word_ratio(source: str, candidate: str) -> float:
    s = len(source.split())
    return len(candidate.split()) / s if s else 0.0


def _log_iteration(it, prompts, completions, results) -> None:
    run = _run_dir()
    rows = []
    for p, c, r in zip(prompts, completions, results, strict=True):
        passive, subord, _ = syntactic_features(c)
        vdbg = r.debug.get("vocab", {})
        fdbg = r.debug.get("fidelity", {})
        rows.append({
            "total": r.total, "components": r.components,
            "passive": passive, "subord": subord,
            "n_glossed": len(vdbg.get("glossed", [])),
            "n_bare_source": len(vdbg.get("bare_source", [])),
            "core_recall": fdbg.get("recall"),
            "n_unsupported": fdbg.get("n_unsupported"),
            "word_ratio": _word_ratio(p, c),
        })
    with (run / "rollouts.jsonl").open("a") as f:
        for r, extra in zip(results, rows, strict=True):
            f.write(json.dumps({"iter": it, "ts": time.time(), **extra,
                                "debug": r.debug}) + "\n")

    totals = [r.total for r in results]
    glossed = sum(x["n_glossed"] for x in rows)
    bare = sum(x["n_bare_source"] for x in rows)
    recalls = [x["core_recall"] for x in rows if x["core_recall"] is not None]
    summary = {
        "iter": it, "ts": time.time(), "n": len(results),
        "reward_mean": statistics.mean(totals) if totals else 0.0,
        "reward_std": statistics.pstdev(totals) if len(totals) > 1 else 0.0,
        "passive_mean": statistics.mean(x["passive"] for x in rows) if rows else 0.0,
        "subord_mean": statistics.mean(x["subord"] for x in rows) if rows else 0.0,
        # gloss rate: of off-list source terms the model TOUCHED, how many it glossed
        # rather than kept bare. Rising = learning the intended escape hatch.
        "gloss_rate": glossed / (glossed + bare) if (glossed + bare) else None,
        "core_recall_mean": statistics.mean(recalls) if recalls else None,
        "halluc_flag_rate": sum(1 for x in rows if (x["n_unsupported"] or 0) >= 1) / len(rows) if rows else 0.0,
        "word_ratio_mean": statistics.mean(x["word_ratio"] for x in rows) if rows else 0.0,
    }
    for comp in ("level_band", "vocab", "fidelity", "gates"):
        vals = [r.components.get(comp, 0.0) for r in results]
        summary[f"{comp}_mean"] = statistics.mean(vals) if vals else 0.0
        summary[f"{comp}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    scorer = default_scorer()
    summary["judge_failures"] = scorer.failures
    summary["judge_cache_hits"] = scorer.cache_hits

    with (run / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")

    if it == 0 and summary["level_band_std"] < 1e-6:
        _log.warning("v11: level_band std ~0 at iter 0 — the syntactic A2 defense "
                     "isn't discriminating; inspect runs/gspo/rollouts.jsonl.")


@register_reward_function()
def cefr_a2_reward(prompts, completions, answer, types=None) -> list[float]:
    band = default_band()
    scorer = default_scorer()
    n = len(completions)
    results: list[RewardResult | None] = [None] * n

    def work(i):
        return i, compute_reward(prompts[i], completions[i], band=band, scorer=scorer,
                                 use_fidelity=True)

    if _CONCURRENCY > 1 and n > 1:
        with ThreadPoolExecutor(max_workers=_CONCURRENCY) as ex:
            for i, res in ex.map(work, range(n)):
                results[i] = res
    else:
        for i in range(n):
            _, results[i] = work(i)

    it = _iter_count["a2"]
    _iter_count["a2"] = it + 1
    _log_iteration(it, list(prompts), list(completions), results)
    return [r.total for r in results]
