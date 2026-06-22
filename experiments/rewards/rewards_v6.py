"""v6 GRPO rewards: 8-axis fidelity + per-group A2-quality ranking.

Two complementary judge-backed verifiers per training step:

  * FidelityReward (per-rollout, 1 judge call each)
        Atomic-claim decomposition with 8 weighted error categories
        (Guidroz et al., 2025). Replaces v5's pointwise `f`/`h` axes.

  * GroupRankReward (per-group, 1 judge call total)
        Comparative A2-quality judgment across all G rollouts with
        CEFR anchors. Replaces v5's pointwise CEFR bucket score.
        Forces advantage spread even at small G (no "loss=0 forever").

Combined with deterministic hard gates (markdown, loop — reused from v5)
and a smooth Gaussian length factor (preserves gradient on misshapen
rollouts). See V6_SPEC.md for the design contract.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from typing import Any

from experiments.rewards.rewards_legacy import (
    _A1_ANCHOR,
    _A2_ANCHOR,
    _B1_ANCHOR,
    RewardComponent,
    RewardContext,
    _get_judge,
    _v5_has_loop,
    _v5_has_markdown,
    register_reward_function,
)
from claro.verifier import BaseJudge

_log = logging.getLogger(__name__)


# ---------- Judge prompts ----------

_FIDELITY_PROMPT_TEMPLATE = """You are auditing a text simplification for fidelity. You will return ONLY a JSON object.

First, decompose the SOURCE into atomic claims (one fact each). Then check each claim against the OUTPUT.

Then count errors across these categories:

INFORMATION LOSS (claim from source missing or weakened in output):
- missing_full: claim entirely absent (weight 2)
- missing_specificity: claim present but lost specific detail (weight 1)
- missing_nuance: claim present but lost nuance or connotation (weight 2)

INFORMATION GAIN (content in output not in source):
- hallucinated: unfactual claim invented (weight 4)
- off_topic: present but irrelevant tangent (weight 1)

DISTORTION (claim present but altered):
- factuality_distorted: claim present but factually wrong (weight 4)
- fidelity_major: significant fidelity loss (weight 3)
- fidelity_minor: minor wording shift (weight 1)

SOURCE:
{source}

OUTPUT:
{output}

Respond with ONLY this JSON, nothing else:
{{
  "n_source_claims": <int>,
  "missing_full": <int>,
  "missing_specificity": <int>,
  "missing_nuance": <int>,
  "hallucinated": <int>,
  "off_topic": <int>,
  "factuality_distorted": <int>,
  "fidelity_major": <int>,
  "fidelity_minor": <int>
}}"""


_RANK_PROMPT_TEMPLATE = f"""You are ranking text simplifications. The TARGET level is CEFR A2 (Elementary English) while preserving source meaning.

Reference levels (for calibration):

A1 example:
{_A1_ANCHOR}

A2 example (TARGET):
{_A2_ANCHOR}

B1 example:
{_B1_ANCHOR}

SOURCE:
{{source}}

CANDIDATE SIMPLIFICATIONS:
{{candidates}}

Rank candidates from best to worst as A2 simplifications of the source. Favor outputs that:
- Use A2-appropriate vocabulary and sentence structure
- Preserve the source's meaning (rank outputs lower if they invent or distort facts)

Respond with ONLY this JSON object, where "order" is a JSON array of candidate IDs, best first:
{{{{"order": [<id_best>, ..., <id_worst>]}}}}"""


# ---------- 8-axis fidelity ----------

# Weights from Guidroz et al. (2025) — "subjective judgement on the relative
# severity of each type of error." Adopted as-is; recalibrate later if our
# audit suggests otherwise.
_FIDELITY_WEIGHTS: dict[str, int] = {
    "missing_full": 2,
    "missing_specificity": 1,
    "missing_nuance": 2,
    "hallucinated": 4,
    "off_topic": 1,
    "factuality_distorted": 4,
    "fidelity_major": 3,
    "fidelity_minor": 1,
}

# Worst-case-per-claim weight. Used to normalize the error count into a
# [0, 1] score: max_errors = _MAX_CLAIM_WEIGHT * n_source_claims.
_MAX_CLAIM_WEIGHT = 4

# Separate cache from v5's `_judge_cache` — different prompt + schema.
# Bounded FIFO; intra-batch sharing only.
_fidelity_cache: dict[tuple[str, str], float] = {}
_FIDELITY_CACHE_MAX = 1024


def _score_fidelity(result: dict[str, Any]) -> float:
    """Apply the weighted-error formula to a judge response dict.

    Returns 0.5 on a missing/malformed `n_source_claims` (neutral fallback);
    returns 1.0 when the source has no claims to lose.
    """
    raw = result.get("n_source_claims")
    try:
        n_claims = int(raw)
    except (TypeError, ValueError):
        return 0.5
    if n_claims < 0:
        return 0.5
    if n_claims == 0:
        return 1.0

    weighted = 0
    for key, w in _FIDELITY_WEIGHTS.items():
        try:
            weighted += w * int(result.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0.5

    max_errors = _MAX_CLAIM_WEIGHT * n_claims
    return max(0.0, min(1.0, 1.0 - weighted / max_errors))


class FidelityReward(RewardComponent):
    """Per-rollout 8-axis fidelity score in [0, 1].

    One judge call per `(source, output)` pair, cached. Malformed judge
    responses return 0.5 (neutral) rather than 0 so a single bad call
    doesn't zero a whole training step.
    """

    name = "fidelity"

    def compute(self, output: str, ctx: RewardContext, judge: BaseJudge | None = None) -> float:
        if judge is None:
            return 0.5
        key = (ctx.source, output)
        cached = _fidelity_cache.get(key)
        if cached is not None:
            return cached

        prompt = _FIDELITY_PROMPT_TEMPLATE.format(source=ctx.source, output=output)
        try:
            result = judge.evaluate(prompt)
        except Exception as e:  # judge transport failure
            _log.warning("fidelity judge call failed: %s", e)
            return 0.5
        if not isinstance(result, dict) or "error" in result:
            return 0.5

        score = _score_fidelity(result)
        _fidelity_cache[key] = score
        if len(_fidelity_cache) > _FIDELITY_CACHE_MAX:
            _fidelity_cache.pop(next(iter(_fidelity_cache)))
        return score


# ---------- Per-group A2-quality ranking ----------


def _score_ranks(order: list[int], g: int) -> list[float]:
    """Map a permutation of [0..G-1] to per-rollout linear rank scores.

    rank_score_i = (G - rank_i - 1) / (G - 1) — 1.0 for best, 0.0 for worst.
    Special-cases G=1 to return [1.0] (no spread, but reward is well-defined).
    """
    if g == 1:
        return [1.0]
    scores = [0.0] * g
    for rank, rid in enumerate(order):
        scores[rid] = (g - rank - 1) / (g - 1)
    return scores


def _parse_rank_order(result: Any, g: int) -> list[int] | None:
    """Validate the judge's ranking reply. Returns the permutation on
    success, or None to signal "fall back to all-equal."
    """
    if not isinstance(result, dict):
        return None
    order = result.get("order")
    if not isinstance(order, list) or len(order) != g:
        return None
    try:
        order = [int(x) for x in order]
    except (TypeError, ValueError):
        return None
    if sorted(order) != list(range(g)):
        return None
    return order


class GroupRankReward:
    """Per-group A2-quality ranking. One judge call ranks all G rollouts.

    Doesn't fit `RewardComponent.compute` because it needs the full group;
    callers use `compute_group(source, outputs, judge)` and get a list of
    G floats in the same order as `outputs`.

    Robustness: any malformed reply (non-dict, wrong length, duplicate IDs,
    non-integer entries) falls back to all-equal `0.5` so GRPO can keep
    training; we log a warning so the failure is visible.
    """

    name = "rank"

    def compute_group(
        self,
        source: str,
        outputs: list[str],
        judge: BaseJudge | None = None,
    ) -> list[float]:
        g = len(outputs)
        if g == 0:
            return []
        if g == 1:
            return [1.0]
        if judge is None:
            return [0.5] * g

        candidates = "\n".join(f"[{i}] {o}" for i, o in enumerate(outputs))
        prompt = _RANK_PROMPT_TEMPLATE.format(source=source, candidates=candidates)
        try:
            result = judge.evaluate(prompt)
        except Exception as e:
            _log.warning("rank judge call failed: %s", e)
            return [0.5] * g

        order = _parse_rank_order(result, g)
        if order is None:
            warnings.warn(f"GroupRankReward: malformed judge reply, using neutral fallback ({result!r:.120s})")
            return [0.5] * g
        return _score_ranks(order, g)


# ---------- Length factor & combined reward ----------


def length_factor(source: str, output: str, sigma: float = 0.4) -> float:
    """Gaussian centered at output/source word ratio = 1.0.

    Smooth instead of v5's hard cliff: a 4.75× over-long rollout scores
    ~1e-4 (very low but non-zero), so the policy still gets a gradient
    pointing back toward sensible length during cold-start.
    """
    sw = len(source.split())
    if sw == 0:
        return 0.0
    r = len(output.split()) / sw
    return math.exp(-((r - 1.0) ** 2) / (2 * sigma**2))


class CombinedRewardV6:
    """Top-level v6 reward.

    Per rollout:
        base = 0.5 * fidelity + 0.5 * rank
        reward = base * length_factor * gate
        if fidelity < FLOOR and rank > FLOOR_RANK: reward *= ATTENUATE

    Gates (multiplicative, 0 or 1):
        * no markdown markers
        * no n-gram loop / repeated-sentence pattern

    Gate-failed rollouts still participate in the group ranking — gated
    text is usually garbage, so the ranker pushes it to the bottom
    naturally, and GRPO advantage stays alive.
    """

    name = "combined_v6"

    # Soft fidelity floor: if a rollout ranks well but is hallucinating,
    # attenuate the reward rather than zero it — keeps a small gradient
    # pointing toward less-bad in the all-bad-rollout case.
    FIDELITY_FLOOR = 0.2
    FLOOR_RANK_THRESHOLD = 0.5
    FLOOR_ATTENUATION = 0.2

    def __init__(
        self,
        fidelity_weight: float = 0.5,
        rank_weight: float = 0.5,
    ):
        self.fidelity = FidelityReward()
        self.rank = GroupRankReward()
        self.fidelity_weight = fidelity_weight
        self.rank_weight = rank_weight

    def compute_group(
        self,
        source: str,
        outputs: list[str],
        judge: BaseJudge | None = None,
    ) -> list[float]:
        rank_scores = self.rank.compute_group(source, outputs, judge=judge)
        ctx = RewardContext(source=source)
        rewards: list[float] = []
        for output, rs in zip(outputs, rank_scores, strict=True):
            fid = self.fidelity.compute(output, ctx, judge=judge)
            rewards.append(self._combine(source, output, fid, rs))
        return rewards

    def compute(
        self,
        output: str,
        ctx: RewardContext,
        judge: BaseJudge | None = None,
    ) -> float:
        """Offline-audit entry point. Treats the rollout as a group of 1
        (rank_score = 1.0). Useful for `audit_record`-style diagnostics
        where there is no group; do not use during GRPO training (the
        per-group call is what generates advantage spread)."""
        fid = self.fidelity.compute(output, ctx, judge=judge)
        return self._combine(ctx.source, output, fid, rank_score=1.0)

    def _combine(self, source: str, output: str, fidelity: float, rank_score: float) -> float:
        gate = 0.0 if (_v5_has_markdown(output) or _v5_has_loop(output)) else 1.0
        base = self.fidelity_weight * fidelity + self.rank_weight * rank_score
        reward = base * length_factor(source, output) * gate
        if fidelity < self.FIDELITY_FLOOR and rank_score > self.FLOOR_RANK_THRESHOLD:
            reward *= self.FLOOR_ATTENUATION
        return max(0.0, min(1.0, reward))


def _default_combined_v6() -> CombinedRewardV6:
    return CombinedRewardV6(fidelity_weight=0.5, rank_weight=0.5)


# ---------- mlx-lm-lora entry point ----------


def _group_runs(prompts: list[str]) -> list[tuple[int, int]]:
    """Split `prompts` into contiguous runs of identical source.

    mlx-lm-lora calls reward functions with batch_size × G items where the
    first G share `prompts[0..G-1]`, the next G share `prompts[G..2G-1]`,
    etc. Returns [(start, end_exclusive), ...].
    """
    runs: list[tuple[int, int]] = []
    if not prompts:
        return runs
    start = 0
    for i in range(1, len(prompts)):
        if prompts[i] != prompts[start]:
            runs.append((start, i))
            start = i
    runs.append((start, len(prompts)))
    return runs


_COMBINED_V6 = _default_combined_v6()


@register_reward_function()
def v6_combined_reward(prompts, completions, answer, types=None) -> list[float]:
    """v6 stack as a single mlx-lm-lora reward function.

    Use with `--reward-functions v6_combined_reward --reward-weights [1.0]`.

    Per group of G rollouts sharing a prompt:
      * 1 ranking judge call → linear rank scores in [0, 1]
      * G fidelity judge calls (cached per (source, output))
      * combine + gate + length-factor

    Returns rewards in the same order as `completions`.
    """
    judge = _get_judge()
    rewards: list[float] = [0.0] * len(completions)
    for start, end in _group_runs(prompts):
        source = prompts[start]
        outputs = list(completions[start:end])
        group_rewards = _COMBINED_V6.compute_group(source, outputs, judge=judge)
        for i, r in zip(range(start, end), group_rewards, strict=True):
            rewards[i] = r
    return rewards
