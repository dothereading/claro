"""v7 GRPO reward: sparse-geometric ranking, no fidelity, no per-rollout judge call.

One judge call per group of G rollouts. The judge sees the source paragraph
plus the G candidates and emits a JSON array of candidate IDs ranked best
to worst, e.g. `[3, 0, 5, 1, 7, 2, 4, 6]` for G=8. No prose, no error
breakdown, no per-axis scores — just the permutation.

The rank → reward mapping is sparse and skewed: top-floor(G/2) ranks get
geometric scores (`base**rank` with `base=0.5`); the rest get 0. After GRPO
standardizes `(reward - mean) / std`, the winner is ~+1.7σ, 2nd ~+0.5σ,
4th ~−0.3σ, and the bottom half is ~−0.5σ each. This matches what a judge
can actually tell you: top-rank identity is signal, bottom-rank ordering
is noise.

Gates (markdown, loop) and the Gaussian length factor are kept as a
multiplicative mask so malformed rollouts can't "win" a degenerate batch.

Combine with `--reward-functions v7_sparse_rank_reward --reward-weights [1.0]`.
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Any

from langsimp.training.rewards import (
    _get_judge,
    _v5_has_loop,
    _v5_has_markdown,
    register_reward_function,
)
from langsimp.training.rewards_v6 import length_factor
from langsimp.verifier import BaseJudge

_log = logging.getLogger(__name__)


# ---------- Judge prompt ----------

_RANK_PROMPT_TEMPLATE = """You are ranking CEFR A2 simplifications of an English source paragraph.

The best simplification gives an A2 learner everything the source says, in words and sentences they can read. Faithful and accessible matter equally — neither is worth sacrificing for the other.

Apply these criteria, in order of importance:

1. FAITHFULNESS first. Every important noun, number, and relation in the source must survive in the output. Softening a specific term into a generic one (a named person, place, organization, work, or event becoming a category word) is a fidelity loss even when the output reads smoothly. A candidate that invents facts not in the source — or that contradicts what the source actually says — is the worst possible defect; it ranks below any candidate that merely simplifies clumsily.

2. ACCESSIBILITY. Short, simple sentences. Common everyday words (about the 1500 most frequent English words).

Don't over-pack: a sentence that crams two or three specific terms and a date together is hard for an A2 reader, even when every fact is correct. Break dense information across multiple short sentences. Prefer "X is a kind of Y. It was made in 1934. A musician named Z designed it." over "X, a kind of Y made in 1934, was designed by musician Z."

When introducing an unfamiliar term, briefly give context drawn from elsewhere in the same source paragraph — enough that an A2 reader can guess the term from surrounding words. Do not import outside knowledge. Adding biographical, geographic, or relational detail that the source does not contain is invention, even if the added detail is true in the wider world. Rank candidates that import outside facts below candidates that leave the term unexplained.

All else equal, prefer outputs with natural rhythm — varied sentence beginnings and lengths — over robotic, formulaic prose ("X is Y. X did Z. X has W."). Stylistic variety keeps the prose readable.

3. CLEAN OUTPUT. Plain prose only. Markdown markers (asterisks, headings, bullets), chatbot preambles like "Here is the rewrite:" or "Sure! Here is...", and redundant trailing sentences that re-state the main point are all defects — rank such candidates lower than candidates without them.

Length: aim for length comparable to the source. Slight growth is fine when it serves readability — breaking one dense sentence into two shorter ones, or adding a brief in-source context clue for a hard term. Slight shrinkage when faithful is also fine. Padding with filler clauses, or large drops that lose source information, are both bad.

SOURCE PARAGRAPH:
{source}

CANDIDATES:
{candidates}

Your entire reply must be EXACTLY a JSON array of candidate IDs ordered best-first — no prose, no preamble, no trailing explanation, no markdown code fences, no leading whitespace. The reply must be parseable by Python's json.loads(). Example (for 8 candidates): [3, 0, 5, 1, 7, 2, 4, 6]"""

# Substring used by tests / introspection to confirm we sent the v7 prompt.
_RANK_PROMPT_MARKER = "You are ranking CEFR A2 simplifications"


# ---------- Sparse-geometric rank → score ----------


def _score_ranks_sparse(
    order: list[int],
    g: int,
    base: float = 0.5,
    k: int | None = None,
) -> list[float]:
    """Map a permutation of [0..G-1] to per-rollout sparse-geometric scores.

    The top `k` ranks get `base**rank` (so rank 0 = 1.0, rank 1 = base,
    rank 2 = base**2, ...). All remaining rollouts get 0.0.

    Defaults: `k = max(1, G // 2)` (top half), `base = 0.5` (each tier is
    worth half the previous).
    """
    if g == 1:
        return [1.0]
    if k is None:
        k = max(1, g // 2)
    scores = [0.0] * g
    for rank, rid in enumerate(order):
        if rank < k:
            scores[rid] = base ** rank
    return scores


# ---------- Judge reply parsing ----------


def _parse_rank_list(result: Any, g: int) -> list[int] | None:
    """Validate and extract a length-G permutation from the judge reply.

    Accepts:
        * `list[int]`               — what `json.loads("[3, 0, 5, ...]")` returns
        * `dict` with key `"order"` — backward-compat with v6's schema
        * `dict` containing an `"error"` value with digits we can recover

    Returns the permutation on success; `None` to signal "fall back to
    all-zero / all-equal" so GRPO can keep training when the judge misfires.
    """
    candidate: Any = result
    if isinstance(result, dict):
        # Either v6-style {"order": [...]} or an {"error": "...stuff [5,3,...]..."}
        # we can pluck digits out of.
        if "order" in result:
            candidate = result["order"]
        elif "error" in result and isinstance(result["error"], str):
            candidate = result["error"]
        else:
            return None

    if isinstance(candidate, str):
        nums = re.findall(r"\d+", candidate)
        if len(nums) != g:
            return None
        try:
            order = [int(x) for x in nums]
        except ValueError:
            return None
    elif isinstance(candidate, list):
        if len(candidate) != g:
            return None
        try:
            order = [int(x) for x in candidate]
        except (TypeError, ValueError):
            return None
    else:
        return None

    if sorted(order) != list(range(g)):
        return None
    return order


# ---------- Per-group ranker ----------


class SparseRankReward:
    """One judge call per group. Returns sparse-geometric rank scores.

    Doesn't fit `RewardComponent.compute` (no single-rollout meaning);
    callers use `compute_group(source, outputs, judge)`.

    On any malformed reply, returns all-zero so the group's reward
    standard deviation is 0 — callers should detect that and skip the
    GRPO step rather than divide by zero.
    """

    name = "sparse_rank"

    def __init__(self, base: float = 0.5, k: int | None = None):
        self.base = base
        self.k = k

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
            return [0.0] * g

        candidates = "\n".join(f"[{i}] {o}" for i, o in enumerate(outputs))
        prompt = _RANK_PROMPT_TEMPLATE.format(source=source, candidates=candidates)
        try:
            result = judge.evaluate(prompt)
        except Exception as e:
            _log.warning("v7 rank judge call failed: %s", e)
            return [0.0] * g

        order = _parse_rank_list(result, g)
        if order is None:
            warnings.warn(
                f"SparseRankReward: malformed judge reply, using all-zero fallback "
                f"({result!r:.120s})"
            )
            return [0.0] * g
        return _score_ranks_sparse(order, g, base=self.base, k=self.k)


# ---------- Combined reward ----------


class CombinedRewardV7:
    """Per rollout: `rank_score * length_factor * gate`. No fidelity term.

    Gates (multiplicative, 0 or 1):
        * no markdown markers
        * no n-gram loop / repeated-sentence pattern

    Gate-failed rollouts still participate in the ranking; gated text
    is usually garbage and the judge will rank it low, but a hard zero
    on top of that prevents a degenerate batch where everything else
    is worse from accidentally rewarding markdown/loops.
    """

    name = "combined_v7"

    def __init__(self, base: float = 0.5, k: int | None = None):
        self.rank = SparseRankReward(base=base, k=k)

    def compute_group(
        self,
        source: str,
        outputs: list[str],
        judge: BaseJudge | None = None,
    ) -> list[float]:
        rank_scores = self.rank.compute_group(source, outputs, judge=judge)
        rewards: list[float] = []
        for output, rs in zip(outputs, rank_scores, strict=True):
            gate = 0.0 if (_v5_has_markdown(output) or _v5_has_loop(output)) else 1.0
            r = rs * length_factor(source, output) * gate
            rewards.append(max(0.0, min(1.0, r)))
        return rewards


def _default_combined_v7() -> CombinedRewardV7:
    return CombinedRewardV7()


# ---------- mlx-lm-lora entry point ----------


def _group_runs(prompts: list[str]) -> list[tuple[int, int]]:
    """Split `prompts` into contiguous runs of identical source.

    Identical to v6's helper; duplicated rather than imported so v7 stays
    self-contained at the entry point.
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


_COMBINED_V7 = _default_combined_v7()


@register_reward_function()
def v7_sparse_rank_reward(prompts, completions, answer, types=None) -> list[float]:
    """v7 stack as a single mlx-lm-lora reward function.

    Use with `--reward-functions v7_sparse_rank_reward --reward-weights [1.0]`.

    Per group of G rollouts sharing a prompt:
      * 1 ranking judge call → sparse-geometric rank scores
      * gate (markdown/loop) and length-factor applied per rollout

    Returns rewards in the same order as `completions`.
    """
    judge = _get_judge()
    rewards: list[float] = [0.0] * len(completions)
    for start, end in _group_runs(prompts):
        source = prompts[start]
        outputs = list(completions[start:end])
        group_rewards = _COMBINED_V7.compute_group(source, outputs, judge=judge)
        for i, r in zip(range(start, end), group_rewards, strict=True):
            rewards[i] = r
    return rewards
