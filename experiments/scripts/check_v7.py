"""Quick live-judge sanity check for v7.

Picks the Tomorrow Never Dies source + 8 real candidates pulled from the
saved 4B eval files (n=0/250/500/750/1000/1250/1500 plus a synthetic
hallucination), sends them through the live judge with the v7 prompt,
and prints the raw reply, the parsed ranking, and the resulting scores.

Confirms two things end-to-end:
  1. The DeepSeek judge actually returns a parseable JSON array when
     asked for one (vs. wrapping it in prose or a dict).
  2. `SparseRankReward` extracts a valid permutation and the score
     table assigns geometric weights to the top half.

Run:
    MEANING_JUDGE_BACKEND=openrouter \\
    MEANING_JUDGE_MODEL=deepseek/deepseek-v4-pro:gmicloud/fp8 \\
    uv run python scripts/check_v7.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from experiments.rewards.rewards_legacy import _get_judge  # noqa: E402
from experiments.rewards.rewards_v7 import (  # noqa: E402
    _RANK_PROMPT_TEMPLATE,
    SparseRankReward,
    _parse_rank_list,
    _score_ranks_sparse,
)

TND_TITLE = "Tomorrow Never Dies"
EVALS = [
    ("base_4b_eval30.json", "n=0 base"),
    ("sft_n250_4b_eval30.json", "n=250"),
    ("sft_n500_4b_eval30.json", "n=500"),
    ("sft_n750_4b_eval30.json", "n=750"),
    ("sft_n1000_4b_eval30.json", "n=1000"),
    ("sft_n1250_4b_eval30.json", "n=1250"),
    ("sft_n1500_4b_eval30.json", "n=1500"),
]


def _load_candidate(eval_name: str) -> tuple[str, str]:
    """Return (source, output) for TND from a given eval file."""
    d = json.load(open(REPO_ROOT / "eval_results" / eval_name))
    for r in d["results"]:
        if (r.get("title") or "") == TND_TITLE:
            return r["complex"], r["output"]
    raise KeyError(f"{TND_TITLE} not found in {eval_name}")


def main() -> int:
    judge = _get_judge()
    if judge is None:
        print("ERROR: no judge configured. Set MEANING_JUDGE_BACKEND=openrouter.")
        return 2
    print(f"Judge: {judge.model} @ {judge.endpoint}\n")

    source, _ = _load_candidate("base_4b_eval30.json")
    print(f"SOURCE ({len(source.split())}w):\n{source}\n")

    candidates: list[tuple[str, str]] = []  # (label, text)
    for fname, label in EVALS:
        _, out = _load_candidate(fname)
        candidates.append((label, out))
    # 8th candidate: a deliberate hallucination so we can confirm the judge
    # places it at the bottom.
    halluc = (
        candidates[-1][1]
        + " The film was originally directed by Steven Spielberg in 1965 before being remade."
    )
    candidates.append(("HALLUC", halluc))

    g = len(candidates)
    print(f"\nG = {g} candidates:")
    for i, (label, text) in enumerate(candidates):
        print(f"  [{i}] {label:<10} ({len(text.split())}w) {text[:80]}...")

    # ---- run the live judge ----
    candidate_block = "\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(candidates))
    prompt = _RANK_PROMPT_TEMPLATE.format(source=source, candidates=candidate_block)

    print("\n--- sending to judge... ---")
    raw = judge.evaluate(prompt)
    print(f"raw reply type: {type(raw).__name__}")
    print(f"raw reply value: {raw!r}")

    order = _parse_rank_list(raw, g)
    print(f"\nparsed order (best → worst): {order}")
    if order is None:
        print("PARSE FAILED. Inspect the raw reply above.")
        return 1

    print("\nranking by label (best → worst):")
    for rank, rid in enumerate(order):
        print(f"  rank {rank}: [{rid}] {candidates[rid][0]}")

    scores = _score_ranks_sparse(order, g)
    print("\nsparse-geometric scores (by candidate id):")
    for i, (label, _) in enumerate(candidates):
        print(f"  [{i}] {label:<10} → {scores[i]:.4f}")

    # Sanity asserts
    halluc_idx = g - 1
    halluc_rank = order.index(halluc_idx)
    print(f"\nhallucination ranked: {halluc_rank} (out of {g - 1}); "
          f"score = {scores[halluc_idx]:.4f}")
    if halluc_rank < g // 2:
        print("  ⚠️  hallucination landed in the TOP HALF — judge variance is high.")
    else:
        print("  ✓ hallucination in the bottom half (expected).")

    # Also smoke-test through the full reward class
    print("\n--- running through SparseRankReward.compute_group ---")
    r = SparseRankReward()
    scores2 = r.compute_group(source, [c[1] for c in candidates], judge=judge)
    print(f"scores: {[f'{s:.4f}' for s in scores2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
