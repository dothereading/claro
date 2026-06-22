"""Q3 judge-swap: re-score stored SFT + GRPO outputs with a DIFFERENT judge.

The headline faithfulness result was produced with the same Haiku judge the
model trained against. This re-scores the identical outputs with a disjoint
judge family (DeepSeek) using the identical fidelity prompt. We don't expect
absolute rates to match; we ask whether the SFT-vs-GRPO *gap* survives the
judge swap. If it does, "the model just learned to fool Haiku" is mostly dead.

Run:
    OPENROUTER_API_KEY=... uv run python scripts/rescore_judge_swap.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from langsimp.verifier import LocalJudge  # noqa: E402
from reward.c3_fidelity import (  # noqa: E402
    _PROMPT_PATH,
    FIDELITY_RESPONSE_FORMAT,
    FidelityScorer,
)

# Disjoint-family judge for the circularity check. DeepSeek (pinned OR unpinned)
# fails open ~80% on this structured fidelity prompt — it returns null content
# and malformed JSON regardless of provider. A GPT/Gemini judge honors the
# json_schema strict response_format (0% parse failures, like Haiku) AND is a
# different family from the Anthropic judge the model trained against, so it
# actually tests "did we just learn to fool Haiku?".
SWAP_JUDGE = os.environ.get("SWAP_JUDGE", "openai/gpt-4o")


def build_swap_scorer() -> FidelityScorer:
    judge = LocalJudge(
        base_url="https://openrouter.ai/api/v1",
        model_name=SWAP_JUDGE,
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0.0,
        # Generous budget: thinking-capable judges (Gemini 3.x) spend reasoning
        # tokens before the JSON, and a tight cap truncates it ("Unterminated
        # string"). 4000 leaves room for the verdict to close.
        max_tokens=int(os.environ.get("SWAP_MAX_TOKENS", "4000")),
        response_format=FIDELITY_RESPONSE_FORMAT,
    )
    # Fresh cache namespace (different judge) via model_id in the key.
    return FidelityScorer(judge, SWAP_JUDGE, _PROMPT_PATH.read_text(), "v11-swap", cache=None)


def score_set(scorer, items):
    def one(it):
        src, cand = it
        r = scorer.score(src, cand)
        return r.recall, r.n_unsupported, (1 if r.n_unsupported >= 1 else 0), r.failed
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(one, items))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grpo", default="eval_results/gspo_eval.json",
                    help="GRPO eval JSON (has complex + output)")
    ap.add_argument("--sft", default="eval_results/sft_n750_1b_eval80.json",
                    help="SFT baseline eval JSON (has output)")
    args = ap.parse_args()

    grpo_data = json.loads((ROOT / args.grpo).read_text())["results"]
    g = {r["title"]: r for r in grpo_data}
    s = {r["title"]: r for r in json.loads((ROOT / args.sft).read_text())["results"]}
    src = {r["title"]: r["complex"] for r in grpo_data}
    titles = [t for t in g if t in s and t in src]

    scorer = build_swap_scorer()
    print(f"[swap] judge={SWAP_JUDGE}, {len(titles)} paired items")
    sft_items = [(src[t], s[t]["output"]) for t in titles]
    grpo_items = [(src[t], g[t]["output"]) for t in titles]
    print("[swap] scoring SFT...", flush=True)
    sft = score_set(scorer, sft_items)
    print("[swap] scoring GRPO...", flush=True)
    grpo = score_set(scorer, grpo_items)

    def agg(rows, idx):
        vals = [r[idx] for r in rows if not r[3]]
        return statistics.mean(vals)

    print(f"\n=== JUDGE-SWAP RE-SCORE ({SWAP_JUDGE}) — does the SFT→GRPO gap survive? ===")
    print(f"{'metric':22s} {'SFT':>8s} {'GRPO':>8s} {'delta':>8s}")
    for name, idx in [("recall", 0), ("n_unsupported", 1), ("halluc_flag_rate", 2)]:
        sv, gv = agg(sft, idx), agg(grpo, idx)
        print(f"{name:22s} {sv:8.3f} {gv:8.3f} {gv - sv:+8.3f}")
    print(f"\njudge failures: sft={sum(r[3] for r in sft)} grpo={sum(r[3] for r in grpo)}")


if __name__ == "__main__":
    main()
