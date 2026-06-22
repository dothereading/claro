"""Paired A2 difficulty-classifier eval: adapter vs a stored baseline.

The project's headline metric is the DeepSeek difficulty classifier's A2
hit-rate. This runs it PAIRED on the same held-out paragraphs for a trained
adapter (greedy generations, persisted) and a stored baseline eval JSON,
using the stabilized protocol: temp-0 judge, mode of N votes, same judge
model and CEFR anchors as the baseline.

Run:
    OPENROUTER_API_KEY=... uv run python scripts/eval_difficulty_paired.py \
        --adapter adapters/gspo \
        --baseline eval_results/sft_n750_1b_eval80.json \
        --out eval_results/gspo_eval.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from langsimp.inference.engine import load_model_with_adapter, make_generate_fn  # noqa: E402
from langsimp.verifier import DifficultyRankingTest, LocalJudge  # noqa: E402

JUDGE_MODEL = "deepseek/deepseek-v4-pro:gmicloud/fp8"


def _load_anchors() -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {"A1": [], "A2": [], "B1": []}
    for line in (ROOT / "samples.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["level"] in buckets:
            buckets[r["level"]].append(r["text"])
    return buckets


def classify_mode(test: DifficultyRankingTest, judge: LocalJudge, text: str, votes: int) -> str:
    labels = [test.classify(text, judge) for _ in range(votes)]
    return collections.Counter(labels).most_common(1)[0][0]


def classify_all(texts: list[str], votes: int, workers: int = 8) -> list[str]:
    samples = _load_anchors()
    judge = LocalJudge(
        base_url="https://openrouter.ai/api/v1",
        model_name=JUDGE_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0.0,
    )
    test = DifficultyRankingTest(
        a1_samples=samples["A1"], b1_samples=samples["B1"], a2_samples=samples["A2"], n_words=100
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda t: classify_mode(test, judge, t, votes), texts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--baseline", required=True, help="stored eval JSON to compare against")
    ap.add_argument("--out", required=True, help="where to persist adapter generations + labels")
    ap.add_argument("--model", default="mlx-community/gemma-3-1b-it-bf16")
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    base = {r["title"]: r for r in json.loads(Path(args.baseline).read_text())["results"]}
    paras = [json.loads(s) for s in (ROOT / "data" / "eval.jsonl").read_text().splitlines() if s.strip()]
    paras = [p for p in paras if p["title"] in base]  # same items as baseline
    print(f"[eval] {len(paras)} paired paragraphs; judge={JUDGE_MODEL} votes={args.votes}")

    # 1) generate adapter outputs (greedy), persist them.
    model, tok = load_model_with_adapter(args.model, args.adapter)
    gen = make_generate_fn(model, tok, max_tokens=args.max_tokens, temp=0.0)
    adapter_out = []
    for i, p in enumerate(paras):
        o = gen(p["complex"])
        adapter_out.append(o)
        print(f"  [{i + 1}/{len(paras)}] {p['title'][:40]}", flush=True)

    # 2) classify both, same protocol.
    print("[eval] classifying adapter outputs...", flush=True)
    adapter_lvl = classify_all(adapter_out, args.votes)
    print("[eval] classifying baseline outputs...", flush=True)
    base_out = [base[p["title"]]["output"] for p in paras]
    base_lvl = classify_all(base_out, args.votes)

    def pct_a2(levels):
        return sum(lv == "A2" for lv in levels) / len(levels)

    def counts(levels):
        return dict(collections.Counter(levels))

    # paired flip matrix (baseline -> adapter)
    flips = collections.Counter(zip(base_lvl, adapter_lvl, strict=True))

    print(f"\n=== PAIRED A2 DIFFICULTY EVAL (same {len(paras)} items, mode-of-{args.votes}, temp 0) ===")
    print(f"baseline ({Path(args.baseline).name}): A2={pct_a2(base_lvl):.1%}  {counts(base_lvl)}")
    print(f"adapter  ({args.adapter}): A2={pct_a2(adapter_lvl):.1%}  {counts(adapter_lvl)}")
    pairs = list(zip(base_lvl, adapter_lvl, strict=True))
    a2_to_nona2 = sum(1 for b, a in pairs if b == "A2" and a != "A2")
    nona2_to_a2 = sum(1 for b, a in pairs if b != "A2" and a == "A2")
    print(f"\npaired: baseline-A2 that left A2: {a2_to_nona2}   non-A2 that became A2: {nona2_to_a2}")
    print("flip matrix (baseline -> adapter):")
    for (b, a), n in sorted(flips.items(), key=lambda kv: -kv[1]):
        mark = "" if b == a else "  <-- changed"
        print(f"  {b:>4} -> {a:<4}: {n}{mark}")

    # persist adapter generations + labels
    results = [
        {"title": p["title"], "complex": p["complex"], "output": o, "level": lv,
         "baseline_level": bl}
        for p, o, lv, bl in zip(paras, adapter_out, adapter_lvl, base_lvl, strict=True)
    ]
    out = {
        "meta": {"adapter": args.adapter, "judge_model": JUDGE_MODEL, "votes": args.votes,
                 "baseline": args.baseline, "ts": time.time()},
        "summary": {"count": len(paras), "pct_a2_adapter": pct_a2(adapter_lvl),
                    "pct_a2_baseline": pct_a2(base_lvl),
                    "levels_adapter": counts(adapter_lvl), "levels_baseline": counts(base_lvl)},
        "results": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
