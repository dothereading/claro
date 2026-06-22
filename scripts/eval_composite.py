"""Composite readout for a stored eval JSON — the single command PLAN_4B Phase 0
defines and Phases 1/3 reuse.

Given an eval file with per-item `complex` (source) + `output` (candidate) and
optionally `level` (a stored difficulty vote), it scores every item through the
full v11.1 `reward()` composition. That one call yields, cached:

- the free deterministic signals: `level_band`, `vocab` components;
- the Haiku fidelity judge: recall, n_unsupported, halluc-flag.

It prints the composite the plan watches: pct_{A1,A2,B1} (from stored votes,
relative-comparison only), halluc-flag rate, mean recall, and the headline
**faithful-A2 rate** = fraction of items that are A2 (stored vote) AND have zero
unsupported claims. The fidelity judge is cached (SQLite), so re-runs and the
overlapping checkpoint items are free.

    OPENROUTER_API_KEY=... uv run python scripts/eval_composite.py \
        eval_results/sft_n750_4b_eval30.json [more.json ...]
    # --no-fidelity for the free level_band-only pass (no judge calls)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from claro.reward.compose import default_band, default_scorer  # noqa: E402
from claro.reward.compose import reward as compute_reward  # noqa: E402


@dataclass
class ItemScore:
    level: str | None
    level_band: float
    vocab: float
    fidelity: float
    recall: float | None
    n_unsupported: int | None
    failed: bool


def _items(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["results"] if isinstance(data, dict) and "results" in data else data


def score_file(path: Path, *, use_fidelity: bool, workers: int,
               deepseek_level: bool = False) -> list[ItemScore]:
    band = default_band()
    scorer = default_scorer() if use_fidelity else None
    rows = _items(path)

    # De-noise the difficulty signal: re-classify stored outputs with the
    # DeepSeek mode-of-3 classifier (no generation). Off by default — the
    # plan reserves this for finalist SFT starts / checkpoints, not everything.
    levels = [r.get("level") for r in rows]
    if deepseek_level:
        from scripts.eval_difficulty_paired import classify_all  # heavy import; lazy
        levels = classify_all([r["output"] for r in rows], votes=3)

    def one(r: dict, lvl: str | None) -> ItemScore:
        res = compute_reward(r["complex"], r["output"], band=band, scorer=scorer,
                             use_fidelity=use_fidelity)
        fdbg = res.debug.get("fidelity", {})
        return ItemScore(
            level=lvl,
            level_band=res.components["level_band"],
            vocab=res.components["vocab"],
            fidelity=res.components["fidelity"],
            recall=fdbg.get("recall"),
            n_unsupported=fdbg.get("n_unsupported"),
            failed=bool(fdbg.get("failed", False)),
        )

    if workers > 1 and len(rows) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, rows, levels))
    return [one(r, lvl) for r, lvl in zip(rows, levels, strict=True)]


def summarize(name: str, scores: list[ItemScore], *, use_fidelity: bool) -> dict:
    n = len(scores)
    levels = [s.level for s in scores if s.level]
    def pct(k: str) -> float:
        return 100 * sum(1 for x in levels if x == k) / len(levels) if levels else float("nan")

    out = {
        "name": name, "n": n,
        "pct_A1": pct("A1"), "pct_A2": pct("A2"), "pct_B1": pct("B1"),
        "level_band_mean": statistics.mean(s.level_band for s in scores),
        "vocab_mean": statistics.mean(s.vocab for s in scores),
    }
    if use_fidelity:
        ok = [s for s in scores if not s.failed]
        out["recall_mean"] = statistics.mean(s.recall for s in ok if s.recall is not None) if ok else float("nan")
        out["halluc_flag_rate"] = sum(1 for s in ok if (s.n_unsupported or 0) >= 1) / len(ok) if ok else float("nan")
        out["n_unsupported_mean"] = statistics.mean(s.n_unsupported for s in ok if s.n_unsupported is not None) if ok else float("nan")
        # headline: A2 (stored vote) AND zero unsupported claims
        faithful_a2 = [s for s in scores if s.level == "A2" and not s.failed and (s.n_unsupported or 0) == 0]
        out["faithful_A2_rate"] = 100 * len(faithful_a2) / n if n else float("nan")
        out["judge_failures"] = sum(1 for s in scores if s.failed)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", help="eval JSON files (complex + output [+ level])")
    ap.add_argument("--no-fidelity", action="store_true", help="free level_band pass, no judge")
    ap.add_argument("--deepseek-level", action="store_true",
                    help="re-classify stored outputs with DeepSeek mode-of-3 (finalists only)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    use_fidelity = not args.no_fidelity

    summaries = []
    for f in args.files:
        path = (ROOT / f) if not Path(f).is_absolute() else Path(f)
        scores = score_file(path, use_fidelity=use_fidelity, workers=args.workers,
                            deepseek_level=args.deepseek_level)
        summaries.append(summarize(path.stem, scores, use_fidelity=use_fidelity))

    cols = ["name", "n", "pct_A1", "pct_A2", "pct_B1", "level_band_mean", "vocab_mean"]
    if use_fidelity:
        cols += ["recall_mean", "halluc_flag_rate", "n_unsupported_mean", "faithful_A2_rate", "judge_failures"]
    widths = {c: max(len(c), 22 if c == "name" else 9) for c in cols}
    print(" ".join(f"{c:>{widths[c]}}" for c in cols))
    for s in summaries:
        cells = []
        for c in cols:
            v = s.get(c)
            if isinstance(v, float):
                cells.append(f"{v:>{widths[c]}.3f}")
            else:
                cells.append(f"{str(v):>{widths[c]}}")
        print(" ".join(cells))

    if args.json_out:
        args.json_out.write_text(json.dumps(summaries, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
