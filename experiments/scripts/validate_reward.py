"""§5 offline validation harness for the v10 reward. BUILD/RUN THIS FIRST.

Scores the stored eval generations for SFT_n750, GRPO v7, and GRPO v9 on the
50 held-out paragraphs with the FULL reward (Arm A), then checks the reward
reproduces known-correct orderings. Hard checks assert and exit nonzero on
failure; soft diagnostics only print.

Hard checks (must pass before training):
  1. Melicope:            reward(GRPO) > reward(SFT)   — SFT invented stamens.
  2. Tomorrow Never Dies: reward(GRPO) > reward(SFT)   — SFT hallucinated a clause.
  3. Ali Darassa:         reward(v9)  < reward(v7)      — v9 invented two countries.
  4. No brevity bias:     Spearman(reward, word_count) > -0.4 across all outputs.

Budget ~150 judge calls (50 paragraphs x 3 models); the SQLite cache makes
re-runs free. Needs OPENROUTER_API_KEY.

Run:
    OPENROUTER_API_KEY=... uv run python scripts/validate_reward.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from reward.compose import default_band, reward  # noqa: E402
from reward.c3_fidelity import build_scorer, load_fidelity_config  # noqa: E402

EVAL_FILES = {
    "sft": ROOT / "eval_results" / "sft_n750_1b_eval80.json",
    "v7": ROOT / "eval_results" / "grpo_v7_eval80.json",
    "v9": ROOT / "eval_results" / "grpo_v9_eval80.json",
}
LOG_PATH = ROOT / "runs" / "validate_reward.jsonl"


# ---------- small stats helpers (no scipy) ----------


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average rank for ties (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((r - ma) ** 2 for r in ra) ** 0.5
    vb = sum((r - mb) ** 2 for r in rb) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def bootstrap_ci(xs: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    import random

    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [xs[rng.randrange(len(xs))] for _ in xs]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, hi


# ---------- scoring ----------


def load_models() -> dict[str, dict[str, dict]]:
    models = {}
    for name, path in EVAL_FILES.items():
        models[name] = {r["title"]: r for r in json.loads(path.read_text())["results"]}
    return models


def find_title(titles, needle: str) -> str:
    hits = [t for t in titles if needle.lower() in t.lower()]
    if not hits:
        raise KeyError(f"no eval record matching {needle!r}")
    return hits[0]


def score_all(models, scorer, band, concurrency: int) -> dict[str, dict[str, dict]]:
    """reward.total + components for every (model, title). Returns
    scores[model][title] = {"total":.., "components":.., "debug":.., "words":..}"""
    band_obj = band
    jobs = []
    for model, recs in models.items():
        for title, r in recs.items():
            jobs.append((model, title, r))

    def run(job):
        model, title, r = job
        res = reward(r["complex"], r["output"], band=band_obj, scorer=scorer)
        return model, title, res, r

    scores: dict[str, dict[str, dict]] = {m: {} for m in models}
    log_lines = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for model, title, res, r in ex.map(run, jobs):
            entry = {
                "total": res.total,
                "components": res.components,
                "debug": res.debug,
                "words": r["output_words"],
            }
            scores[model][title] = entry
            log_lines.append(json.dumps({"model": model, "title": title, **entry}))
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(log_lines) + "\n")
    return scores


# ---------- checks ----------


def hard_checks(models, scores) -> list[tuple[str, bool, str]]:
    titles = set(models["sft"])
    out = []

    mel = find_title(titles, "Melicope")
    ok1 = scores["v9"][mel]["total"] > scores["sft"][mel]["total"]
    out.append(("1. Melicope: v9 > sft", ok1,
                f"v9={scores['v9'][mel]['total']:.4f} v7={scores['v7'][mel]['total']:.4f} sft={scores['sft'][mel]['total']:.4f}"))

    tnd = find_title(titles, "Tomorrow Never Dies")
    ok2 = scores["v9"][tnd]["total"] > scores["sft"][tnd]["total"]
    out.append(("2. Tomorrow Never Dies: v9 > sft", ok2,
                f"v9={scores['v9'][tnd]['total']:.4f} v7={scores['v7'][tnd]['total']:.4f} sft={scores['sft'][tnd]['total']:.4f}"))

    ali = find_title(titles, "Darassa")
    ok3 = scores["v9"][ali]["total"] < scores["v7"][ali]["total"]
    out.append(("3. Ali Darassa: v9 < v7", ok3,
                f"v9={scores['v9'][ali]['total']:.4f} v7={scores['v7'][ali]['total']:.4f}"))

    all_rewards, all_words = [], []
    for model in scores:
        for title in scores[model]:
            all_rewards.append(scores[model][title]["total"])
            all_words.append(scores[model][title]["words"])
    rho = spearman(all_rewards, all_words)
    ok4 = rho > -0.4
    out.append(("4. No brevity bias: spearman(reward, words) > -0.4", ok4, f"rho={rho:.3f}"))
    return out


def soft_diagnostics(models, scores, scorer) -> None:
    print("\n=== SOFT DIAGNOSTICS ===")

    # 5. per-component dynamic range + drowned-component flag
    comps = ["level_band", "vocab", "fidelity", "gates"]
    print("\n[5] per-component range (min / mean / max), IQR:")
    iqrs = {}
    for c in comps:
        vals = [scores[m][t]["components"][c] for m in scores for t in scores[m]]
        q = statistics.quantiles(vals, n=4) if len(vals) >= 4 else [min(vals), 0, max(vals)]
        iqr = q[2] - q[0]
        iqrs[c] = iqr
        print(f"  {c:11s}: {min(vals):.3f} / {statistics.mean(vals):.3f} / {max(vals):.3f}   IQR={iqr:.3f}")
    widest = max(iqrs.values()) or 1.0
    for c, iqr in iqrs.items():
        if iqr < 0.25 * widest:
            print(f"  ⚠️  '{c}' IQR ({iqr:.3f}) < 25% of widest ({widest:.3f}) — "
                  f"drowned; consider rescaling (x ** k) before training.")

    # 6. mean reward per model + bootstrap CI
    print("\n[6] mean reward per model (95% bootstrap CI):")
    for m in scores:
        vals = [scores[m][t]["total"] for t in scores[m]]
        lo, hi = bootstrap_ci(vals)
        print(f"  {m:4s}: mean={statistics.mean(vals):.4f}  CI=[{lo:.4f}, {hi:.4f}]  n={len(vals)}")

    # 7. judge failure + cache hit rate
    print("\n[7] judge:")
    print(f"  calls={scorer.calls}  failures={scorer.failures}  cache_hits={scorer.cache_hits}")
    denom = scorer.calls + scorer.cache_hits
    if denom:
        print(f"  failure_rate={scorer.failures / max(scorer.calls, 1):.3f}  "
              f"cache_hit_rate={scorer.cache_hits / denom:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-cache", action="store_true", help="ignore the verdict cache")
    args = ap.parse_args()

    cfg = load_fidelity_config()
    scorer = build_scorer(use_cache=not args.no_cache, config=cfg)
    band = default_band()
    models = load_models()

    n_titles = len(set.intersection(*[set(m) for m in models.values()]))
    print(f"scoring {sum(len(m) for m in models.values())} outputs "
          f"({n_titles} common titles x {len(models)} models), judge={cfg['model']} ...")

    scores = score_all(models, scorer, band, concurrency=cfg.get("concurrency", 4))

    print("\n=== HARD CHECKS ===")
    results = hard_checks(models, scores)
    all_ok = True
    for label, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}   ({detail})")
        all_ok = all_ok and ok

    soft_diagnostics(models, scores, scorer)

    print(f"\nper-rollout log -> {LOG_PATH}")
    if not all_ok:
        print("\n❌ one or more hard checks failed — iterate the judge prompt before training.")
        sys.exit(1)
    print("\n✅ all hard checks passed.")


if __name__ == "__main__":
    main()
