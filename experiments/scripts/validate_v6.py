"""V6 reward pre-flight validation (see V6_SPEC.md §"Pre-flight validation").

Three offline checks against a real judge backend. Run BEFORE launching any
v6 GRPO training run.

  1. FidelityReward Spearman ρ vs subjective rank (mean ≥ 0.5 across 10 prompts).
  2. GroupRankReward places a clear hallucination at rank G−1 (3 prompts, G=8).
  3. GroupRankReward top & bottom slots match subjective within ±1 (5 prompts).

Reads audit data from /tmp/reward_audit5.json and /tmp/vocab_validate.json,
with subjective ranks inlined below (sourced from /tmp/v5_test.py).

Run:
    MEANING_JUDGE_BACKEND=openrouter \\
    OPENROUTER_API_KEY=... \\
    uv run python scripts/validate_v6.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from scipy.stats import spearmanr  # noqa: E402

from experiments.rewards.rewards_legacy import RewardContext, _get_judge  # noqa: E402
from experiments.rewards.rewards_v6 import FidelityReward, GroupRankReward  # noqa: E402

# Subjective ranks (1=best, 5=worst). Lifted verbatim from /tmp/v5_test.py
# so this script doesn't depend on a tmp file we don't own.
SUBJECTIVE_ORIG = {
    "Cayan Tower":                {"opus_long": 1, "n1500": 2, "base_long": 3, "n500": 4, "base_short": 5},
    "Collective animal behavior": {"opus_long": 1, "n500": 2, "base_long": 3, "base_short": 4, "n1500": 5},
    "Christina Pagel":            {"opus_long": 1, "n1500": 2, "n500": 3, "base_long": 4, "base_short": 5},
    "Tomorrow Never Dies":        {"opus_long": 1, "n500": 2, "n1500": 3, "base_short": 4, "base_long": 5},
    "Melicope":                   {"opus_long": 1, "n500": 2, "n1500": 3, "base_short": 4, "base_long": 5},
}
SUBJECTIVE_NEW = {
    "Blackledge–Kearney House":     {"opus_long": 1, "base_long": 2, "base_short": 3, "n500": 4, "n1500": 5},
    "Donald Irvine (physician)":    {"opus_long": 1, "base_short": 2, "n1500": 3, "base_long": 4, "n500": 5},
    "ʼPhags-pa script":             {"opus_long": 1, "n1500": 2, "base_short": 3, "base_long": 4, "n500": 5},
    "Ulmus × hollandica 'Smithii'": {"opus_long": 1, "base_short": 2, "n1500": 3, "base_long": 4, "n500": 5},
    "Migrant (album)":              {"opus_long": 1, "base_long": 2, "n1500": 3, "n500": 4, "base_short": 5},
}

VARIANTS = ["base_short", "base_long", "n500", "n1500", "opus_long"]


def load_audit(path: str) -> dict:
    return json.load(open(path))


def variant_text(audit: dict, prompt: str, variant: str) -> str:
    sv = audit[prompt]["variants"][variant]
    return sv["text"] if "text" in sv else sv  # always "text" in our files


# ---------- Check 1: fidelity vs subjective ----------


def check_fidelity_correlation(judge) -> dict:
    """For each prompt, score the 5 variants with FidelityReward, then
    Spearman-correlate against subjective rank. Return per-prompt ρ and
    the across-prompt mean. Threshold: mean ≥ 0.5.

    Pass: returns dict with "passed": True.
    """
    print("\n=== Check 1: FidelityReward Spearman ρ vs subjective ===")
    fidelity = FidelityReward()
    all_rho: list[float] = []
    per_prompt: list[tuple[str, float, list[tuple[str, int, float]]]] = []

    for audit_path, ranks in [
        ("/tmp/reward_audit5.json", SUBJECTIVE_ORIG),
        ("/tmp/vocab_validate.json", SUBJECTIVE_NEW),
    ]:
        audit = load_audit(audit_path)
        for prompt, subj in ranks.items():
            source = audit[prompt]["source"]
            scored: list[tuple[str, int, float]] = []
            for v in VARIANTS:
                text = variant_text(audit, prompt, v)
                fid = fidelity.compute(text, RewardContext(source=source), judge=judge)
                scored.append((v, subj[v], fid))
            # Higher fidelity should mean LOWER subjective rank (rank 1 = best).
            # Spearman expects monotonic relationship; correlate
            # subjective rank (asc, 1=best) with NEGATIVE fidelity (so both
            # go in the same direction → best variant has smallest rank
            # AND smallest -fidelity).
            sub_ranks = [r for _, r, _ in scored]
            neg_fid = [-f for _, _, f in scored]
            rho, _ = spearmanr(sub_ranks, neg_fid)
            all_rho.append(rho)
            per_prompt.append((prompt, rho, scored))
            print(f"  {prompt:<32} ρ={rho:+.3f}  fidelities={[f'{f:.2f}' for _,_,f in scored]}")

    mean_rho = sum(all_rho) / len(all_rho)
    print(f"\n  → mean ρ = {mean_rho:+.3f} (threshold ≥ 0.5)")
    return {"passed": mean_rho >= 0.5, "mean_rho": mean_rho, "per_prompt": per_prompt}


# ---------- Check 2: group ranker places hallucination last ----------

# A clearly fake fact appended to a valid summary. The 8-axis judge should
# tag this as `hallucinated`; the ranker should rank it worst.
_FAKE_FACT = " The work was secretly funded by Christina Pagel in 1750, before being moved underground in 1834."


def build_g8(audit: dict, prompt: str) -> tuple[list[str], int]:
    """5 valid variants + 2 mild paraphrases + 1 hallucination = 8 outputs.
    Returns (outputs, hallucination_index)."""
    base_long_text = variant_text(audit, prompt, "base_long")
    opus_long_text = variant_text(audit, prompt, "opus_long")

    outputs: list[str] = []
    for v in VARIANTS:
        outputs.append(variant_text(audit, prompt, v))
    # 2 mild paraphrases (prefix tweaks, not factually different)
    outputs.append("In short: " + opus_long_text)
    outputs.append(base_long_text + " Overall, that is the main idea.")
    # 1 hallucination — last so we can identify it by index
    outputs.append(opus_long_text + _FAKE_FACT)
    return outputs, len(outputs) - 1


def check_hallucination_ranked_worst(judge) -> dict:
    print("\n=== Check 2: GroupRankReward places hallucination at rank G−1 (G=8) ===")
    audit_orig = load_audit("/tmp/reward_audit5.json")
    ranker = GroupRankReward()
    results: list[tuple[str, int, list[float]]] = []
    passes = 0
    prompts = ["Cayan Tower", "Collective animal behavior", "Christina Pagel"]
    for prompt in prompts:
        source = audit_orig[prompt]["source"]
        outputs, halluc_idx = build_g8(audit_orig, prompt)
        scores = ranker.compute_group(source, outputs, judge=judge)
        # rank 0 = best (highest rank_score = 1.0); G-1 = worst (0.0)
        # We want the hallucinated output (last in outputs) to have the
        # minimum rank_score (i.e., judged worst).
        worst_idx = min(range(len(scores)), key=lambda i: scores[i])
        ok = worst_idx == halluc_idx
        passes += int(ok)
        results.append((prompt, halluc_idx, scores))
        print(f"  {prompt:<32} halluc_idx={halluc_idx}  worst_idx={worst_idx}  {'OK' if ok else 'FAIL'}")
        print(f"    scores: {[f'{s:.2f}' for s in scores]}")
    print(f"\n  → {passes}/{len(prompts)} prompts placed hallucination worst (threshold ≥ 2/3)")
    return {"passed": passes >= 2, "results": results}


# ---------- Check 3: ranker top/bottom match subjective ±1 ----------


def check_rank_matches_subjective(judge) -> dict:
    print("\n=== Check 3: GroupRankReward top & bottom match subjective (±1) ===")
    audit_orig = load_audit("/tmp/reward_audit5.json")
    ranker = GroupRankReward()
    top_hits = 0
    bot_hits = 0
    n = 0
    for prompt, subj in SUBJECTIVE_ORIG.items():
        source = audit_orig[prompt]["source"]
        outputs = [variant_text(audit_orig, prompt, v) for v in VARIANTS]
        scores = ranker.compute_group(source, outputs, judge=judge)
        # judged_best = variant with max rank_score; subj_best = variant with rank 1
        judged_best = VARIANTS[max(range(5), key=lambda i: scores[i])]
        judged_worst = VARIANTS[min(range(5), key=lambda i: scores[i])]
        # ±1 means the judged best is among subjective ranks 1 or 2
        top_ok = subj[judged_best] <= 2
        bot_ok = subj[judged_worst] >= 4
        top_hits += int(top_ok)
        bot_hits += int(bot_ok)
        n += 1
        print(
            f"  {prompt:<32} judged_best={judged_best:<11} (subj rank {subj[judged_best]}) "
            f"{'OK' if top_ok else 'FAIL'}  judged_worst={judged_worst:<11} "
            f"(subj rank {subj[judged_worst]}) {'OK' if bot_ok else 'FAIL'}"
        )
    print(f"\n  → top ±1: {top_hits}/{n}, bottom ±1: {bot_hits}/{n} (each threshold ≥ 4/5)")
    return {"passed": top_hits >= 4 and bot_hits >= 4, "top": top_hits, "bot": bot_hits, "n": n}


# ---------- Driver ----------


def main() -> int:
    judge = _get_judge()
    if judge is None:
        print("ERROR: no judge configured. Set MEANING_JUDGE_BACKEND=openrouter + OPENROUTER_API_KEY.")
        return 2
    print(f"Judge: {judge.model} @ {judge.endpoint}")

    r1 = check_fidelity_correlation(judge)
    r2 = check_hallucination_ranked_worst(judge)
    r3 = check_rank_matches_subjective(judge)

    print("\n" + "=" * 60)
    print("PRE-FLIGHT SUMMARY")
    print(f"  Check 1 (fidelity ρ ≥ 0.5)         : {'PASS' if r1['passed'] else 'FAIL'} (mean ρ = {r1['mean_rho']:+.3f})")
    print(f"  Check 2 (halluc ranked worst ≥ 2/3): {'PASS' if r2['passed'] else 'FAIL'}")
    print(f"  Check 3 (top/bot match ≥ 4/5 each) : {'PASS' if r3['passed'] else 'FAIL'} (top {r3['top']}/{r3['n']}, bot {r3['bot']}/{r3['n']})")
    all_passed = r1["passed"] and r2["passed"] and r3["passed"]
    print(f"\nALL PASSED: {all_passed}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
