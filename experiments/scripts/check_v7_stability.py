"""v7 judge-stability harness.

For each of 3 source paragraphs, gather 8 candidates (base, n=500, n=750,
n=1000, n=1500 4B SFT outputs + a fresh Opus simplification + a deliberate
hallucination + n=0 base), send them through the live ranking judge
TWICE per source per prompt variant, and tabulate where each named
candidate landed.

The point is to measure JUDGE STABILITY, not absolute quality. The Opus
output is our "gold" anchor — it should land top-1 consistently. The
HALLUC candidate is our "floor" anchor — it should land last consistently.
Variance between Opus and HALLUC is signal; everything else is noise we
can't directly control.

Run:
    MEANING_JUDGE_BACKEND=openrouter \\
    MEANING_JUDGE_MODEL=deepseek/deepseek-v4-pro:gmicloud/fp8 \\
    uv run python scripts/check_v7_stability.py --round 1

Round 2 uses an alternate prompt baked in below.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from langsimp.data.distill import Teacher  # noqa: E402
from langsimp.prompts import DISTILL_SYSTEM_PROMPT  # noqa: E402
from experiments.rewards.rewards_legacy import _get_judge  # noqa: E402
from experiments.rewards.rewards_v7 import _parse_rank_list  # noqa: E402

# ---------- Three source paragraphs ----------

EXAMPLES = ["Margot Sponer", "Cayan Tower", "ʼPhags-pa script"]

EVAL_FILES = {
    "base": "base_4b_eval30.json",
    "n=500": "sft_n500_4b_eval30.json",
    "n=750": "sft_n750_4b_eval30.json",
    "n=1000": "sft_n1000_4b_eval30.json",
    "n=1500": "sft_n1500_4b_eval30.json",
}

OPUS_CACHE = REPO_ROOT / "scripts" / ".opus_cache.json"

# ---------- Two prompt variants (round 1 = current v7; round 2 = revised) ----------

PROMPT_ROUND_1 = """Rank these CEFR A2 simplifications of the source paragraph from best to worst.

A good A2 simplification:
- Uses simple, A2-level English (short sentences, common ~1500 words, no idioms)
- Faithfully preserves the source's meaning and key facts (no invented or distorted claims)

SOURCE PARAGRAPH:
{source}

CANDIDATES:
{candidates}

Respond with ONLY a JSON array of candidate IDs ordered best-first. Example for 8 candidates: [3, 0, 5, 1, 7, 2, 4, 6]"""

# Round 2 prompt is the validated canonical prompt (synced with
# langsimp/training/rewards_v7.py:_RANK_PROMPT_TEMPLATE).
PROMPT_ROUND_2 = """You are ranking CEFR A2 simplifications of an English source paragraph.

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


# ---------- Helpers ----------


def load_candidate(eval_name: str, title: str) -> tuple[str, str]:
    d = json.load(open(REPO_ROOT / "eval_results" / eval_name))
    for r in d["results"]:
        if (r.get("title") or "") == title:
            return r["complex"], r["output"]
    raise KeyError(f"{title} not found in {eval_name}")


async def get_opus_outputs(sources: dict[str, str]) -> dict[str, str]:
    """Generate (or read from cache) Opus simplifications for the 3 sources."""
    cache: dict[str, str] = {}
    if OPUS_CACHE.exists():
        cache = json.loads(OPUS_CACHE.read_text())
    todo = {title: src for title, src in sources.items() if title not in cache}
    if not todo:
        return cache

    teacher = Teacher.from_env(model="anthropic/claude-opus-4-5", temperature=0.2)
    print(f"[opus] generating {len(todo)} simplifications...", flush=True)
    for title, src in todo.items():
        out = await teacher.simplify(DISTILL_SYSTEM_PROMPT, src)
        if out:
            cache[title] = out
            print(f"  ✓ {title} ({len(out.split())}w)", flush=True)
    OPUS_CACHE.write_text(json.dumps(cache, indent=2))
    return cache


def make_hallucination(opus_text: str) -> str:
    return (
        opus_text
        + " Scholars have noted that this was originally proposed by Christina Pagel in 1750."
    )


def build_candidates(title: str, opus_text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (source, [(label, text), ...]) — 8 candidates per example."""
    source, _ = load_candidate(EVAL_FILES["base"], title)
    cands: list[tuple[str, str]] = [("base", load_candidate(EVAL_FILES["base"], title)[1])]
    for label in ("n=500", "n=750", "n=1000", "n=1500"):
        cands.append((label, load_candidate(EVAL_FILES[label], title)[1]))
    cands.append(("OPUS", opus_text))
    cands.append(("HALLUC", make_hallucination(opus_text)))
    # 7 candidates so far. Add one more to reach G=8 — duplicate base with a
    # markdown header so we can also confirm the judge penalizes markdown.
    cands.append(("base+md", "## Summary\n\n" + cands[0][1]))
    assert len(cands) == 8, len(cands)
    return source, cands


def run_judge_once(judge, prompt_template: str, source: str, candidates: list[tuple[str, str]]):
    candidate_block = "\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(candidates))
    prompt = prompt_template.format(source=source, candidates=candidate_block)
    raw = judge.evaluate(prompt)
    order = _parse_rank_list(raw, len(candidates))
    return raw, order


# ---------- Summary table ----------


def summarize(title: str, candidates: list[tuple[str, str]], rounds: list[list[int]]):
    """rounds: list of permutations (best→worst) from N repeat judge calls."""
    print(f"\n--- {title} ---")
    print("calls:")
    for k, order in enumerate(rounds):
        ranking = " > ".join(candidates[i][0] for i in order)
        print(f"  call {k + 1}: {ranking}")

    # Rank of each label in each call (lower = better; rank 0 = top)
    labels = [c[0] for c in candidates]
    ranks: dict[str, list[int]] = {label: [] for label in labels}
    for order in rounds:
        for rank, cid in enumerate(order):
            ranks[candidates[cid][0]].append(rank)

    print("\nrank per call (lower = better):")
    print(f"  {'label':<10}  " + "  ".join(f"c{i + 1}" for i in range(len(rounds))) + "   avg")
    for label in labels:
        rs = ranks[label]
        avg = sum(rs) / len(rs)
        print(f"  {label:<10}  " + "  ".join(f"{r:>2}" for r in rs) + f"   {avg:.1f}")

    # Top-1 agreement and key anchor checks
    top1 = Counter(candidates[order[0]][0] for order in rounds)
    print(f"\ntop-1 distribution: {dict(top1)}")
    halluc_ranks = ranks.get("HALLUC", [])
    opus_ranks = ranks.get("OPUS", [])
    return {
        "title": title,
        "opus_ranks": opus_ranks,
        "halluc_ranks": halluc_ranks,
        "base_ranks": ranks.get("base", []),
        "base_md_ranks": ranks.get("base+md", []),
        "top1_counter": dict(top1),
    }


# ---------- Driver ----------


async def amain(round_num: int, n_calls: int = 2):
    judge = _get_judge()
    if judge is None:
        print("ERROR: no judge configured.")
        return 2
    print(f"Judge: {judge.model}")
    print(f"Round: {round_num}, calls per example: {n_calls}")

    sources = {title: load_candidate(EVAL_FILES["base"], title)[0] for title in EXAMPLES}
    opus = await get_opus_outputs(sources)

    prompt = PROMPT_ROUND_1 if round_num == 1 else PROMPT_ROUND_2
    print(f"\nPrompt preamble: {prompt[:200].splitlines()[0]}...")

    all_summaries = []
    for title in EXAMPLES:
        source, candidates = build_candidates(title, opus[title])
        rounds: list[list[int]] = []
        for k in range(n_calls):
            raw, order = run_judge_once(judge, prompt, source, candidates)
            if order is None:
                print(f"  call {k + 1} FAILED to parse: {raw!r}")
                continue
            rounds.append(order)
        s = summarize(title, candidates, rounds)
        all_summaries.append(s)

    # Aggregate across the 3 examples
    print("\n" + "=" * 60)
    print(f"ROUND {round_num} AGGREGATE")
    print("=" * 60)
    all_opus = [r for s in all_summaries for r in s["opus_ranks"]]
    all_halluc = [r for s in all_summaries for r in s["halluc_ranks"]]
    all_base = [r for s in all_summaries for r in s["base_ranks"]]
    all_base_md = [r for s in all_summaries for r in s["base_md_ranks"]]
    n = len(all_opus)
    print(f"OPUS    mean rank: {sum(all_opus)/n:.2f}  (target: ~0, best)")
    print(f"base    mean rank: {sum(all_base)/n:.2f}  (target: ~5-6, bottom half)")
    print(f"base+md mean rank: {sum(all_base_md)/n:.2f}  (target: ~6-7, near bottom)")
    print(f"HALLUC  mean rank: {sum(all_halluc)/n:.2f}  (target: ~7, worst)")
    print(f"OPUS top-1 rate: {sum(1 for r in all_opus if r == 0)}/{n}")
    print(f"HALLUC bottom-1 rate: {sum(1 for r in all_halluc if r == 7)}/{n}")
    print(f"HALLUC bottom-half rate: {sum(1 for r in all_halluc if r >= 4)}/{n}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, choices=[1, 2], default=1)
    p.add_argument("--n-calls", type=int, default=2)
    args = p.parse_args()
    sys.exit(asyncio.run(amain(args.round, args.n_calls)))


if __name__ == "__main__":
    main()
