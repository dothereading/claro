"""Generate a BLIND human audit of judge-flagged "unsupported claims".

Both judges flag ~90% of SFT outputs as containing a hallucination — high
enough to suspect the judge counts paraphrase/gloss as invention. This pulls
N flagged claims per model, strips model identity, shuffles them, and writes:

  runs/audit/blind_audit.md   — what you read & fill in (no model labels)
  runs/audit/audit_key.json   — the hidden answer key (don't open until done)

After you fill in verdicts, `score_blind_audit.py` computes the over-flag
rate per model.

Run:  OPENROUTER_API_KEY=... uv run python scripts/make_blind_audit.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from reward.c3_fidelity import build_scorer  # noqa: E402

N_PER_MODEL = 10
SEED = 0
OUT_DIR = ROOT / "runs" / "audit"

SOURCES = {
    "GRPO": ROOT / "eval_results" / "grpo_v10_full_eval50.json",
    "SFT": ROOT / "eval_results" / "sft_n750_1b_eval80.json",
}


def collect_claims(scorer) -> list[dict]:
    grpo = json.loads(SOURCES["GRPO"].read_text())["results"]
    src_by_title = {r["title"]: r["complex"] for r in grpo}
    sft = {r["title"]: r for r in json.loads(SOURCES["SFT"].read_text())["results"]}

    out: list[dict] = []
    for model, recs in (("GRPO", {r["title"]: r for r in grpo}), ("SFT", sft)):
        for title, r in recs.items():
            if title not in src_by_title:
                continue
            src = src_by_title[title]
            res = scorer.score(src, r["output"])
            claims = (res.judge_json or {}).get("unsupported_claims", []) if not res.failed else []
            for claim in claims:
                out.append({"model": model, "title": title, "source": src,
                            "candidate": r["output"], "claim": claim})
    return out


HEADER = """# Blind faithfulness audit — {n} flagged claims

## What this is and why you're doing it

An LLM judge flags "unsupported claims" (hallucinations) in our model's
simplifications — and it flags ~90% of one model's outputs, which is
suspiciously high. We need to know whether those flags are **real inventions**
or whether the judge is **over-flagging** ordinary paraphrase / simplification.
That changes what our headline hallucination numbers mean.

Each item below is one claim the judge marked "unsupported". **You won't be told
which model produced each text** (that's the point — it keeps you unbiased).

## What to do

For each of the {n} items:
1. Read the **SOURCE** and the **SIMPLIFIED TEXT**.
2. Look at the **FLAGGED CLAIM** (a fact the judge says the simplification asserts).
3. Decide whether the **source actually supports that claim**, and write one letter
   in the `verdict:` blank:
   - **U** = genuinely **U**nsupported. The claim states a specific fact
     (name / number / place / date / relationship / explanation) that the source
     does **not** contain, or that **contradicts** the source. A real invention.
   - **S** = actually **S**upported. The claim is just a simpler restatement, a
     gloss of a source term, or a reasonable paraphrase of something in the source.
     (The judge over-flagged.) Omitting detail also counts as S, not U — leaving
     things out isn't inventing.
   - **?** = genuinely can't tell / borderline.

Rule of thumb: **adding** a specific fact the source lacks → U. **Rewording or
simplifying** a fact that's in the source → S.

## How to return it

Fill each `verdict:` blank with U, S, or ?. Then either send this file back, or
just paste me the list like `1:U 2:S 3:? 4:U ...`. I'll score the over-flag rate
per model using the hidden key (`runs/audit/audit_key.json`). **Don't open the key
until you're done.**

---
"""


def main() -> None:
    rng = random.Random(SEED)
    scorer = build_scorer(use_cache=True)
    claims = collect_claims(scorer)

    by_model: dict[str, list[dict]] = {"GRPO": [], "SFT": []}
    for c in claims:
        by_model[c["model"]].append(c)
    picked: list[dict] = []
    for model in ("SFT", "GRPO"):
        pool = by_model[model]
        rng.shuffle(pool)
        picked.extend(pool[:N_PER_MODEL])
    rng.shuffle(picked)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [HEADER.format(n=len(picked))]
    key = {}
    for i, c in enumerate(picked, 1):
        key[str(i)] = {"model": c["model"], "title": c["title"]}
        lines.append(f"## Item {i}\n")
        lines.append(f"**SOURCE:**\n\n{c['source'].strip()}\n")
        lines.append(f"**SIMPLIFIED TEXT:**\n\n{c['candidate'].strip()}\n")
        lines.append(f'**FLAGGED CLAIM:**\n\n> {c["claim"]}\n')
        lines.append("**verdict:** `___`   (U = unsupported/invented · S = supported/paraphrase · ? = unsure)\n")
        lines.append("---\n")

    (OUT_DIR / "blind_audit.md").write_text("\n".join(lines))
    (OUT_DIR / "audit_key.json").write_text(json.dumps(key, indent=1))
    print(f"wrote {OUT_DIR/'blind_audit.md'} ({len(picked)} items: "
          f"{sum(1 for c in picked if c['model']=='SFT')} SFT, "
          f"{sum(1 for c in picked if c['model']=='GRPO')} GRPO)")
    print(f"wrote {OUT_DIR/'audit_key.json'} (hidden answer key)")
    print(f"\ntotal flagged claims available: SFT={len(by_model['SFT'])} GRPO={len(by_model['GRPO'])}")
    print(f"judge calls={scorer.calls} cache_hits={scorer.cache_hits}")


if __name__ == "__main__":
    main()
