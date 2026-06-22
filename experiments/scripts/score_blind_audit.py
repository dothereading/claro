"""Score the filled-in blind audit against the hidden key.

Reads your verdicts from runs/audit/blind_audit.md (the letter you put in each
`**verdict:** ` blank) and runs/audit/audit_key.json, then reports the
judge OVER-FLAG rate per model = fraction of flagged claims you marked S
(actually supported). High over-flag = the judge's hallucination rate is
inflated.

Run:  uv run python scripts/score_blind_audit.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "runs" / "audit"

_ITEM_RE = re.compile(r"## Item (\d+)\b")
_VERDICT_RE = re.compile(r"\*\*verdict:\*\*\s*`?\s*([USsu?])\s*`?", re.IGNORECASE)


def parse_verdicts(md: str) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    chunks = re.split(r"(?=## Item \d+\b)", md)
    for ch in chunks:
        m = _ITEM_RE.search(ch)
        if not m:
            continue
        v = _VERDICT_RE.search(ch)
        if not v:
            continue
        letter = v.group(1).upper()
        if letter == "_":  # still blank
            continue
        verdicts[m.group(1)] = letter
    return verdicts


def main() -> None:
    md = (AUDIT / "blind_audit.md").read_text()
    key = json.loads((AUDIT / "audit_key.json").read_text())
    verdicts = parse_verdicts(md)

    filled = {k: v for k, v in verdicts.items() if v in {"U", "S", "?"}}
    missing = [k for k in key if k not in filled]
    if missing:
        print(f"[warn] {len(missing)} items still blank: {', '.join(sorted(missing, key=int))}\n")

    per_model: dict[str, list[str]] = defaultdict(list)
    for item_id, info in key.items():
        if item_id in filled:
            per_model[info["model"]].append(filled[item_id])

    print(f"=== BLIND AUDIT RESULT ({len(filled)}/{len(key)} judged) ===")
    print(f"{'model':6s} {'n':>3s} {'U(real)':>8s} {'S(overflag)':>12s} {'?':>3s} {'over-flag rate':>15s}")
    for model in ("SFT", "GRPO"):
        v = per_model.get(model, [])
        if not v:
            continue
        nU, nS, nQ = v.count("U"), v.count("S"), v.count("?")
        rate = nS / len(v) if v else 0.0
        print(f"{model:6s} {len(v):>3d} {nU:>8d} {nS:>12d} {nQ:>3d} {rate:>14.0%}")
    print("\nInterpretation: a high S rate means the judge over-flags (counts paraphrase/")
    print("gloss as invention), so the absolute hallucination numbers are inflated. Compare")
    print("the two models' over-flag rates — if similar, the relative SFT→GRPO gap is clean.")


if __name__ == "__main__":
    main()
