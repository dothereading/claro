"""§6 probe prompts: eyeball a v10 checkpoint on 5 fixed paragraphs.

Generates greedy (temp 0) and temp-1.0 samples from the given adapter for
the three regression cases (Tomorrow Never Dies, Melicope, Ali Darassa) plus
two more from the eval set, scores each with the full v10 reward, and writes
a human-readable markdown report with per-component scores and flagged hard
words inline.

Run every ~25 iterations during a training run:
    OPENROUTER_API_KEY=... uv run python scripts/probe_v10.py \
        --adapter adapters/grpo_v10_full/latest --arm full --iter 25

Without OPENROUTER_API_KEY it still runs, scoring the no-judge reward.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from reward.compose import default_band, default_scorer, reward  # noqa: E402

# 3 regression cases + 2 fixed others from the eval set.
PROBE_TITLES = ["Tomorrow Never Dies", "Melicope", "Ali Darassa", "Posad", "Margot Sponer"]


def _load_sources() -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in (ROOT / "data" / "eval.jsonl").read_text().splitlines() if line.strip()]
    by_title = {r["title"]: r for r in rows}
    out = []
    for needle in PROBE_TITLES:
        hit = next((t for t in by_title if needle.lower() in t.lower()), None)
        if hit:
            out.append((hit, by_title[hit]["complex"]))
    return out


def _fmt_hard(hard_words: list[list[str]]) -> str:
    flat = [w for sent in (hard_words or []) for w in sent]
    return ", ".join(flat) if flat else "(none)"


def _score_block(title: str, source: str, label: str, candidate: str, band, scorer) -> str:
    res = reward(source, candidate, band=band, scorer=scorer, use_fidelity=scorer is not None)
    c = res.components
    fd = res.debug.get("fidelity", {})
    lines = [
        f"**{label}** — total **{res.total:.4f}**  "
        f"(level_band={c['level_band']:.3f} · vocab={c['vocab']:.3f} · "
        f"fidelity={c['fidelity']:.3f} · gates={c['gates']:.0f})",
        "",
        f"> {candidate.strip()}",
        "",
        f"- hard words: {_fmt_hard(res.debug.get('hard_words'))}",
    ]
    if scorer is not None and not res.skipped_judge:
        lines.append(
            f"- fidelity: recall={fd.get('recall')}, "
            f"unsupported={fd.get('n_unsupported')}, failed={fd.get('failed')}"
        )
        claims = (fd.get("judge") or {}).get("unsupported_claims") if fd.get("judge") else None
        if claims:
            lines.append(f"- unsupported claims: {claims}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, help="adapter dir/file, or 'base'")
    ap.add_argument("--arm", default="full", choices=["full", "nojudge"])
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--model", default="mlx-community/gemma-3-1b-it-bf16")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    from langsimp.inference.engine import load_model_with_adapter, make_generate_fn

    use_judge = args.arm == "full" and bool(os.environ.get("OPENROUTER_API_KEY"))
    band = default_band()
    scorer = default_scorer() if use_judge else None

    adapter = None if args.adapter == "base" else args.adapter
    model, tokenizer = load_model_with_adapter(args.model, adapter)
    greedy = make_generate_fn(model, tokenizer, max_tokens=args.max_tokens, temp=0.0)
    sampled = make_generate_fn(model, tokenizer, max_tokens=args.max_tokens, temp=1.0)

    sources = _load_sources()
    parts = [
        f"# v10 probe — arm={args.arm}, iter={args.iter}",
        f"adapter: `{args.adapter}` · judge: {'on' if use_judge else 'off'}",
        "",
    ]
    for title, source in sources:
        print(f"[probe] generating: {title}", flush=True)
        parts.append(f"## {title}\n")
        parts.append(f"*source ({len(source.split())}w):* {source.strip()}\n")
        parts.append(_score_block(title, source, "greedy (temp 0)", greedy(source), band, scorer))
        parts.append("")
        parts.append(_score_block(title, source, "sample (temp 1.0)", sampled(source), band, scorer))
        parts.append("\n---\n")

    out_dir = ROOT / "runs" / args.arm / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"iter_{args.iter}.md"
    out_path.write_text("\n".join(parts) + "\n")
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
