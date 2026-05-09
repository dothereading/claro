"""Convert generated JSONL data into the on-disk format mlx-lm expects.

  * `python mlx_data.py sft` reads data/sft.jsonl (records with `complex` and
    `simple` fields) and writes data/mlx/{train,valid}.jsonl as chat records.

  * `python mlx_data.py dpo` reads data/dpo.jsonl (records with `prompt`,
    `chosen`, `rejected`) and writes data/dpo_mlx/{train,valid}.jsonl in the
    DPO format mlx_lm_lora expects.

The shared helpers (`to_mlx_sft_record`, `to_mlx_dpo_record`,
`split_train_valid`) are exported so tests can exercise them directly.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from prompts import SFT_SYSTEM_PROMPT

REPO_ROOT = Path(__file__).resolve().parent


def to_mlx_sft_record(complex_text: str, simple_text: str) -> dict:
    """Format one (complex, simple) pair as an mlx-lm chat record."""
    return {
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": complex_text.strip()},
            {"role": "assistant", "content": simple_text.strip()},
        ]
    }


def to_mlx_dpo_record(prompt: str, chosen: str, rejected: str) -> dict:
    """Format one preference triple as an mlx_lm_lora DPO record."""
    return {
        "system": SFT_SYSTEM_PROMPT,
        "prompt": prompt.strip(),
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
    }


def split_train_valid(
    rows: list[dict], valid_frac: float, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    """Shuffle `rows` and split into (train, valid). At least one valid row."""
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_valid = max(1, int(len(shuffled) * valid_frac))
    return shuffled[n_valid:], shuffled[:n_valid]


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_split(out_dir: Path, split_name: str, records: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split_name}.jsonl"
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} → {path}")


def _sft_main(args: argparse.Namespace) -> None:
    rows = _read_jsonl(Path(args.input))
    if args.n and len(rows) < args.n:
        raise SystemExit(f"requested --n={args.n} but only {len(rows)} rows in {args.input}")
    if args.n:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.n]

    train, valid = split_train_valid(rows, args.valid_frac, args.seed)
    out_dir = Path(args.output_dir)
    _write_split(out_dir, "train", [to_mlx_sft_record(r["complex"], r["simple"]) for r in train])
    _write_split(out_dir, "valid", [to_mlx_sft_record(r["complex"], r["simple"]) for r in valid])


def _dpo_main(args: argparse.Namespace) -> None:
    rows = _read_jsonl(Path(args.input))
    train, valid = split_train_valid(rows, args.valid_frac, args.seed)
    out_dir = Path(args.output_dir)
    _write_split(
        out_dir, "train",
        [to_mlx_dpo_record(r["prompt"], r["chosen"], r["rejected"]) for r in train],
    )
    _write_split(
        out_dir, "valid",
        [to_mlx_dpo_record(r["prompt"], r["chosen"], r["rejected"]) for r in valid],
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sft = sub.add_parser("sft", help="convert data/sft.jsonl → data/mlx/{train,valid}.jsonl")
    sft.add_argument("--input", default=str(REPO_ROOT / "data" / "sft.jsonl"))
    sft.add_argument("--output-dir", default=str(REPO_ROOT / "data" / "mlx"))
    sft.add_argument("--n", type=int, default=0, help="0 = use all rows")
    sft.add_argument("--valid-frac", type=float, default=0.1)
    sft.add_argument("--seed", type=int, default=0)

    dpo = sub.add_parser("dpo", help="convert data/dpo.jsonl → data/dpo_mlx/{train,valid}.jsonl")
    dpo.add_argument("--input", default=str(REPO_ROOT / "data" / "dpo.jsonl"))
    dpo.add_argument("--output-dir", default=str(REPO_ROOT / "data" / "dpo_mlx"))
    dpo.add_argument("--valid-frac", type=float, default=0.1)
    dpo.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.cmd == "sft":
        _sft_main(args)
    elif args.cmd == "dpo":
        _dpo_main(args)


if __name__ == "__main__":
    main()
