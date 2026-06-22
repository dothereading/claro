"""PLAN_4B Phase 3 generation: greedy outputs from each GRPO checkpoint.

mlx-lm's `load(model, adapter_path=...)` wants a *directory* (adapter_config.json
+ adapters.safetensors), but a checkpointed run leaves per-iter files
(`0000050_adapters.safetensors`, ...) in one dir — and that dir gets fused to a
full model at the end, which would double-apply if loaded as a base. So for each
checkpoint we stage a temp adapter dir (the run's adapter_config.json + the
checkpoint file renamed to adapters.safetensors) on top of the BASE model, then
generate greedy over the held-out items.

It only generates + persists `{title, complex, output}` JSON per checkpoint;
scoring is the separate concern of `eval_composite.py`, run over the outputs:

    OPENROUTER_API_KEY=... uv run python scripts/select_checkpoint.py \
        --run-dir adapters/grpo_v11_4b_n1500 \
        --items eval_results/grpo_v11_4b_eval.json \
        --out-prefix eval_results/ckpt_n1500
    uv run python scripts/eval_composite.py eval_results/ckpt_n1500_*.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from claro.inference.engine import load_model_with_adapter, make_generate_fn  # noqa: E402


def _checkpoints(run_dir: Path, iters: list[int] | None) -> list[tuple[int, Path]]:
    found = {}
    for p in run_dir.glob("*_adapters.safetensors"):
        m = re.match(r"0*(\d+)_adapters\.safetensors", p.name)
        if m:
            found[int(m.group(1))] = p
    if iters:
        return [(i, found[i]) for i in iters if i in found]
    return sorted(found.items())


def _items(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows = data["results"] if isinstance(data, dict) and "results" in data else data
    return [{"title": r["title"], "complex": r["complex"]} for r in rows]


def generate_for_checkpoint(model_id: str, config_path: Path, ckpt: Path,
                            items: list[dict], max_tokens: int) -> list[dict]:
    """Stage a temp adapter dir for one checkpoint, load base+adapter, gen greedy."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        shutil.copy(config_path, d / "adapter_config.json")
        shutil.copy(ckpt, d / "adapters.safetensors")
        model, tok = load_model_with_adapter(model_id, str(d))
        gen = make_generate_fn(model, tok, max_tokens=max_tokens, temp=0.0)
        out = []
        for i, it in enumerate(items):
            out.append({"title": it["title"], "complex": it["complex"],
                        "output": gen(it["complex"])})
            print(f"    [{i + 1}/{len(items)}] {it['title'][:40]}", flush=True)
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="GRPO adapter dir with checkpoints")
    ap.add_argument("--items", required=True, help="eval JSON providing title+complex sources")
    ap.add_argument("--model", default="mlx-community/gemma-3-4b-it-bf16")
    ap.add_argument("--iters", type=int, nargs="*", default=None,
                    help="checkpoint iters to eval (default: all found)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="first N items (smoke)")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    config_path = run_dir / "adapter_config.json"
    items = _items(ROOT / args.items if not Path(args.items).is_absolute() else Path(args.items))
    if args.limit:
        items = items[: args.limit]
    ckpts = _checkpoints(run_dir, args.iters)
    if not ckpts:
        raise SystemExit(f"no checkpoints found in {run_dir}")
    print(f"[select] {len(ckpts)} checkpoints x {len(items)} items, model={args.model}")

    for it, ckpt in ckpts:
        print(f"[select] checkpoint iter {it}: {ckpt.name}", flush=True)
        results = generate_for_checkpoint(args.model, config_path, ckpt, items, args.max_tokens)
        out_path = Path(f"{args.out_prefix}_{it:04d}.json")
        out_path.write_text(json.dumps({"meta": {"run_dir": str(run_dir), "iter": it,
                                                  "model": args.model},
                                        "results": results}, indent=1))
        print(f"[select] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
