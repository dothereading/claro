"""Training entrypoint with W&B observability.

Wraps the underlying mlx-lm / mlx-lm-lora trainers in a subprocess, tees
their stdout to the terminal *and* through a regex parser that forwards
metrics to Weights & Biases in real time.

Two subcommands mirroring the previous shell scripts:

    uv run python train.py sft  --model ... --data data/mlx     --iters 300 ...
    uv run python train.py dpo  --model ... --data data/dpo_mlx --iters 300 --beta 0.1 ...

Offline-safe: if WANDB_API_KEY is missing or WANDB_MODE=disabled, training
runs without W&B and prints a one-line note. Training never breaks because
of an observability problem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")


# ---------- adapter versioning helpers ----------

def git_short_sha() -> Optional[str]:
    """Short git SHA for the current HEAD, or None if not in a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def dataset_hash(data_dir: Path) -> str:
    """SHA-256 over train.jsonl + valid.jsonl content. 'missing' if dir absent."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return "missing"
    h = hashlib.sha256()
    for name in ("train.jsonl", "valid.jsonl"):
        p = data_dir / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def make_adapter_dir(
    base_dir: Path,
    stage: str,
    timestamp: str,
    git_sha: Optional[str],
) -> Path:
    """Create and return adapters/<stage>/<timestamp>-<sha>/."""
    sha_part = git_sha or "nosha"
    out = Path(base_dir) / stage / f"{timestamp}-{sha_part}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def update_latest_symlink(link_path: Path, target: Path) -> None:
    """Atomically point `link_path` at `target` (relative path for portability).

    Replaces an existing symlink or file. No-op if the link already points at
    the target.
    """
    link_path = Path(link_path)
    target = Path(target)
    rel = os.path.relpath(target, link_path.parent)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(rel)


def write_meta(adapter_dir: Path, meta: dict) -> None:
    """Persist meta.json next to the adapter weights."""
    (Path(adapter_dir) / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n"
    )

# ---------- log parsers ----------
#
# Lines we care about, sampled from logs/sft_train.log and logs/dpo_train.log:
#
#   Iter 10: Train loss 2.249, Learning Rate 1.000e-04, It/sec 1.380,
#       Tokens/sec 404.447, Trained Tokens 2931, Peak mem 4.005 GB
#   Iter 50: Val loss 1.728, Val took 0.282s
#
#   Iter 10: loss 0.011, chosen_r 83.361, rejected_r 65.698, acc 1.000,
#       margin 17.663, lr 5.000e-06, it/s 2.012, tok/s 1172.116, peak_mem 8.719GB
#   Iter 50: Val loss 0.000, Val chosen reward 0.143, Val rejected reward 0.122,
#       Val accuracy 1.000, Val margin 10.016, Val took 0.765s

_NUM = r"-?[\d.]+(?:e[+-]?\d+)?"

SFT_TRAIN_RE = re.compile(
    rf"Iter (\d+): Train loss ({_NUM}), Learning Rate ({_NUM}), "
    rf"It/sec ({_NUM}), Tokens/sec ({_NUM}), Trained Tokens (\d+), Peak mem ({_NUM}) GB"
)
SFT_VAL_RE = re.compile(rf"Iter (\d+): Val loss ({_NUM}), Val took")

DPO_TRAIN_RE = re.compile(
    rf"Iter (\d+): loss ({_NUM}), chosen_r ({_NUM}), rejected_r ({_NUM}), "
    rf"acc ({_NUM}), margin ({_NUM}), lr ({_NUM}), it/s ({_NUM}), tok/s ({_NUM}), peak_mem ({_NUM})GB"
)
DPO_VAL_RE = re.compile(
    rf"Iter (\d+): Val loss ({_NUM}), Val chosen reward ({_NUM}), "
    rf"Val rejected reward ({_NUM}), Val accuracy ({_NUM}), Val margin ({_NUM})"
)


def parse_sft_line(line: str) -> Optional[dict]:
    m = SFT_TRAIN_RE.search(line)
    if m:
        return {
            "iter": int(m.group(1)),
            "train/loss": float(m.group(2)),
            "train/lr": float(m.group(3)),
            "train/it_per_sec": float(m.group(4)),
            "train/tok_per_sec": float(m.group(5)),
            "train/trained_tokens": int(m.group(6)),
            "train/peak_mem_gb": float(m.group(7)),
        }
    m = SFT_VAL_RE.search(line)
    if m:
        return {"iter": int(m.group(1)), "valid/loss": float(m.group(2))}
    return None


def parse_dpo_line(line: str) -> Optional[dict]:
    m = DPO_TRAIN_RE.search(line)
    if m:
        return {
            "iter": int(m.group(1)),
            "train/loss": float(m.group(2)),
            "train/chosen_reward": float(m.group(3)),
            "train/rejected_reward": float(m.group(4)),
            "train/accuracy": float(m.group(5)),
            "train/margin": float(m.group(6)),
            "train/lr": float(m.group(7)),
            "train/it_per_sec": float(m.group(8)),
            "train/tok_per_sec": float(m.group(9)),
            "train/peak_mem_gb": float(m.group(10)),
        }
    m = DPO_VAL_RE.search(line)
    if m:
        return {
            "iter": int(m.group(1)),
            "valid/loss": float(m.group(2)),
            "valid/chosen_reward": float(m.group(3)),
            "valid/rejected_reward": float(m.group(4)),
            "valid/accuracy": float(m.group(5)),
            "valid/margin": float(m.group(6)),
        }
    return None


# ---------- run name ----------

def _short_model(model: str) -> str:
    """`mlx-community/gemma-3-1b-it-bf16` → `gemma-3-1b`."""
    leaf = model.rsplit("/", 1)[-1]
    # keep up to the parameter-count token (e.g. "gemma-3-1b")
    parts = leaf.split("-")
    keep: list[str] = []
    for p in parts:
        keep.append(p)
        if re.match(r"^\d+[bm]$", p):  # "1b", "4b", "7b", "70b", "350m"
            break
    return "-".join(keep)


def build_run_name(stage: str, model: str, config: dict) -> str:
    ts = time.strftime("%Y%m%dT%H%M%S")
    parts = [ts, stage, _short_model(model)]
    if "iters" in config:
        parts.append(f"iters{config['iters']}")
    if "lr" in config:
        parts.append(f"lr{config['lr']:g}")
    if stage == "dpo" and "beta" in config:
        parts.append(f"beta{config['beta']:g}")
    return "-".join(parts)


# ---------- W&B integration ----------

def _wandb_or_none(project: str, run_name: str, config: dict, tags: list[str]):
    """Return an initialized wandb run, or None if W&B is unavailable.

    Failure modes that fall through to None (with a printed note):
      * WANDB_MODE=disabled
      * WANDB_API_KEY missing AND no cached login
      * wandb.init() raises (network down, etc.)
    """
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        print("[train] WANDB_MODE=disabled — running without W&B", flush=True)
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("[train] WANDB_API_KEY not set — running without W&B", flush=True)
        return None
    try:
        import wandb
        return wandb.init(project=project, name=run_name, config=config, tags=tags)
    except Exception as e:
        print(f"[train] wandb.init failed ({e}) — running without W&B", flush=True)
        return None


def run_with_logging(
    cmd: list[str],
    parser: Callable[[str], Optional[dict]],
    project: str,
    run_name: str,
    config: dict,
    tags: list[str],
) -> dict:
    """Launch `cmd` as a subprocess, parse stdout, forward metrics to W&B.

    Returns {exit_code, last_train_metrics, last_valid_metrics, wandb_run_id}.
    Always tees output to stdout so the user sees training progress live,
    with or without W&B.
    """
    run = _wandb_or_none(project, run_name, config, tags)
    log_metric = run.log if run is not None else (lambda *a, **kw: None)
    last_train: dict = {}
    last_valid: dict = {}

    print(f"[train] launching: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            metrics = parser(line)
            if metrics:
                step = metrics.pop("iter", None)
                log_metric(metrics, step=step)
                # Cache the latest train/* and valid/* metrics for meta.json.
                if any(k.startswith("train/") for k in metrics):
                    last_train = {**metrics, "iter": step}
                elif any(k.startswith("valid/") for k in metrics):
                    last_valid = {**metrics, "iter": step}
        proc.wait()
    finally:
        if run is not None:
            run.finish()
    return {
        "exit_code": proc.returncode,
        "last_train_metrics": last_train,
        "last_valid_metrics": last_valid,
        "wandb_run_id": run.id if run is not None else None,
        "wandb_url": run.url if run is not None else None,
    }


# ---------- subcommands ----------

def _resolve_adapter_dir(args: argparse.Namespace, stage: str) -> tuple[Path, str, Optional[str]]:
    """Compute the versioned adapter dir, plus the timestamp/sha used.

    If --adapter-path is set explicitly, honor it (no versioning) — useful
    for ad-hoc runs that want to overwrite a known location.
    """
    if args.adapter_path:
        adapter_dir = Path(args.adapter_path)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        return adapter_dir, "explicit", None
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    sha = git_short_sha()
    adapter_dir = make_adapter_dir(REPO_ROOT / "adapters", stage, timestamp, sha)
    return adapter_dir, timestamp, sha


def _finalize(stage: str, adapter_dir: Path, run_result: dict, config: dict, cmd: list[str], timestamp: str) -> None:
    """Write meta.json and update the `latest` symlink. No-op on bad runs."""
    if run_result["exit_code"] != 0:
        print(f"[train] non-zero exit ({run_result['exit_code']}); skipping meta + symlink update", flush=True)
        return
    meta = {
        "timestamp": timestamp,
        "stage": stage,
        "git_sha": git_short_sha(),
        "config": config,
        "training_command": cmd,
        "wandb_run_id": run_result.get("wandb_run_id"),
        "wandb_url": run_result.get("wandb_url"),
        "final_train_metrics": run_result.get("last_train_metrics", {}),
        "final_valid_metrics": run_result.get("last_valid_metrics", {}),
    }
    write_meta(adapter_dir, meta)
    print(f"[train] wrote {adapter_dir}/meta.json", flush=True)

    # Update adapters/<stage>/latest unless caller pinned an explicit dir.
    if timestamp != "explicit":
        link = REPO_ROOT / "adapters" / stage / "latest"
        update_latest_symlink(link, adapter_dir)
        print(f"[train] updated {link} → {adapter_dir.name}", flush=True)


def _sft(args: argparse.Namespace) -> int:
    adapter_dir, timestamp, sha = _resolve_adapter_dir(args, "sft")
    cmd = [
        "uv", "run", "python", "-m", "mlx_lm", "lora",
        "--model", args.model,
        "--train",
        "--data", args.data,
        "--adapter-path", str(adapter_dir),
        "--batch-size", str(args.batch_size),
        "--num-layers", str(args.lora_layers),
        "--iters", str(args.iters),
        "--learning-rate", str(args.lr),
        "--val-batches", str(args.val_batches),
        "--steps-per-eval", str(args.steps_per_eval),
        "--steps-per-report", str(args.steps_per_report),
        "--grad-checkpoint",
    ]
    config = {
        "stage": "sft", "model": args.model, "data": args.data,
        "iters": args.iters, "lr": args.lr,
        "batch_size": args.batch_size, "lora_layers": args.lora_layers,
        "dataset_hash": dataset_hash(Path(args.data)),
        "adapter_dir": str(adapter_dir),
    }
    name = build_run_name("sft", args.model, config)
    result = run_with_logging(
        cmd, parse_sft_line, project=args.project, run_name=name,
        config=config, tags=["sft", "mlx-lm"],
    )
    _finalize("sft", adapter_dir, result, config, cmd, timestamp)
    return result["exit_code"]


def _dpo(args: argparse.Namespace) -> int:
    adapter_dir, timestamp, sha = _resolve_adapter_dir(args, "dpo")
    cmd = [
        "uv", "run", "python", "-m", "mlx_lm_lora.train",
        "--model", args.model,
        "--train",
        "--train-mode", "dpo",
        "--data", args.data,
        "--adapter-path", str(adapter_dir),
        "--batch-size", str(args.batch_size),
        "--num-layers", str(args.lora_layers),
        "--iters", str(args.iters),
        "--learning-rate", str(args.lr),
        "--beta", str(args.beta),
        "--dpo-cpo-loss-type", "sigmoid",
        "--val-batches", str(args.val_batches),
        "--steps-per-eval", str(args.steps_per_eval),
        "--steps-per-report", str(args.steps_per_report),
        "--grad-checkpoint",
    ]
    if args.resume_adapter and Path(args.resume_adapter).exists():
        cmd.extend(["--resume-adapter-file", args.resume_adapter])
        print(f"[train] resuming from {args.resume_adapter}", flush=True)
    config = {
        "stage": "dpo", "model": args.model, "data": args.data,
        "iters": args.iters, "lr": args.lr, "beta": args.beta,
        "batch_size": args.batch_size, "lora_layers": args.lora_layers,
        "dataset_hash": dataset_hash(Path(args.data)),
        "resume_from": args.resume_adapter,
        "adapter_dir": str(adapter_dir),
    }
    name = build_run_name("dpo", args.model, config)
    result = run_with_logging(
        cmd, parse_dpo_line, project=args.project, run_name=name,
        config=config, tags=["dpo", "mlx-lm-lora"],
    )
    _finalize("dpo", adapter_dir, result, config, cmd, timestamp)
    return result["exit_code"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sft = sub.add_parser("sft", help="LoRA SFT via mlx-lm with W&B logging")
    sft.add_argument("--model", default="mlx-community/gemma-3-1b-it-bf16")
    sft.add_argument("--data", default="data/mlx")
    sft.add_argument("--adapter-path", default=None,
                     help="explicit adapter dir; default = adapters/sft/<timestamp>-<sha>/ + latest symlink")
    sft.add_argument("--iters", type=int, default=300)
    sft.add_argument("--batch-size", type=int, default=1)
    sft.add_argument("--lr", type=float, default=1e-4)
    sft.add_argument("--lora-layers", type=int, default=16)
    sft.add_argument("--val-batches", type=int, default=5)
    sft.add_argument("--steps-per-eval", type=int, default=50)
    sft.add_argument("--steps-per-report", type=int, default=10)
    sft.add_argument("--project", default="lang-simp-sft")

    dpo = sub.add_parser("dpo", help="LoRA DPO via mlx-lm-lora with W&B logging")
    dpo.add_argument("--model", default="mlx-community/gemma-3-1b-it-bf16")
    dpo.add_argument("--data", default="data/dpo_mlx")
    dpo.add_argument("--adapter-path", default=None,
                     help="explicit adapter dir; default = adapters/dpo/<timestamp>-<sha>/ + latest symlink")
    dpo.add_argument("--resume-adapter", default="adapters/sft/latest/adapters.safetensors",
                     help="resume from this adapter file (silently ignored if missing)")
    dpo.add_argument("--iters", type=int, default=300)
    dpo.add_argument("--batch-size", type=int, default=1)
    dpo.add_argument("--lr", type=float, default=5e-6)
    dpo.add_argument("--beta", type=float, default=0.1)
    dpo.add_argument("--lora-layers", type=int, default=16)
    dpo.add_argument("--val-batches", type=int, default=5)
    dpo.add_argument("--steps-per-eval", type=int, default=50)
    dpo.add_argument("--steps-per-report", type=int, default=10)
    dpo.add_argument("--project", default="lang-simp-dpo")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.cmd == "sft":
        return _sft(args)
    if args.cmd == "dpo":
        return _dpo(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
