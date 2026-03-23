#!/usr/bin/env python3
"""CVPR-style ablation study runner.

Ablation matrix:
  full       : self + relation(spatio-temporal graph)
  no_gat     : relation branch uses MLP instead of spatio-temporal GAT
  self_only  : pure appearance baseline (no relation)

Supports single-GPU and multi-GPU (torchrun) launch:
  # single GPU
  conda run -n svip python src/run_ablation.py --batch_size 16 --num_epochs 15

  # multi GPU
  CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n svip --no-capture-output \
    python src/run_ablation.py --nproc_per_node 4 --batch_size 32 --num_epochs 15
"""

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ABLATION_CONFIGS = {
    "full": {},
    "self_only": {"--relation_enabled": "0"},
    "no_gat": {"--no_gat": None},
    "no_temporal": {"--no_temporal_edges": None},
    "no_edge_feat": {"--no_edge_features": None},
    "no_geom": {"--no_geom": None},
}

ABLATION_ORDER = ["full", "self_only", "no_gat", "no_temporal", "no_edge_feat", "no_geom"]


def parse_args():
    parser = argparse.ArgumentParser(description="CVPR Ablation Study Runner")
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", "-e", type=int, default=15)
    parser.add_argument("--learning_rate", "-l", type=float, default=5e-5)
    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--num_workers", "-w", type=int, default=8)
    parser.add_argument("--roi_chunk", type=int, default=512)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of GPUs for DDP")
    parser.add_argument("--experiments", type=str, default=None, help="Comma-separated subset")
    parser.add_argument("--output_base", type=str, default="outputs")
    return parser.parse_args()


_next_port = 29500


def build_command(args, experiment_name: str, output_dir: str) -> list[str]:
    global _next_port

    train_args = [
        "src/train.py",
        "--batch_size", str(args.batch_size),
        "--accumulation_steps", str(args.accumulation_steps),
        "--num_epochs", str(args.num_epochs),
        "--learning_rate", str(args.learning_rate),
        "--data_ratio", str(args.data_ratio),
        "--num_workers", str(args.num_workers),
        "--roi_chunk", str(args.roi_chunk),
        "--early_stop", str(args.early_stop),
        "--seed", str(args.seed),
        "--output_dir", output_dir,
    ]

    for flag, value in ABLATION_CONFIGS[experiment_name].items():
        train_args.append(flag)
        if value is not None:
            train_args.append(str(value))

    if args.nproc_per_node > 1:
        port = _next_port
        _next_port += 1
        torchrun = shutil.which("torchrun")
        if torchrun is None:
            cmd = [sys.executable, "-m", "torch.distributed.run",
                   "--nproc_per_node", str(args.nproc_per_node),
                   "--master_port", str(port)] + train_args
        else:
            cmd = [torchrun,
                   "--nproc_per_node", str(args.nproc_per_node),
                   "--master_port", str(port)] + train_args
    else:
        cmd = [sys.executable] + train_args

    return cmd


def extract_best_metrics(run_dir: Path) -> dict:
    csv_path = run_dir / "records" / "losses.csv"
    if not csv_path.exists():
        return {}

    import csv
    best = {"best_rank1": 0.0, "best_rank2": 0.0, "best_rank3": 0.0, "best_epoch": 0, "best_val_acc": 0.0}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r1 = float(row.get("rank@1", 0))
            if r1 > best["best_rank1"]:
                best["best_rank1"] = r1
                best["best_rank2"] = float(row.get("rank@2", 0))
                best["best_rank3"] = float(row.get("rank@3", 0))
                best["best_epoch"] = int(row.get("epoch", 0))
                best["best_val_acc"] = float(row.get("val_acc", 0))
                best["best_val_loss"] = float(row.get("val_loss", 0))
    return best


def find_run_subdir(output_dir: Path) -> Path | None:
    candidates = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name not in ("logs",)],
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main():
    args = parse_args()

    if args.experiments:
        experiment_names = [e.strip() for e in args.experiments.split(",")]
        for name in experiment_names:
            if name not in ABLATION_CONFIGS:
                print(f"ERROR: Unknown experiment '{name}'. Available: {list(ABLATION_CONFIGS.keys())}")
                sys.exit(1)
    else:
        experiment_names = ABLATION_ORDER

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_root = Path(args.output_base) / f"ablation_{timestamp}"
    ablation_root.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp": timestamp,
        "experiments": experiment_names,
        "nproc_per_node": args.nproc_per_node,
        "shared_args": {
            "batch_size": args.batch_size,
            "accumulation_steps": args.accumulation_steps,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "data_ratio": args.data_ratio,
            "early_stop": args.early_stop,
            "seed": args.seed,
        },
        "ablation_configs": {k: ABLATION_CONFIGS[k] for k in experiment_names},
    }
    with open(ablation_root / "ablation_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    results = {}
    total = len(experiment_names)
    gpu_str = f"{args.nproc_per_node} GPUs (DDP)" if args.nproc_per_node > 1 else "1 GPU"

    print(f"{'='*60}")
    print(f"CVPR Ablation Study: {total} experiments, {gpu_str}")
    print(f"Output: {ablation_root}")
    print(f"Shared: bs={args.batch_size} accum={args.accumulation_steps} epochs={args.num_epochs} "
          f"lr={args.learning_rate} data_ratio={args.data_ratio} seed={args.seed}")
    print(f"{'='*60}")

    for i, name in enumerate(experiment_names):
        exp_output_dir = ablation_root / name
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_command(args, name, str(exp_output_dir))
        flags_str = " ".join(f"{k}={v}" for k, v in ABLATION_CONFIGS[name].items()) or "(full model)"

        print(f"\n[{i+1}/{total}] Running: {name}")
        print(f"  Ablation: {flags_str}")
        print(f"  Command:  {' '.join(cmd)}")

        t0 = time.time()
        stdout_path = exp_output_dir / "stdout.log"
        stderr_path = exp_output_dir / "stderr.log"
        try:
            with open(stdout_path, "w") as f_out, open(stderr_path, "w") as f_err:
                proc = subprocess.Popen(
                    cmd, stdout=f_out, stderr=f_err,
                    start_new_session=False,  # keep in same process group for Ctrl+C
                )
                proc.wait()
        except KeyboardInterrupt:
            print(f"\n  Ctrl+C received, killing {name} ...")
            proc.kill()
            proc.wait()
            # Save partial results before exit
            results[name] = {"status": "INTERRUPTED", "time_seconds": time.time() - t0}
            with open(ablation_root / "ablation_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"Partial results saved to: {ablation_root / 'ablation_results.json'}")
            sys.exit(1)
        dt = time.time() - t0

        if proc.returncode != 0:
            print(f"  FAILED (exit code {proc.returncode}, {dt:.0f}s)")
            err_text = stderr_path.read_text()
            err_lines = err_text.strip().split("\n")
            for line in err_lines[-5:]:
                print(f"    {line}")
            results[name] = {"status": "FAILED", "time_seconds": dt}
            continue

        run_dir = find_run_subdir(exp_output_dir)
        metrics = extract_best_metrics(run_dir) if run_dir else {}

        results[name] = {"status": "OK", "time_seconds": dt, **metrics}

        r1 = metrics.get("best_rank1", 0)
        r2 = metrics.get("best_rank2", 0)
        r3 = metrics.get("best_rank3", 0)
        ep = metrics.get("best_epoch", 0)
        print(f"  OK ({dt:.0f}s) best_epoch={ep} rank@1={r1:.2f}% rank@2={r2:.2f}% rank@3={r3:.2f}%")

    with open(ablation_root / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"ABLATION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Experiment':<16} {'Status':<8} {'Rank@1':>8} {'Rank@2':>8} {'Rank@3':>8} {'Epoch':>6} {'Time':>8}")
    print(f"{'-'*80}")

    full_r1 = results.get("full", {}).get("best_rank1", 0)
    for name in experiment_names:
        r = results.get(name, {})
        status = r.get("status", "N/A")
        if status == "OK":
            r1 = r.get("best_rank1", 0)
            r2 = r.get("best_rank2", 0)
            r3 = r.get("best_rank3", 0)
            ep = r.get("best_epoch", 0)
            dt = r.get("time_seconds", 0)
            delta = f"({r1 - full_r1:+.2f})" if name != "full" and full_r1 > 0 else ""
            print(f"{name:<16} {'OK':<8} {r1:>7.2f}% {r2:>7.2f}% {r3:>7.2f}% {ep:>6} {dt:>7.0f}s {delta}")
        else:
            print(f"{name:<16} {'FAILED':<8}")

    print(f"{'='*80}")
    print(f"Results saved to: {ablation_root / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
