#!/usr/bin/env python3
"""Unified experiment runner for CVPR submission.

Modes:
  all         — run ablation + seed + hyperparam in sequence
  ablation    — ablation study (6 experiments)
  seed        — multi-seed stability (full model × 3 seeds)
  hyperparam  — hyperparameter sensitivity (6 params, 20 configs)

Usage:
  # Run ALL experiments at once
  python src/run_experiments.py all --batch_size 32 --num_epochs 15 --nproc_per_node 7

  # Quick test with tiny data
  python src/run_experiments.py all --batch_size 2 --num_epochs 1 --data_ratio 0.05 --num_workers 4

  # Run specific mode
  python src/run_experiments.py ablation --batch_size 32 --num_epochs 15
  python src/run_experiments.py seed --experiments full --seeds 42,3407,2026
  python src/run_experiments.py hyperparam --experiments lr_1e5,lr_5e5,lr_1e4

  # Multi-GPU
  python src/run_experiments.py all --nproc_per_node 7 --batch_size 32 --num_epochs 15
"""

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# Ablation: 6 experiments — each removes one design component
# ============================================================
ABLATION_CONFIGS = {
    "full":         {},
    "self_only":    {"--relation_enabled": "0"},
    "rel_only":     {"--self_enabled": "0"},
    "no_gat":       {"--no_gat": None},
    "no_temporal":  {"--no_temporal_edges": None},
    "no_edge_feat": {"--no_edge_features": None},
    "no_geom":      {"--no_geom": None},
    "gcn":          {"--graph_type": "gcn"},
}
ABLATION_ORDER = ["full", "self_only", "rel_only", "no_gat", "gcn", "no_temporal", "no_edge_feat", "no_geom"]

# ============================================================
# Multi-seed: full model × 3 seeds
# ============================================================
DEFAULT_SEEDS = [42, 3407, 2026]

# ============================================================
# Hyperparameter sensitivity: 6 params × 3-5 values = 20 configs
# ============================================================
HYPERPARAM_CONFIGS = {
    # GAT layers: {1, 2*, 3}
    "gat_layers_1":  {"--gat_num_layers": "1"},
    "gat_layers_2":  {},
    "gat_layers_3":  {"--gat_num_layers": "3"},
    # GAT heads: {2, 4*, 8}
    "gat_heads_2":   {"--gat_heads": "2"},
    "gat_heads_4":   {},
    "gat_heads_8":   {"--gat_heads": "8"},
    # Temporal window: {1, 3*, 5}
    "tw_1":          {"--temporal_window": "1"},
    "tw_3":          {},
    "tw_5":          {"--temporal_window": "5"},
    # TopK neighbors: {2, 4*, all}
    "topk_2":        {"--gat_topk_neighbors": "2"},
    "topk_4":        {},
    "topk_all":      {"--gat_topk_neighbors": "0"},
    # Learning rate: log-uniform {1e-5, 2e-5, 5e-5*, 1e-4, 2e-4}
    "lr_1e5":        {"--learning_rate": "1e-5"},
    "lr_2e5":        {"--learning_rate": "2e-5"},
    "lr_5e5":        {},
    "lr_1e4":        {"--learning_rate": "1e-4"},
    "lr_2e4":        {"--learning_rate": "2e-4"},
    # DINOv2 unfreeze: {0(frozen), 1*(default)}
    "unfreeze_0":    {"--unfreeze_layers": "0"},
    "unfreeze_1":    {},
}
HYPERPARAM_ORDER = [
    "gat_layers_1", "gat_layers_2", "gat_layers_3",
    "gat_heads_2", "gat_heads_4", "gat_heads_8",
    "tw_1", "tw_3", "tw_5",
    "topk_2", "topk_4", "topk_all",
    "lr_1e5", "lr_2e5", "lr_5e5", "lr_1e4", "lr_2e4",
    "unfreeze_0", "unfreeze_1",
]


def parse_args():
    p = argparse.ArgumentParser(description="CVPR Experiment Runner")
    p.add_argument("mode", choices=["all", "ablation", "seed", "hyperparam"])
    p.add_argument("--batch_size", "-b", type=int, default=32)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--num_epochs", "-e", type=int, default=15)
    p.add_argument("--learning_rate", "-l", type=float, default=5e-5)
    p.add_argument("--data_ratio", type=float, default=1.0)
    p.add_argument("--num_workers", "-w", type=int, default=8)
    p.add_argument("--roi_chunk", type=int, default=512)
    p.add_argument("--early_stop", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seeds for seed mode")
    p.add_argument("--nproc_per_node", type=int, default=1)
    p.add_argument("--experiments", type=str, default=None,
                   help="Comma-separated experiment subset")
    p.add_argument("--output_base", type=str, default="outputs")
    return p.parse_args()


_next_port = 29500


def build_command(args, extra_flags: dict, output_dir: str, seed: int) -> list[str]:
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
        "--seed", str(seed),
        "--output_dir", output_dir,
    ]
    for flag, value in extra_flags.items():
        train_args.append(flag)
        if value is not None:
            train_args.append(str(value))

    if args.nproc_per_node > 1:
        port = _next_port; _next_port += 1
        torchrun = shutil.which("torchrun")
        launcher = [torchrun] if torchrun else [sys.executable, "-m", "torch.distributed.run"]
        return launcher + ["--nproc_per_node", str(args.nproc_per_node),
                           "--master_port", str(port)] + train_args
    return [sys.executable] + train_args


def extract_best(run_dir: Path) -> dict:
    csv_path = run_dir / "records" / "losses.csv"
    if not csv_path.exists():
        return {}
    best = {"best_rank1": 0.0, "best_rank2": 0.0, "best_rank3": 0.0, "best_epoch": 0}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            r1 = float(row.get("rank@1", 0))
            if r1 > best["best_rank1"]:
                best = {"best_rank1": r1, "best_rank2": float(row.get("rank@2", 0)),
                        "best_rank3": float(row.get("rank@3", 0)),
                        "best_epoch": int(row.get("epoch", 0)),
                        "best_val_acc": float(row.get("val_acc", 0)),
                        "best_val_loss": float(row.get("val_loss", 0))}
    return best


def find_run_subdir(d: Path) -> Path | None:
    cands = sorted([x for x in d.iterdir() if x.is_dir() and x.name != "logs"],
                   key=lambda x: x.name, reverse=True)
    return cands[0] if cands else None


def run_one(args, name: str, flags: dict, out_dir: Path, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args, flags, str(out_dir), seed)
    flags_str = " ".join(f"{k}={v}" for k, v in flags.items()) or "(default)"
    print(f"  Config: {flags_str}  seed={seed}")

    t0 = time.time()
    try:
        with open(out_dir / "stdout.log", "w") as fo, open(out_dir / "stderr.log", "w") as fe:
            # start_new_session=True puts child in its own process group
            # so we can kill the entire group (torchrun + all workers)
            proc = subprocess.Popen(cmd, stdout=fo, stderr=fe, start_new_session=True)
            proc.wait()
    except KeyboardInterrupt:
        # Kill entire process group (torchrun + worker processes)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        print(f"\n  Interrupted, all child processes killed.")
        return {"status": "INTERRUPTED", "time_seconds": time.time() - t0}
    dt = time.time() - t0

    if proc.returncode != 0:
        err = (out_dir / "stderr.log").read_text().strip().split("\n")
        print(f"  FAILED ({dt:.0f}s)")
        for l in err[-3:]: print(f"    {l}")
        return {"status": "FAILED", "time_seconds": dt}

    rd = find_run_subdir(out_dir)
    m = extract_best(rd) if rd else {}
    print(f"  OK ({dt:.0f}s) ep={m.get('best_epoch',0)} "
          f"r@1={m.get('best_rank1',0):.2f}% r@2={m.get('best_rank2',0):.2f}% r@3={m.get('best_rank3',0):.2f}%")
    return {"status": "OK", "time_seconds": dt, **m}


def print_table(results: dict, order: list, ref_key: str = "full"):
    ref_r1 = results.get(ref_key, {}).get("best_rank1", 0)
    print(f"\n{'='*80}")
    print(f"{'Experiment':<16} {'Status':<8} {'Rank@1':>8} {'Rank@2':>8} {'Rank@3':>8} {'Ep':>4} {'Time':>7}")
    print(f"{'-'*80}")
    for n in order:
        r = results.get(n, {})
        if r.get("status") == "OK":
            r1, r2, r3 = r["best_rank1"], r["best_rank2"], r["best_rank3"]
            delta = f"({r1-ref_r1:+.2f})" if n != ref_key and ref_r1 > 0 else ""
            print(f"{n:<16} {'OK':<8} {r1:>7.2f}% {r2:>7.2f}% {r3:>7.2f}% {r.get('best_epoch',0):>4} "
                  f"{r.get('time_seconds',0):>6.0f}s {delta}")
        else:
            print(f"{n:<16} {r.get('status','N/A'):<8}")
    print(f"{'='*80}")


# ============================================================
# Mode: ablation
# ============================================================
def do_ablation(args, root: Path):
    order = ABLATION_ORDER
    if args.experiments:
        order = [e.strip() for e in args.experiments.split(",")]
        for n in order:
            if n not in ABLATION_CONFIGS:
                print(f"ERROR: unknown ablation '{n}'. Available: {list(ABLATION_CONFIGS.keys())}"); sys.exit(1)

    print(f"\n{'#'*60}")
    print(f"# ABLATION STUDY: {len(order)} experiments")
    print(f"{'#'*60}")

    results = {}
    for i, n in enumerate(order):
        print(f"\n[Ablation {i+1}/{len(order)}] {n}")
        results[n] = run_one(args, n, ABLATION_CONFIGS[n], root / "ablation" / n, args.seed)
        if results[n].get("status") == "INTERRUPTED":
            with open(root / "ablation_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print_table(results, order)
            raise KeyboardInterrupt

    with open(root / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print_table(results, order)
    return results


# ============================================================
# Mode: seed
# ============================================================
def do_seed(args, root: Path):
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS
    experiments = [e.strip() for e in args.experiments.split(",")] if args.experiments else ["full"]

    print(f"\n{'#'*60}")
    print(f"# MULTI-SEED: {len(experiments)} exp × {len(seeds)} seeds")
    print(f"# Seeds: {seeds}")
    print(f"{'#'*60}")

    all_results = {}
    idx = 0; total = len(experiments) * len(seeds)
    for exp in experiments:
        if exp not in ABLATION_CONFIGS:
            print(f"ERROR: unknown '{exp}'"); sys.exit(1)
        all_results[exp] = {}
        for seed in seeds:
            idx += 1
            tag = f"{exp}_s{seed}"
            print(f"\n[Seed {idx}/{total}] {tag}")
            r = run_one(args, tag, ABLATION_CONFIGS[exp], root / "seed" / tag, seed)
            all_results[exp][str(seed)] = r
            if r.get("status") == "INTERRUPTED":
                raise KeyboardInterrupt

    # Summarize
    import numpy as np
    summary = {}
    print(f"\n{'='*70}")
    print(f"MULTI-SEED SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<14} {'Rank@1':>18} {'Rank@2':>18} {'Rank@3':>18}")
    print(f"{'-'*70}")
    for exp in experiments:
        sr = all_results[exp]
        r1s = [v["best_rank1"] for v in sr.values() if v.get("status") == "OK"]
        r2s = [v["best_rank2"] for v in sr.values() if v.get("status") == "OK"]
        r3s = [v["best_rank3"] for v in sr.values() if v.get("status") == "OK"]
        if r1s:
            s = {"rank1_mean": np.mean(r1s), "rank1_std": np.std(r1s),
                 "rank2_mean": np.mean(r2s), "rank2_std": np.std(r2s),
                 "rank3_mean": np.mean(r3s), "rank3_std": np.std(r3s),
                 "n": len(r1s), "per_seed": sr}
            summary[exp] = s
            print(f"{exp:<14} {s['rank1_mean']:>6.2f} ± {s['rank1_std']:<6.2f}"
                  f"  {s['rank2_mean']:>6.2f} ± {s['rank2_std']:<6.2f}"
                  f"  {s['rank3_mean']:>6.2f} ± {s['rank3_std']:<6.2f}")
    print(f"{'='*70}")

    with open(root / "seed_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


# ============================================================
# Mode: hyperparam
# ============================================================
def do_hyperparam(args, root: Path):
    order = HYPERPARAM_ORDER
    if args.experiments:
        order = [e.strip() for e in args.experiments.split(",")]
        for n in order:
            if n not in HYPERPARAM_CONFIGS:
                print(f"ERROR: unknown hp '{n}'. Available: {list(HYPERPARAM_CONFIGS.keys())}"); sys.exit(1)

    # Skip default duplicates — only run configs that differ from full default
    dedup_order = []
    seen_default = False
    for n in order:
        if not HYPERPARAM_CONFIGS[n]:  # empty = default
            if not seen_default:
                dedup_order.append(n)
                seen_default = True
            # else skip duplicate default
        else:
            dedup_order.append(n)
    order = dedup_order

    print(f"\n{'#'*60}")
    print(f"# HYPERPARAMETER SENSITIVITY: {len(order)} configs")
    print(f"{'#'*60}")

    results = {}
    for i, n in enumerate(order):
        print(f"\n[HP {i+1}/{len(order)}] {n}")
        results[n] = run_one(args, n, HYPERPARAM_CONFIGS[n], root / "hyperparam" / n, args.seed)
        if results[n].get("status") == "INTERRUPTED":
            with open(root / "hyperparam_results.json", "w") as f:
                json.dump(results, f, indent=2)
            raise KeyboardInterrupt

    # Use first default as reference
    ref = next((n for n in order if not HYPERPARAM_CONFIGS.get(n, {"x": 1})), order[0])
    with open(root / "hyperparam_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print_table(results, order, ref_key=ref)
    return results


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_base) / f"experiments_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)

    # Save meta
    meta = {"mode": args.mode, "timestamp": timestamp,
            "args": {k: v for k, v in vars(args).items() if v is not None}}
    with open(root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    gpu_str = f"{args.nproc_per_node} GPUs" if args.nproc_per_node > 1 else "1 GPU"
    print(f"{'='*60}")
    print(f"CVPR Experiments — mode={args.mode} — {gpu_str}")
    print(f"bs={args.batch_size} accum={args.accumulation_steps} epochs={args.num_epochs} "
          f"lr={args.learning_rate} data_ratio={args.data_ratio}")
    print(f"Output: {root}")
    print(f"{'='*60}")

    if args.mode == "ablation":
        do_ablation(args, root)
    elif args.mode == "seed":
        do_seed(args, root)
    elif args.mode == "hyperparam":
        do_hyperparam(args, root)
    elif args.mode == "all":
        try:
            do_ablation(args, root)
            do_seed(args, root)
            do_hyperparam(args, root)
        except KeyboardInterrupt:
            print(f"\nInterrupted. Partial results saved in: {root}")
            sys.exit(1)

    print(f"\nAll done. Results in: {root}")


if __name__ == "__main__":
    main()
