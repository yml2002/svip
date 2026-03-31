#!/usr/bin/env python3
"""Unified experiment runner for CVPR-style experiments.

Modes:
  all         — run ablation + seed + hyperparam in sequence
  ablation    — task-driven mechanism ablations for the paper model
  seed        — multi-seed stability (full model × 3 seeds)
  hyperparam  — hyperparameter sensitivity for meaningful design knobs

Usage:
  # Run ALL experiments at once
  python src/experiments.py all --batch_size 32 --num_epochs 15 --nproc_per_node 7

  # Quick test with tiny data
  python src/experiments.py all --batch_size 2 --num_epochs 1 --data_ratio 0.05 --num_workers 4

  # Run specific mode
  python src/experiments.py ablation --batch_size 32 --num_epochs 15
  python src/experiments.py seed --experiments full_model --seeds 42,3407,2026
  python src/experiments.py hyperparam --experiments topk_4,topk_8,topk_all

  # Multi-GPU
  python src/experiments.py all --nproc_per_node 7 --batch_size 32 --num_epochs 15
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

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# Ablation: only task-driven mechanism studies used in the paper
# ============================================================
ABLATION_CONFIGS = {
    "full_model":                           {},
    "wo_relation_refinement":          {"--relation_enabled": "0"},
    "wo_scene_context_conditioning":   {"--no_global_context": None},
    "wo_temporal_social_memory":       {"--no_temporal_edges": None},
    "wo_spatial_social_interaction":   {"--no_spatial_edges": None},
    "wo_dynamic_unary_prior":          {"--mean_only_unary": None},
    "wo_geometry_prior":               {"--no_geom": None},
}
ABLATION_ORDER = [
    "full_model",
    "wo_relation_refinement",
    "wo_scene_context_conditioning",
    "wo_temporal_social_memory",
    "wo_spatial_social_interaction",
    "wo_dynamic_unary_prior",
    "wo_geometry_prior",
]

# ============================================================
# Multi-seed: full model × 3 seeds
# ============================================================
DEFAULT_SEEDS = [42, 3407, 2026]

# ============================================================
# Hyperparameter sensitivity: only the four paper-worthy groups
# ============================================================
HYPERPARAM_CONFIGS = {
    "topk_4":              {"--gat_topk_neighbors": "4"},
    "topk_8":              {},
    "topk_all":            {"--gat_topk_neighbors": "0"},
    "delta_050":           {"--relation_delta_scale": "0.50"},
    "delta_085":           {},
    "delta_120":           {"--relation_delta_scale": "1.20"},
    "pref_000":            {"--preference_weight": "0.00"},
    "pref_025":            {},
    "pref_050":            {"--preference_weight": "0.50"},
    "unfreeze_0":          {"--unfreeze_layers": "0"},
    "unfreeze_1":          {"--unfreeze_layers": "1"},
    "unfreeze_2":          {"--unfreeze_layers": "2"},
}
HYPERPARAM_ORDER = [
    "topk_4", "topk_8", "topk_all",
    "delta_050", "delta_085", "delta_120",
    "pref_000", "pref_025", "pref_050",
    "unfreeze_0", "unfreeze_1", "unfreeze_2",
]


def parse_args():
    from src.config import get_default_config
    cfg = get_default_config()
    tr = cfg.training

    p = argparse.ArgumentParser(
        description="CVPR Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("mode", choices=["all", "ablation", "seed", "hyperparam"])
    # Defaults come from config.py — no duplicate hardcoding here.
    p.add_argument("--batch_size", "-b", type=int, default=None,
                   help=f"per-GPU batch size (config default: {tr.batch_size})")
    p.add_argument("--accumulation_steps", type=int, default=None,
                   help=f"gradient accumulation steps (config default: {tr.accumulation_steps})")
    p.add_argument("--num_epochs", "-e", type=int, default=None,
                   help=f"training epochs (config default: {tr.num_epochs})")
    p.add_argument("--learning_rate", "-l", type=float, default=None,
                   help=f"learning rate (config default: {tr.learning_rate})")
    p.add_argument("--data_ratio", type=float, default=None,
                   help="fraction of data to use, (0,1] (config default: 1.0)")
    p.add_argument("--num_workers", "-w", type=int, default=None,
                   help=f"DataLoader workers (config default: {tr.num_workers})")
    p.add_argument("--roi_chunk", type=int, default=None,
                   help=f"ROI crop chunk size (config default: {tr.roi_chunk})")
    p.add_argument("--early_stop", type=int, default=None,
                   help=f"early-stop patience epochs (config default: {tr.early_stop})")
    p.add_argument("--backbone_lr_scale", type=float, default=None,
                   help=f"backbone LR scale (config default: {tr.backbone_lr_scale})")
    p.add_argument("--backbone_warmup_epochs", type=int, default=None,
                   help=f"backbone warmup epochs (config default: {tr.backbone_warmup_epochs})")
    p.add_argument("--backbone_train_mode", type=str, default=None,
                   choices=["frozen", "attn_ln", "full_block"],
                   help=f"backbone train mode (config default: {tr.backbone_train_mode})")
    p.add_argument("--seed", type=int, default=None,
                   help=f"Random seed (config default: {tr.seed})")
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
    from src.config import get_default_config as _cfg
    effective_seed = seed if seed is not None else _cfg().training.seed
    train_args = ["src/train.py", "--seed", str(effective_seed), "--output_dir", output_dir]

    # Only forward flags that were explicitly set; omitted ones use config.py defaults.
    optional = {
        "--batch_size":        args.batch_size,
        "--accumulation_steps": args.accumulation_steps,
        "--num_epochs":        args.num_epochs,
        "--learning_rate":     args.learning_rate,
        "--data_ratio":        args.data_ratio,
        "--num_workers":       args.num_workers,
        "--roi_chunk":         args.roi_chunk,
        "--early_stop":        args.early_stop,
        "--backbone_lr_scale": args.backbone_lr_scale,
        "--backbone_warmup_epochs": args.backbone_warmup_epochs,
    }
    for flag, value in optional.items():
        if value is not None:
            train_args += [flag, str(value)]

    if args.backbone_train_mode is not None:
        train_args += ["--backbone_train_mode", str(args.backbone_train_mode)]

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
    proc = None

    try:
        with open(out_dir / "stdout.log", "w") as fo, \
             open(out_dir / "stderr.log", "w") as fe:

            proc = subprocess.Popen(
                cmd,
                stdout=fo,
                stderr=fe,
                start_new_session=True,
            )

            # non-blocking wait
            while proc.poll() is None:
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    continue

    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt received, killing all child processes...")

        if proc is not None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:
                pass

            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

        # 保险：杀 torchrun
        subprocess.run(
            ["pkill", "-f", "torchrun"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print("✅ All processes killed")
        return {
            "status": "INTERRUPTED",
            "time_seconds": time.time() - t0
        }

    dt = time.time() - t0

    if proc.returncode != 0:
        err = (out_dir / "stderr.log").read_text().strip().split("\n")
        print(f"  FAILED ({dt:.0f}s)")
        for l in err[-3:]:
            print(f"    {l}")
        return {"status": "FAILED", "time_seconds": dt}

    rd = find_run_subdir(out_dir)
    m = extract_best(rd) if rd else {}
    print(
        f"  OK ({dt:.0f}s) ep={m.get('best_epoch',0)} "
        f"r@1={m.get('best_rank1',0):.2f}% "
        f"r@2={m.get('best_rank2',0):.2f}% "
        f"r@3={m.get('best_rank3',0):.2f}%"
    )
    return {"status": "OK", "time_seconds": dt, **m}

def print_table(results: dict, order: list, ref_key: str = "full_model"):
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
    experiments = [e.strip() for e in args.experiments.split(",")] if args.experiments else ["full_model"]

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

    from src.config import get_default_config
    _cfg = get_default_config().training
    # Resolve seed once so all sub-functions see a concrete int
    if args.seed is None:
        args.seed = _cfg.seed
    # Effective values: CLI arg if set, otherwise config.py default
    eff_bs     = args.batch_size        or _cfg.batch_size
    eff_accum  = args.accumulation_steps or _cfg.accumulation_steps
    eff_epochs = args.num_epochs        or _cfg.num_epochs
    eff_lr     = args.learning_rate     or _cfg.learning_rate
    eff_ratio  = args.data_ratio        or 1.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_base) / f"experiments_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)

    meta = {"mode": args.mode, "timestamp": timestamp,
            "args": {k: v for k, v in vars(args).items() if v is not None}}
    with open(root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    gpu_str = f"{args.nproc_per_node} GPUs" if args.nproc_per_node > 1 else "1 GPU"
    print(f"{'='*60}")
    print(f"CVPR Experiments — mode={args.mode} — {gpu_str}")
    print(f"bs={eff_bs} accum={eff_accum} epochs={eff_epochs} "
          f"lr={eff_lr} data_ratio={eff_ratio}")
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
