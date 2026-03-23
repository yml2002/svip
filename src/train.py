#!/usr/bin/env python3
"""MSG_VIP Training Script

Supports both single GPU and multi-GPU training using torchrun.

Usage:
    python src/train.py --data_dir <data_dir>
    torchrun --nproc_per_node=2 src/train.py --data_dir <data_dir>
"""

import argparse
import logging
import os
import random
import signal
import sys
import warnings
from datetime import datetime
from pathlib import Path

import faulthandler
import numpy as np
import torch


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.runtime import TrainingRuntime
from src.configs.config import get_default_config


def setup_cuda_environment(cuda_base_path: str | None = None) -> None:
    if cuda_base_path is None:
        cuda_base_path = "/usr/local/cuda-12.8"
    cuda_lib_paths = [
        f"{cuda_base_path}/targets/x86_64-linux/lib",
        f"{cuda_base_path}/lib64",
    ]
    existing_paths = [p for p in cuda_lib_paths if os.path.exists(p)]
    if not existing_paths:
        return
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld_path = ":".join(existing_paths + ([current_ld_path] if current_ld_path else []))
    os.environ["LD_LIBRARY_PATH"] = new_ld_path


setup_cuda_environment()

warnings.filterwarnings("ignore", category=FutureWarning, message=r".*`estimate` is deprecated.*")


def configure_cuda_allocator() -> None:
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or not torch.cuda.is_available():
        return
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"


def resolve_data_path(path_str: str) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return str(path)


def create_output_structure(base_dir: str) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {
        "run_dir": Path(base_dir) / timestamp,
        "checkpoints": Path(base_dir) / timestamp / "checkpoints",
        "logs": Path(base_dir) / timestamp / "logs",
        "records": Path(base_dir) / timestamp / "records",
        "visualizations": Path(base_dir) / timestamp / "visualizations",
        "predictions": Path(base_dir) / timestamp / "predictions",
        "configs": Path(base_dir) / timestamp / "configs",
    }
    for name, path in paths.items():
        if name != "checkpoints":
            path.mkdir(parents=True, exist_ok=True)
    return paths


def set_global_seed(seed: int) -> None:
    s = int(seed)
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def parse_args(argv=None):
    base_config = get_default_config()
    parser = argparse.ArgumentParser(description="MSG_VIP Training")

    parser.add_argument("--data_dir", type=str, default=str(base_config.data.data_dir))
    parser.add_argument("--output_dir", type=str, default=str(base_config.output_dir))
    parser.add_argument("--batch_size", "-b", type=int, default=int(base_config.training.batch_size))
    parser.add_argument("--accumulation_steps", type=int, default=int(base_config.training.accumulation_steps))
    parser.add_argument("--learning_rate", "-l", type=float, default=float(base_config.training.learning_rate))
    parser.add_argument("--num_epochs", "-e", type=int, default=int(base_config.training.num_epochs))
    parser.add_argument("--num_workers", "-w", type=int, default=int(base_config.training.num_workers))
    parser.add_argument("--roi_chunk", type=int, default=None)
    parser.add_argument("--data_ratio", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--early_stop", type=int, default=None)

    # Loss weights
    parser.add_argument("--importance_weight", type=float, default=None)
    parser.add_argument("--preference_weight", type=float, default=None)
    # Architecture toggles
    parser.add_argument("--no_gat", action="store_true")
    parser.add_argument("--no_temporal_edges", action="store_true")
    parser.add_argument("--no_edge_features", action="store_true")
    parser.add_argument("--no_geom", action="store_true")
    parser.add_argument("--graph_type", type=str, default=None, choices=["gatv2", "gcn"])
    parser.add_argument("--gat_topk_neighbors", type=int, default=None)
    parser.add_argument("--gat_num_layers", type=int, default=None)
    parser.add_argument("--gat_heads", type=int, default=None)
    parser.add_argument("--temporal_window", type=int, default=None)
    parser.add_argument("--self_enabled", type=int, choices=[0, 1], default=None)
    parser.add_argument("--relation_enabled", type=int, choices=[0, 1], default=None)
    parser.add_argument("--unfreeze_layers", type=int, default=None)
    parser.add_argument("--logit_temperature", type=float, default=None)

    parser.add_argument("--swap_splits", action="store_true")
    parser.add_argument("--swap_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args(argv)
    setattr(args, "_config_data_dir", str(base_config.data.data_dir))
    return args


def _print_launch_banner(args) -> None:
    is_distributed = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if is_distributed else 0
    if rank != 0:
        return

    if is_distributed:
        print(f"Distributed training: {torch.distributed.get_world_size()} GPUs")
    else:
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"Single GPU training: {device_name}")

    print(f"batch_size={args.batch_size}, accum={args.accumulation_steps}, lr={args.learning_rate}, epochs={args.num_epochs}")
    if args.data_ratio:
        print(f"data_ratio={args.data_ratio:.3f}")
    print("=" * 50)


def main() -> int:
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    try:
        faulthandler.enable(all_threads=True)
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except Exception:
        pass

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl")

    args = parse_args()
    set_global_seed(args.seed)
    raw_data_dir_arg = args.data_dir
    args.data_dir = resolve_data_path(args.data_dir)
    config_default_dir = getattr(args, "_config_data_dir", raw_data_dir_arg)
    setattr(args, "_data_dir_overridden", raw_data_dir_arg != config_default_dir)

    _print_launch_banner(args)

    runtime = TrainingRuntime(
        args=args,
        project_root=project_root,
        output_creator=create_output_structure,
        cuda_allocator=configure_cuda_allocator,
    )

    exit_code = runtime.run()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
