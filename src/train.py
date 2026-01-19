#!/usr/bin/env python3
"""MSG_VIP Training Script

Supports both single GPU and multi-GPU training using torchrun.

This file is part of the "engineering shell" and is kept stable so that:
- log/visualization/checkpoint/CSV behaviors remain consistent
- torchrun/DDP launch works the same way

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
import torch


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.runtime import TrainingRuntime
from src.configs.config import get_default_config


def setup_cuda_environment(cuda_base_path: str | None = None) -> None:
    """Setup CUDA/TensorRT environment variables (best-effort)."""
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

# Clean up stale DDP environment variables before setting new ones
# This prevents conflicts when switching from torchrun to direct python execution
stale_ddp_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", 
                  "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID"]
for var in stale_ddp_vars:
    if var in os.environ:
        del os.environ[var]

# Set random port for DDP to avoid EADDRINUSE errors
if "MASTER_PORT" not in os.environ:
    # Use random port in range 29500-65535 to avoid conflicts
    random_port = random.randint(29500, 65535)
    os.environ["MASTER_PORT"] = str(random_port)
    print(f"[INFO] Using random DDP port: {random_port}")

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*`estimate` is deprecated.*",
)


def configure_cuda_allocator() -> None:
    """Configure CUDA allocator with conservative defaults."""
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or not torch.cuda.is_available():
        return

    allocator_parts = ["max_split_size_mb:64"]
    if os.environ.get("MSGVIP_USE_EXPANDABLE_SEGMENTS") == "1":
        allocator_parts.append("expandable_segments:True")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(allocator_parts)


def resolve_data_path(path_str: str) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return str(path)


def create_output_structure(base_dir: str) -> dict:
    """Create unified output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = timestamp

    paths = {
        "run_dir": Path(base_dir) / run_name,
        "checkpoints": Path(base_dir) / run_name / "checkpoints",
        "logs": Path(base_dir) / run_name / "logs",
        "records": Path(base_dir) / run_name / "records",
        "visualizations": Path(base_dir) / run_name / "visualizations",
        "predictions": Path(base_dir) / run_name / "predictions",
        "configs": Path(base_dir) / run_name / "configs",
    }

    for name, path in paths.items():
        if name not in ["checkpoints"]:
            path.mkdir(parents=True, exist_ok=True)

    return paths


def parse_args(argv=None):
    base_config = get_default_config()

    parser = argparse.ArgumentParser(description="MSG_VIP Training")

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(base_config.data.data_dir),
        help="Data directory path (relative paths resolved from project root)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(getattr(base_config, "output_dir", "outputs")),
        help="Output directory for all results",
    )

    parser.add_argument("--batch_size", "-b", type=int, default=int(base_config.training.batch_size))
    parser.add_argument("--accumulation_steps", type=int, default=int(getattr(base_config.training, "accumulation_steps", 1)), help="Gradient accumulation steps",)
    parser.add_argument("--learning_rate", "-l", type=float, default=float(base_config.training.learning_rate))
    parser.add_argument("--num_epochs", "-e", type=int, default=int(base_config.training.num_epochs))
    parser.add_argument("--num_workers", "-w", type=int, default=int(base_config.training.num_workers))

    parser.add_argument(
        "--data_ratio",
        type=float,
        default=None,
        help="Data ratio to use (e.g., 0.01 for 1% of data). If not specified, use full dataset",
    )

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--early_stop", type=int, default=None)

    parser.add_argument("--importance_weight", type=float, default=None)
    parser.add_argument("--preference_weight", type=float, default=None)
    parser.add_argument("--pairwise_weight", type=float, default=None)
    parser.add_argument("--pairwise_margin", type=float, default=None)
    parser.add_argument("--logit_temperature", type=float, default=None)
    parser.add_argument("--feature_lr_multiplier", type=float, default=None)
    parser.add_argument("--head_lr_multiplier", type=float, default=None)
    parser.add_argument("--run_tag", type=str, default=None)

    args = parser.parse_args(argv)
    setattr(args, "_config_data_dir", str(base_config.data.data_dir))
    setattr(args, "_resolved_config_path", "src/configs/config.py")
    return args


def _print_launch_banner(args) -> None:
    is_distributed = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if is_distributed else 0
    if rank != 0:
        return

    if is_distributed:
        gpu_count = torch.distributed.get_world_size()
        print(f"🚀 分布式训练: {gpu_count} GPUs")
    else:
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"🚀 单GPU训练: {device_name}")

    print(f"📊 训练参数: batch_size={args.batch_size}, accum={getattr(args, 'accumulation_steps', 1)}, lr={args.learning_rate}, epochs={args.num_epochs}")
    if args.data_ratio:
        print(f"📉 数据比例: {args.data_ratio:.3f}")
    if getattr(args, "_data_dir_overridden", False):
        print(f"📁 数据目录: {args.data_dir}")
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

    # Only init DDP if explicitly launched with torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl")

    args = parse_args()
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
