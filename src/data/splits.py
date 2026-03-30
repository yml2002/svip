"""Dataset split construction helpers.

Responsible for:
- Resolving split directories and applying data_ratio sampling
- Building train/val file lists
- Constructing VideoDataset instances ready for DataLoader
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.data.dataset import VideoDataset

logger = logging.getLogger(__name__)


class _FileListDataset(VideoDataset):
    """VideoDataset that uses an explicit pre-built file list instead of scanning a directory."""

    def __init__(self, file_list: List[Path], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.file_list = [str(p) for p in file_list]


def _load_split_files(
    data_root: Path,
    split: str,
    ratio: float,
    *,
    required: bool = True,
) -> List[Path]:
    """Return NPZ file list for a split, trimmed to ratio."""
    split_dir = data_root / split
    if not split_dir.exists():
        if required:
            raise ValueError(f"Split directory not found: {split_dir}")
        return []

    files = sorted(split_dir.glob("*.npz"))
    if not files:
        if required:
            raise ValueError(f"No NPZ files found in {split_dir}")
        return []

    if ratio < 1.0:
        keep = max(1, int(math.ceil(len(files) * ratio)))
        files = files[:keep]

    return list(files)


def build_train_val_datasets(
    config: Any,
) -> Tuple[_FileListDataset, _FileListDataset]:
    """Build train and val VideoDataset instances from config.

    Returns:
        (train_dataset, val_dataset)
    """
    data_cfg = config.data
    data_root = Path(data_cfg.data_dir)
    ratio = float(getattr(data_cfg, "data_ratio", 1.0) or 1.0)

    train_split_names: List[str] = list(getattr(data_cfg, "train_splits", ["train", "test"]))
    val_split_name: str = str(getattr(data_cfg, "val_split", "val"))

    # Collect training files from all configured splits
    train_pool: List[Path] = []
    split_counts: Dict[str, int] = {}
    for split_name in train_split_names:
        files = _load_split_files(data_root, split_name, ratio, required=False)
        split_counts[split_name] = len(files)
        train_pool.extend(files)

    train_pool = sorted(train_pool)
    if not train_pool:
        raise ValueError(
            f"No NPZ files found for train_splits={train_split_names} under {data_root}"
        )

    missing = [name for name, cnt in split_counts.items() if cnt == 0]
    if missing:
        logger.warning("Some train_splits are empty/missing: %s", ", ".join(missing))

    logger.info(
        "Training splits=%s -> pool=%d (%s)",
        train_split_names,
        len(train_pool),
        ", ".join(f"{k}:{v}" for k, v in split_counts.items()),
    )

    val_files = _load_split_files(data_root, val_split_name, ratio=ratio, required=True)

    train_ds = _FileListDataset(
        train_pool, config=config, data_path=str(data_root), split="train", max_samples=None
    )
    val_ds = _FileListDataset(
        val_files, config=config, data_path=str(data_root), split="val", max_samples=None
    )

    logger.info(
        "Dataset sizes: data_ratio=%.3f  train=%d  val=%d",
        ratio,
        len(train_ds),
        len(val_ds),
    )
    return train_ds, val_ds
