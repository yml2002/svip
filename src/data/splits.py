"""Dataset split construction helpers.

Responsible for:
- Resolving split directories and applying data_ratio sampling
- Building train/val file lists (with optional stratified swap)
- Constructing VideoDataset instances ready for DataLoader
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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


def _swap_split_files(
    train_files: List[Path],
    val_files: List[Path],
    swap_fraction: float,
) -> Tuple[List[Path], List[Path]]:
    """Stratified swap of a fraction of val into train (and vice-versa).

    Buckets samples by (scene_category, n_persons) so both splits stay balanced.
    """
    def _bucket_key(npz_path: Path) -> Tuple[str, int]:
        d = np.load(npz_path, allow_pickle=True)
        sc = d["scene_category"].item() if hasattr(d["scene_category"], "item") else str(d["scene_category"])
        pm = d["person_mask"].astype(bool)
        n_people = int(pm.any(axis=0).sum())
        return (str(sc), n_people)

    target_train = len(train_files)
    target_val = len(val_files)
    old_train_set = set(train_files)
    old_val_set = set(val_files)

    buckets: Dict[Tuple[str, int], List[Path]] = defaultdict(list)
    for p in train_files + val_files:
        buckets[_bucket_key(p)].append(p)

    new_train: List[Path] = []
    new_val: List[Path] = []

    for key in sorted(buckets.keys()):
        group = sorted(buckets[key])
        g_train = [p for p in group if p in old_train_set]
        g_val = [p for p in group if p in old_val_set]

        k = int(round(len(g_val) * swap_fraction))
        k = max(0, min(len(g_val), len(g_train), k))

        new_train.extend(g_train[k:])
        new_train.extend(g_val[:k])
        new_val.extend(g_val[k:])
        new_val.extend(g_train[:k])

    def _pad_or_trim(primary: List[Path], secondary: List[Path], target: int) -> List[Path]:
        if len(primary) >= target:
            return primary[:target]
        need = target - len(primary)
        return primary + secondary[:need]

    new_train = _pad_or_trim(new_train, new_val, target_train)
    new_val = _pad_or_trim(new_val, new_train, target_val)
    return new_train, new_val


def build_train_val_datasets(
    config: Any,
    *,
    swap_splits: bool = False,
    swap_fraction: float = 0.5,
) -> Tuple[_FileListDataset, _FileListDataset]:
    """Build train and val VideoDataset instances from config.

    Args:
        config: ExperimentConfig
        swap_splits: if True, perform stratified cross-split swap
        swap_fraction: fraction of val to swap into train

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

    val_files = _load_split_files(data_root, val_split_name, ratio=1.0, required=True)

    if swap_splits and swap_fraction > 0.0:
        train_pool, val_files = _swap_split_files(train_pool, val_files, swap_fraction)

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
