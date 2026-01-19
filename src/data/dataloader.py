"""Data processing module for MSG_VIP.

Loads NPZ samples with fields:
- frames: (T,H,W,3) uint8 or float, converted to (T,3,H,W) float16 in [0,1]
- bboxes: (T,N,4) in absolute pixel coords or normalized; we normalize to [0,1]
- person_mask: (T,N) bool
- frame_mask: (T,) bool
- target_index: scalar int64
- video_id, scene_category

This is kept compatible with the existing tests/NPZ format in the repo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MSGVIPDataset(Dataset):
    def __init__(
        self,
        config: Any,
        data_path: Optional[str] = None,
        split: str = "train",
        cache_data: Optional[bool] = None,
        max_samples: Optional[int] = None,
        **kwargs,
    ) -> None:
        # Contract: runtime always passes an ExperimentConfig with .data present.
        self.config = config.data

        self.data_path = Path(data_path) if data_path is not None else Path(self.config.data_dir)

        self.split = split
        self.cache_data = bool(self.config.cache_data) if cache_data is None else bool(cache_data)
        self.max_samples = self.config.max_samples if max_samples is None else max_samples

        self.num_frames = int(self.config.video_length)
        self.frame_height = int(self.config.image_height)
        self.frame_width = int(self.config.image_width)
        self.num_person_slots = int(self.config.max_persons)

        self.file_list = self._load_file_list()
        self.data_cache = {} if self.cache_data else None

        logger.info("Loaded %d samples from %s split", len(self.file_list), self.split)

    def _load_file_list(self) -> List[str]:
        split_path = self.data_path / self.split
        if not split_path.exists():
            raise FileNotFoundError(f"Split directory not found: {split_path}")
        npz_files = sorted(split_path.glob("*.npz"))
        if not npz_files:
            raise ValueError(f"No NPZ files found in {split_path}")
        if self.max_samples is not None:
            npz_files = npz_files[: int(self.max_samples)]
        return [str(p) for p in npz_files]

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.cache_data and self.data_cache is not None and idx in self.data_cache:
            cached = self.data_cache[idx]
            return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in cached.items()}

        file_path = self.file_list[idx]
        data = np.load(file_path, allow_pickle=True)
        sample = self._extract_sample_data(data, file_path)

        # training-time slot permutation to prevent index overfit
        if self.split == "train":
            N = self.num_person_slots
            perm = torch.randperm(N)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(N)

            sample["bboxes"] = sample["bboxes"][:, perm, :]
            sample["person_ids"] = sample["person_ids"][:, perm]
            sample["person_mask"] = sample["person_mask"][:, perm]
            sample["original_ids"] = sample["original_ids"][perm]
            sample["target_index"] = inv_perm[sample["target_index"]]

        if self.cache_data and self.data_cache is not None:
            self.data_cache[idx] = sample

        return sample

    def _extract_sample_data(self, data: np.lib.npyio.NpzFile, file_path: str) -> Dict[str, Any]:
        try:
            frames = data["frames"]
            bboxes = data["bboxes"]
            person_ids = data["person_ids"]
            person_mask = data["person_mask"]
            frame_mask = data["frame_mask"]
            original_ids = data["original_ids"]
            target_index = data["target_index"]
            video_id = data["video_id"].item() if hasattr(data["video_id"], "item") else str(data["video_id"])
            scene_category = data["scene_category"].item() if hasattr(data["scene_category"], "item") else str(data["scene_category"])
        except KeyError as e:
            raise ValueError(f"Missing field in {file_path}: {e}")
        slots = int(self.num_person_slots)
        T, n_in = int(bboxes.shape[0]), int(bboxes.shape[1])
        target_index_int = int(target_index)

        if n_in > slots:
            if target_index_int >= slots:
                raise ValueError(
                    f"target_index {target_index_int} cannot be represented with max_persons={slots} (file N={n_in}) in {file_path}. "
                    f"Increase config.data.max_persons or fix dataset packing."
                )
            bboxes = bboxes[:, :slots, :]
            person_ids = person_ids[:, :slots]
            person_mask = person_mask[:, :slots]
            original_ids = original_ids[:slots]
        elif n_in < slots:
            pad_n = slots - n_in
            bboxes = np.concatenate([bboxes, np.zeros((T, pad_n, 4), dtype=bboxes.dtype)], axis=1)
            person_ids = np.concatenate([person_ids, np.full((T, pad_n), -1, dtype=person_ids.dtype)], axis=1)
            person_mask = np.concatenate([person_mask, np.zeros((T, pad_n), dtype=person_mask.dtype)], axis=1)
            original_ids = np.concatenate([original_ids, np.full((pad_n,), -1, dtype=original_ids.dtype)], axis=0)

        frame_tensor = torch.from_numpy(frames)
        if frame_tensor.dtype != torch.uint8:
            frame_tensor = frame_tensor.to(torch.float32)
            if frame_tensor.max() > 1.5:
                frame_tensor = frame_tensor / 255.0
        else:
            frame_tensor = frame_tensor.to(torch.float16).div_(255.0)

        if frame_tensor.ndim != 4 or frame_tensor.shape[-1] != 3:
            raise ValueError(f"frames must be (T,H,W,3), got {tuple(frame_tensor.shape)}")
        frame_tensor = frame_tensor.permute(0, 3, 1, 2).to(dtype=torch.float16)

        bbox_tensor = torch.from_numpy(bboxes).to(dtype=torch.float32)
        # normalize bbox to [0,1] (assume input may be pixel coords)
        # if values already between 0 and 1, this is a no-op (clamp).
        bbox_tensor[..., [0, 2]] = bbox_tensor[..., [0, 2]].clamp(0, float(self.frame_width)) / float(self.frame_width)
        bbox_tensor[..., [1, 3]] = bbox_tensor[..., [1, 3]].clamp(0, float(self.frame_height)) / float(self.frame_height)
        bbox_tensor = bbox_tensor.clamp(0.0, 1.0).to(dtype=torch.float16)

        frame_mask_tensor = torch.from_numpy(frame_mask.astype(np.bool_))
        person_mask_tensor = torch.from_numpy(person_mask.astype(np.bool_))

        frame_tensor[~frame_mask_tensor] = 0.0
        person_mask_tensor = person_mask_tensor & frame_mask_tensor.unsqueeze(-1)
        bbox_tensor = bbox_tensor.masked_fill(~person_mask_tensor.unsqueeze(-1), 0.0)

        return {
            "frames": frame_tensor,
            "bboxes": bbox_tensor,
            "person_ids": torch.from_numpy(person_ids.astype(np.int64)),
            "person_mask": person_mask_tensor,
            "frame_mask": frame_mask_tensor,
            "original_ids": torch.from_numpy(original_ids.astype(np.int64)),
            "target_index": torch.tensor(int(target_index_int), dtype=torch.int64),
            "video_id": video_id,
            "scene_category": scene_category,
        }
