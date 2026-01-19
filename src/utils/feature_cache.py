"""Feature pre-extraction and caching for faster training iterations."""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_config_hash(config: Any) -> str:
    relevant_keys = [
        config.model.features.dino.model_dir,
        config.model.features.dino.feature_dim,
        config.model.features.dino.image_size,
        config.model.features.bbox_geom.feature_dim,
        config.model.features.bbox_geom.hidden_dim,
        config.data.video_length,
        config.data.max_persons,
        config.data.image_height,
        config.data.image_width,
    ]
    config_str = str(relevant_keys)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


class FeatureExtractor:
    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        
        config_hash = compute_config_hash(config)
        
        # Automatically locate project root (same pattern as train.py)
        project_root = Path(__file__).parent.parent.parent
        cache_base = project_root / "feature_cache_tmp"
        
        self.cache_dir = cache_base / config_hash
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.cache_dir / "metadata.txt"
        if not self.metadata_file.exists():
            with open(self.metadata_file, "w") as f:
                f.write(f"config_hash: {config_hash}\n")
                f.write(f"dino_model: {config.model.features.dino.model_dir}\n")
                f.write(f"dino_dim: {config.model.features.dino.feature_dim}\n")
                f.write(f"geom_dim: {config.model.features.bbox_geom.feature_dim}\n")

        logger.info("Feature cache directory: %s", self.cache_dir)

    def extract_features_from_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        frames = batch["frames"].to(self.device)
        bboxes = batch["bboxes"].to(self.device)
        person_mask = batch["person_mask"].to(self.device)
        frame_mask = batch.get("frame_mask")
        if frame_mask is not None:
            frame_mask = frame_mask.to(self.device)

        B, T, N = person_mask.shape
        fm = frame_mask if frame_mask is not None else person_mask.any(dim=-1)
        pm = person_mask & fm.unsqueeze(-1)

        from src.models.importance_ranker import roi_crop_batch

        roi_chunk = int(getattr(self.config.training, "roi_chunk", 256))
        crops = roi_crop_batch(
            frames,
            bboxes,
            pm,
            fm,
            out_size=int(self.config.model.features.dino.image_size),
            roi_chunk=roi_chunk,
        )

        vis_feats = crops.new_zeros(
            (B, T, N, int(self.config.model.features.dino.feature_dim))
        )
        valid_idx = pm.nonzero(as_tuple=False)

        if valid_idx.numel() > 0:
            crops_valid = crops[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
            chunk = roi_chunk
            outs = []
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=False):
                for s in range(0, int(crops_valid.shape[0]), chunk):
                    outs.append(self.model.vision(crops_valid[s : s + chunk]))
            vis_valid = torch.cat(outs, dim=0)
            vis_feats[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]] = vis_valid.to(
                dtype=vis_feats.dtype
            )

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=False):
            geom_feats = self.model.geom(bboxes.float(), pm)

        return {
            "vision_features": vis_feats.cpu(),
            "geometry_features": geom_feats.cpu(),
            "person_mask": pm.cpu(),
            "frame_mask": fm.cpu() if frame_mask is not None else None,
        }

    def is_cache_complete(self, dataloader: DataLoader, split: str = "train") -> bool:
        """Quick check if cache already exists for this split without loading data."""
        split_dir = self.cache_dir / split
        if not split_dir.exists():
            return False
        
        # Count existing cache files
        cached_files = list(split_dir.glob("*.pt"))
        expected_count = len(dataloader.dataset)
        
        if len(cached_files) >= expected_count:
            logger.info("Found complete cache for %s: %d samples", split, len(cached_files))
            return True
        
        logger.info("Incomplete cache for %s: %d/%d samples", split, len(cached_files), expected_count)
        return False

    def extract_and_cache(
        self, dataloader: DataLoader, split: str = "train"
    ) -> Path:
        split_dir = self.cache_dir / split
        split_dir.mkdir(exist_ok=True)
        
        # Skip extraction if cache is complete
        if self.is_cache_complete(dataloader, split):
            logger.info("Skipping feature extraction for %s (cache complete)", split)
            return split_dir

        logger.info("Extracting features for %s split...", split)
        self.model.eval()

        cached_count = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Extracting {split}")):
            batch_size = batch["frames"].shape[0]
            
            for i in range(batch_size):
                video_id = batch.get("video_id", [f"sample_{batch_idx * dataloader.batch_size + i:06d}"])[i]
                safe_filename = str(video_id).replace("/", "_").replace("\\", "_")
                cache_file = split_dir / f"{safe_filename}.pt"
                
                if cache_file.exists():
                    cached_count += 1
                    continue

                single_batch = {k: v[i:i+1] if isinstance(v, torch.Tensor) else v[i:i+1] 
                                for k, v in batch.items()}
                
                features = self.extract_features_from_batch(single_batch)
                
                # Remove batch dimension (B=1) before saving
                features["vision_features"] = features["vision_features"].squeeze(0)  # (1,T,N,D) -> (T,N,D)
                features["geometry_features"] = features["geometry_features"].squeeze(0)  # (1,T,N,D) -> (T,N,D)
                features["person_mask"] = features["person_mask"].squeeze(0)  # (1,T,N) -> (T,N)
                if features["frame_mask"] is not None:
                    features["frame_mask"] = features["frame_mask"].squeeze(0)  # (1,T) -> (T)
                
                features["target_index"] = batch["target_index"][i].cpu()
                features["video_id"] = video_id
                
                torch.save(features, cache_file)

        if cached_count > 0:
            logger.info("Reused %d cached samples for %s", cached_count, split)
        
        logger.info("Feature extraction complete for %s: %s", split, split_dir)
        return split_dir

    def cleanup(self) -> None:
        if self.cache_dir.exists():
            logger.info("Cleaning up feature cache: %s", self.cache_dir)
            shutil.rmtree(self.cache_dir)


class CachedFeatureDataset:
    def __init__(self, cache_dir: Path, split: str = "train") -> None:
        self.cache_dir = cache_dir / split
        if not self.cache_dir.exists():
            raise ValueError(f"Cache directory not found: {self.cache_dir}")
        
        self.sample_files = sorted(self.cache_dir.glob("*.pt"))
        logger.info("Loaded %d cached samples from %s", len(self.sample_files), split)

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return torch.load(self.sample_files[idx])


def should_use_cache(config: Any) -> bool:
    return bool(config.data.cache_data)
