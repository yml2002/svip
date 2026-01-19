"""Training runtime orchestrator.

Responsibilities:
- setup distributed flags and output directories
- load config and apply CLI overrides
- build dataloaders, model, optimizer, trainer

We keep the external behavior similar to src-ref but in a leaner codebase.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.parallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.configs.config import get_default_config
from src.data.dataloader import MSGVIPDataset
from src.models.importance_ranker import ImportanceRanker
from src.training.loss import CombinedLoss
from src.training.trainer import MemoryEfficientTrainer
from src.utils.feature_cache import FeatureExtractor, CachedFeatureDataset, should_use_cache

logger = logging.getLogger(__name__)


class TrainingRuntime:
    def __init__(self, args, project_root: Path, output_creator, cuda_allocator) -> None:
        self.args = args
        self.project_root = project_root
        self._output_creator = output_creator
        self._cuda_allocator = cuda_allocator

        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = torch.device("cpu")
        self.is_main_process = True

        self.output_paths: Dict[str, Path] = {}
        self.config = None
        self.trainer: Optional[MemoryEfficientTrainer] = None
        self.feature_extractor: Optional[FeatureExtractor] = None

        self.logger = logging.getLogger("msg_vip_training")

    def run(self) -> int:
        try:
            self._prepare_environment()
            self._load_and_override_config()
            self._build_trainer()

            if getattr(self.args, "validate_only", False):
                # one pass validate
                self.trainer.current_epoch = 0
                from src.training.loops import validate_epoch

                validate_epoch(self.trainer)
                return 0

            self.trainer.fit()
            
            # Keep cache for reuse across multiple training runs
            # if self.feature_extractor is not None and self.is_main_process:
            #     self.feature_extractor.cleanup()
            
            return 0
        except Exception as exc:
            self.logger.error("Training runtime failed: %s", exc, exc_info=True)
            # Keep cache even on failure for debugging and reuse
            # if self.feature_extractor is not None and self.is_main_process:
            #     try:
            #         self.feature_extractor.cleanup()
            #     except Exception as e:
            #         self.logger.warning("Failed to cleanup feature cache: %s", e)
            return 1

    def _prepare_environment(self) -> None:
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if torch.cuda.is_available() else 0
        self.is_main_process = self.rank == 0

        if torch.cuda.is_available():
            self._cuda_allocator()
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.local_rank)
        else:
            self.device = torch.device("cpu")

        output_map: Optional[Dict[str, str]] = None
        if self.is_main_process:
            created = self._output_creator(self.args.output_dir)
            output_map = {k: str(v) for k, v in created.items()}

        if self.is_distributed:
            payload = [output_map]
            dist.broadcast_object_list(payload, src=0)
            output_map = payload[0]

        if not output_map:
            raise RuntimeError("Failed to initialize output directories")

        self.output_paths = {k: Path(v) for k, v in output_map.items()}
        if not self.is_main_process:
            for name, path in self.output_paths.items():
                if name != "checkpoints":
                    path.mkdir(parents=True, exist_ok=True)

        log_level = "DEBUG" if getattr(self.args, "debug", False) else "INFO"
        numeric_level = getattr(logging, log_level, logging.INFO)
        self.logger.propagate = False
        self.logger.setLevel(numeric_level)

        # Avoid duplicated handlers if runtime is constructed multiple times (e.g. tests).
        if getattr(self.logger, "_msgvip_configured", False):
            return

        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        if self.is_main_process:
            ch = logging.StreamHandler()
            ch.setLevel(numeric_level)
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

            fh = logging.FileHandler(self.output_paths["logs"] / "training.log")
            fh.setLevel(numeric_level)
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

        rh = logging.FileHandler(self.output_paths["logs"] / f"training_rank{self.rank}.log")
        rh.setLevel(numeric_level)
        rh.setFormatter(fmt)
        self.logger.addHandler(rh)
        root = logging.getLogger()
        if not getattr(root, "_msgvip_root_configured", False):
            root.setLevel(numeric_level)
            for h in self.logger.handlers:
                root.addHandler(h)
            setattr(root, "_msgvip_root_configured", True)

        setattr(self.logger, "_msgvip_configured", True)

        if self.is_main_process:
            meta = {
                "timestamp": str(self.output_paths["run_dir"].name),
                "world_size": int(self.world_size),
                "args": {k: v for k, v in vars(self.args).items() if v is not None and not k.startswith("_")},
            }
            with (self.output_paths["records"] / "run_meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        self.logger.info("MSG_VIP training runtime initialized")
        self.logger.info("Device: %s", self.device)
        self.logger.info("Distributed: %s rank=%d/%d", self.is_distributed, self.rank, self.world_size)
        self.logger.info("Output directory: %s", self.output_paths["run_dir"])

    def _load_and_override_config(self) -> None:
        config = get_default_config()

        # CLI overrides
        config.training.debug = bool(getattr(self.args, "debug", False))
        if config.training.debug:
            config.training.debug_output_dir = str(self.output_paths["records"] / "debug")

        data_dir = Path(config.data.data_dir)
        if not data_dir.is_absolute():
            data_dir = (self.project_root / data_dir).resolve()
        if getattr(self.args, "data_dir", None) is not None:
            data_dir = Path(self.args.data_dir)
        config.data.data_dir = str(data_dir)

        if getattr(self.args, "batch_size", None) is not None:
            config.training.batch_size = int(self.args.batch_size)
        if getattr(self.args, "accumulation_steps", None) is not None:
            config.training.accumulation_steps = int(self.args.accumulation_steps)
        if getattr(self.args, "learning_rate", None) is not None:
            config.training.learning_rate = float(self.args.learning_rate)
        if getattr(self.args, "num_epochs", None) is not None:
            config.training.num_epochs = int(self.args.num_epochs)
        if getattr(self.args, "num_workers", None) is not None:
            config.training.num_workers = int(self.args.num_workers)
        if getattr(self.args, "early_stop", None) is not None:
            config.training.early_stop = int(self.args.early_stop)
        if getattr(self.args, "data_ratio", None) is not None:
            ratio = float(self.args.data_ratio)
            if ratio <= 0.0 or ratio > 1.0:
                raise ValueError(f"data_ratio must be in (0,1], got {ratio}")
            config.data.data_ratio = ratio
            config.data.max_samples = None
            
        if getattr(self.args, "importance_weight", None) is not None:
            config.model.loss.importance_weight = float(self.args.importance_weight)
        if getattr(self.args, "preference_weight", None) is not None:
            config.model.loss.preference_weight = float(self.args.preference_weight)
        if getattr(self.args, "pairwise_weight", None) is not None:
            config.model.loss.pairwise_weight = float(self.args.pairwise_weight)
        if getattr(self.args, "pairwise_margin", None) is not None:
            config.model.loss.pairwise_margin = float(self.args.pairwise_margin)

        config.training.distributed = self.is_distributed
        config.training.world_size = self.world_size
        config.training.local_rank = self.local_rank

        self.config = config

        if self.is_main_process:
            with (self.output_paths["configs"] / "run_config.json").open("w", encoding="utf-8") as f:
                json.dump(_to_jsonable(config), f, ensure_ascii=False, indent=2)

    def _build_trainer(self) -> None:
        assert self.config is not None

        ratio = float(getattr(self.config.data, "data_ratio", 1.0) or 1.0)

        def _ratio_to_max_samples(split: str) -> int | None:
            if ratio >= 1.0:
                return int(getattr(self.config.data, "max_samples", None)) if getattr(self.config.data, "max_samples", None) is not None else None

            split_dir = Path(self.config.data.data_dir) / split
            npz_files = list(split_dir.glob("*.npz"))
            total = len(npz_files)
            if total <= 0:
                return None
            keep = max(1, int(math.ceil(total * ratio)))
            return keep

        train_max = _ratio_to_max_samples("train")
        val_max = _ratio_to_max_samples("val")

        # 把旧 val 的多少比例换进 train，同时从 train 换出同样数量到 val
        swap_splits = bool(getattr(self.args, "swap_splits", False))
        swap_fraction = float(getattr(self.args, "swap_fraction", 0.5))

        if not swap_splits or swap_fraction <= 0.0:
            train_ds = MSGVIPDataset(config=self.config, data_path=self.config.data.data_dir, split="train", max_samples=train_max)
            val_ds = MSGVIPDataset(config=self.config, data_path=self.config.data.data_dir, split="val", max_samples=val_max)
        else:
            data_root = Path(self.config.data.data_dir)
            train_files = sorted((data_root / "train").glob("*.npz"))
            val_files = sorted((data_root / "val").glob("*.npz"))

            if not train_files:
                raise ValueError(f"No NPZ files found in {data_root / 'train'}")
            if not val_files:
                raise ValueError(f"No NPZ files found in {data_root / 'val'}")

            train_files = train_files[: int(train_max)] if train_max is not None else train_files
            val_files = val_files[: int(val_max)] if val_max is not None else val_files

            import numpy as np
            from collections import defaultdict

            def _bucket_key(npz_path: Path) -> tuple[str, int]:
                d = np.load(npz_path, allow_pickle=True)
                sc = d["scene_category"].item() if hasattr(d["scene_category"], "item") else str(d["scene_category"])
                pm = d["person_mask"].astype(bool)
                n_people = int(pm.any(axis=0).sum())
                return (str(sc), n_people)

            buckets: dict[tuple[str, int], list[Path]] = defaultdict(list)
            for p in train_files:
                buckets[_bucket_key(p)].append(p)
            for p in val_files:
                buckets[_bucket_key(p)].append(p)

            target_train = len(train_files)
            target_val = len(val_files)

            old_train_set = set(train_files)
            old_val_set = set(val_files)

            new_train: list[Path] = []
            new_val: list[Path] = []

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

            def _pad_or_trim(primary: list[Path], secondary: list[Path], target: int) -> list[Path]:
                if len(primary) >= target:
                    return primary[:target]
                need = target - len(primary)
                return primary + secondary[:need]

            new_train = _pad_or_trim(new_train, new_val, target_train)
            new_val = _pad_or_trim(new_val, new_train, target_val)

            class _FileListDataset(MSGVIPDataset):
                def __init__(self, file_list, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.file_list = [str(p) for p in file_list]

            train_ds = _FileListDataset(new_train, config=self.config, data_path=self.config.data.data_dir, split="train", max_samples=None)
            val_ds = _FileListDataset(new_val, config=self.config, data_path=self.config.data.data_dir, split="val", max_samples=None)

        self.logger.info(
            "Dataset sizing: data_ratio=%.3f train=%d val=%d",
            ratio,
            len(train_ds),
            len(val_ds),
        )

        train_sampler = DistributedSampler(train_ds, shuffle=True) if self.is_distributed else None
        val_sampler = DistributedSampler(val_ds, shuffle=False) if self.is_distributed else None

        train_loader = DataLoader(
            train_ds,
            batch_size=int(self.config.training.batch_size),
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=int(self.config.training.num_workers),
            pin_memory=bool(self.config.training.pin_memory),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(self.config.training.batch_size),
            shuffle=False,
            sampler=val_sampler,
            num_workers=int(self.config.training.num_workers),
            pin_memory=bool(self.config.training.pin_memory),
        )

        model = ImportanceRanker(self.config)
        model = model.to(self.device)
        
        use_cache = should_use_cache(self.config)
        
        if use_cache and self.is_main_process:
            self.logger.info("Feature caching enabled - extracting features first...")
            
            self.feature_extractor = FeatureExtractor(
                model=model,
                config=self.config,
                device=self.device,
            )
            
            train_cache_dir = self.feature_extractor.extract_and_cache(train_loader, split="train")
            val_cache_dir = self.feature_extractor.extract_and_cache(val_loader, split="val")
            
            self.logger.info("Feature extraction complete!")
            
            if self.is_distributed:
                cache_info = [str(self.feature_extractor.cache_dir)]
                dist.broadcast_object_list(cache_info, src=0)
        elif use_cache and self.is_distributed:
            cache_info = [None]
            dist.broadcast_object_list(cache_info, src=0)
            cache_dir = Path(cache_info[0])
            
            self.feature_extractor = FeatureExtractor(
                model=model,
                config=self.config,
                device=self.device,
            )
        
        if use_cache:
            cache_dir = self.feature_extractor.cache_dir
            train_cached_ds = CachedFeatureDataset(cache_dir, split="train")
            val_cached_ds = CachedFeatureDataset(cache_dir, split="val")
            
            train_sampler = DistributedSampler(train_cached_ds, shuffle=True) if self.is_distributed else None
            val_sampler = DistributedSampler(val_cached_ds, shuffle=False) if self.is_distributed else None
            
            train_loader = DataLoader(
                train_cached_ds,
                batch_size=int(self.config.training.batch_size),
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=0,
                pin_memory=bool(self.config.training.pin_memory),
            )
            val_loader = DataLoader(
                val_cached_ds,
                batch_size=int(self.config.training.batch_size),
                shuffle=False,
                sampler=val_sampler,
                num_workers=0,
                pin_memory=bool(self.config.training.pin_memory),
            )
            
            self.logger.info("Using cached features for training")

        if dist.is_initialized():
            find_unused = bool(self.config.training.find_unused_parameters)
            if self.device.type == "cuda":
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=find_unused)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=find_unused)

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params,
            lr=float(self.config.training.learning_rate),
            weight_decay=float(self.config.training.weight_decay),
            betas=self.config.training.betas,
        )

        sched = CosineAnnealingLR(opt, T_max=max(1, int(self.config.training.num_epochs)), eta_min=float(self.config.training.min_lr))

        loss_fn = CombinedLoss(config=self.config)

        self.trainer = MemoryEfficientTrainer(
            config=self.config,
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=opt,
            scheduler=sched,
            loss_function=loss_fn,
            device=self.device,
            visualization_dir=str(self.output_paths["visualizations"]),
            checkpoint_dir=str(self.output_paths["checkpoints"]),
            records_dir=str(self.output_paths["records"]),
            predictions_dir=str(self.output_paths["predictions"]),
        )


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dict__") and not isinstance(obj, dict):
        out = {}
        for k, v in obj.__dict__.items():
            out[k] = _to_jsonable(v)
        return out
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
