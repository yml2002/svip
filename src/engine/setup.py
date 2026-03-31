"""Training setup: assemble distributed env, config, dataloaders, model, optimizer, and trainer."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.parallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.config import get_default_config
from src.data.splits import build_train_val_datasets
from src.models.ranker import PersonRanker
from src.engine.loss import CombinedLoss
from src.engine.trainer import Trainer
from src.utils.io import to_jsonable

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
        self.trainer: Optional[Trainer] = None

        self.logger = logging.getLogger("training")

    def run(self) -> int:
        try:
            self._prepare_environment()
            self._load_and_override_config()
            self._apply_performance_tuning()
            self._build_trainer()

            if getattr(self.args, "validate_only", False):
                self.trainer.current_epoch = 0
                from src.engine.loops import validate_epoch
                validate_epoch(self.trainer)
                return 0

            self.trainer.fit()
            return 0
        except Exception as exc:
            self.logger.error("Training runtime failed: %s", exc, exc_info=True)
            return 1

    def _apply_performance_tuning(self) -> None:
        if self.device.type != "cuda":
            return
        torch.backends.cudnn.benchmark = True
        allow_tf32 = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        self.logger.info(
            "Perf tuning: cudnn_benchmark=%s allow_tf32=%s",
            bool(torch.backends.cudnn.benchmark),
            bool(allow_tf32),
        )

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

        if getattr(self.logger, "_logger_configured", False):
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
        if not getattr(root, "_root_logger_configured", False):
            root.setLevel(numeric_level)
            for h in self.logger.handlers:
                root.addHandler(h)
            setattr(root, "_root_logger_configured", True)

        setattr(self.logger, "_logger_configured", True)

        if self.is_main_process:
            meta = {
                "timestamp": str(self.output_paths["run_dir"].name),
                "world_size": int(self.world_size),
                "args": {k: v for k, v in vars(self.args).items() if v is not None and not k.startswith("_")},
            }
            with (self.output_paths["records"] / "run_meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        self.logger.info("Training runtime initialized")
        self.logger.info("Device: %s", self.device)
        self.logger.info("Distributed: %s rank=%d/%d", self.is_distributed, self.rank, self.world_size)
        self.logger.info("Output directory: %s", self.output_paths["run_dir"])

    def _load_and_override_config(self) -> None:
        config = get_default_config()

        if getattr(self.args, "seed", None) is not None:
            config.training.seed = int(self.args.seed)
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
        if getattr(self.args, "roi_chunk", None) is not None:
            roi_chunk = int(self.args.roi_chunk)
            if roi_chunk <= 0:
                raise ValueError(f"roi_chunk must be > 0, got {roi_chunk}")
            config.training.roi_chunk = roi_chunk
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
        if getattr(self.args, "relation_delta_scale", None) is not None:
            config.model.relation.delta_scale = float(self.args.relation_delta_scale)
        if bool(getattr(self.args, "no_gat", False)):
            config.model.gatv2.enabled = False
        if bool(getattr(self.args, "no_spatial_edges", False)):
            config.model.gatv2.use_spatial_edges = False
        if bool(getattr(self.args, "no_temporal_edges", False)):
            config.model.gatv2.use_temporal_edges = False
        if bool(getattr(self.args, "no_edge_features", False)):
            config.model.gatv2.use_edge_features = False
        if bool(getattr(self.args, "no_geom", False)):
            config.model.features.bbox_geom.enabled = False
        if bool(getattr(self.args, "mean_only_unary", False)):
            config.model.intrinsic.use_max_pool = False
            config.model.intrinsic.use_attention_pool = False
        if getattr(self.args, "gat_topk_neighbors", None) is not None:
            topk = int(self.args.gat_topk_neighbors)
            if topk < 0:
                raise ValueError(f"gat_topk_neighbors must be >= 0, got {topk}")
            config.model.gatv2.topk_neighbors = topk
        if getattr(self.args, "relation_enabled", None) is not None:
            config.model.relation.enabled = bool(int(self.args.relation_enabled))
        if getattr(self.args, "unfreeze_layers", None) is not None:
            config.model.features.dino.unfreeze_layers = int(self.args.unfreeze_layers)
        if getattr(self.args, "backbone_lr_scale", None) is not None:
            config.training.backbone_lr_scale = float(self.args.backbone_lr_scale)
        if getattr(self.args, "backbone_warmup_epochs", None) is not None:
            config.training.backbone_warmup_epochs = int(self.args.backbone_warmup_epochs)
        if getattr(self.args, "backbone_train_mode", None) is not None:
            config.training.backbone_train_mode = str(self.args.backbone_train_mode)

        # Open-world scene context
        if bool(getattr(self.args, "no_global_context", False)):
            config.model.global_context.enabled = False
        if getattr(self.args, "global_num_keyframes", None) is not None:
            config.model.global_context.num_keyframes = int(self.args.global_num_keyframes)
        if getattr(self.args, "global_num_prototypes", None) is not None:
            config.model.global_context.num_prototypes = int(self.args.global_num_prototypes)

        config.training.distributed = self.is_distributed
        config.training.world_size = self.world_size
        config.training.local_rank = self.local_rank

        self.config = config

        if self.is_main_process:
            with (self.output_paths["configs"] / "run_config.json").open("w", encoding="utf-8") as f:
                json.dump(to_jsonable(config), f, ensure_ascii=False, indent=2)

    def _build_trainer(self) -> None:
        assert self.config is not None

        train_ds, val_ds = build_train_val_datasets(self.config)

        train_sampler = DistributedSampler(train_ds, shuffle=True) if self.is_distributed else None
        val_sampler = DistributedSampler(val_ds, shuffle=False) if self.is_distributed else None

        num_workers = int(self.config.training.num_workers)
        prefetch_factor = 2 if num_workers > 0 else None

        train_loader = DataLoader(
            train_ds,
            batch_size=int(self.config.training.batch_size),
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=bool(self.config.training.pin_memory),
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor,
            drop_last=self.is_distributed,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(self.config.training.batch_size),
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=bool(self.config.training.pin_memory),
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor,
            drop_last=False,
        )

        model = PersonRanker(self.config).to(self.device)

        # Differential learning rate: backbone vs. head
        base_lr = float(self.config.training.learning_rate)
        backbone_lr_scale = float(getattr(self.config.training, "backbone_lr_scale", 1.0))
        backbone_lr = base_lr * backbone_lr_scale

        backbone_params = []
        head_params = []
        for name, p in model.named_parameters():
            (backbone_params if "vision.backbone" in name else head_params).append(p)

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr, "group_name": "backbone"})
        if head_params:
            param_groups.append({"params": head_params, "lr": base_lr, "group_name": "head"})

        if backbone_lr_scale != 1.0:
            self.logger.info(
                "Differential LR: backbone=%.2e (scale=%.2f) head=%.2e",
                backbone_lr, backbone_lr_scale, base_lr,
            )

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=float(self.config.training.weight_decay),
            betas=self.config.training.betas,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(self.config.training.num_epochs)),
            eta_min=float(self.config.training.min_lr),
        )

        self.trainer = Trainer(
            config=self.config,
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=CombinedLoss(config=self.config),
            device=self.device,
            visualization_dir=str(self.output_paths["visualizations"]),
            checkpoint_dir=str(self.output_paths["checkpoints"]),
            records_dir=str(self.output_paths["records"]),
            predictions_dir=str(self.output_paths["predictions"]),
        )
