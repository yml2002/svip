"""Trainer (DDP-aware) with logging/visualization/CSV output.

This is a simplified but compatible version of the original trainer.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.training.loss import CombinedLoss
from src.training.loops import train_epoch, validate_epoch
from src.utils.output_manager import OutputManager
from src.utils.visualization import save_metric_plots


logger = logging.getLogger(__name__)


class MemoryEfficientTrainer:
    def __init__(
        self,
        *,
        config: Any,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_function: Optional[nn.Module],
        device: torch.device,
        visualization_dir: Optional[str],
        checkpoint_dir: Optional[str],
        records_dir: Optional[str],
        predictions_dir: Optional[str],
    ) -> None:
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank)) if torch.cuda.is_available() else 0

        self.use_mixed_precision = bool(self.config.training.use_mixed_precision)
        self.accumulation_steps = int(self.config.training.accumulation_steps)
        self.max_grad_norm = float(self.config.training.max_grad_norm)
        self.debug = bool(self.config.training.debug)

        model = model.to(self.device)
        if self.is_distributed:
            ddp_kwargs: Dict[str, Any] = {
                "find_unused_parameters": bool(self.config.training.find_unused_parameters),
            }
            if self.device.type == "cuda":
                ddp_kwargs["device_ids"] = [self.local_rank]
                ddp_kwargs["output_device"] = self.local_rank
            self.model = DDP(model, **ddp_kwargs)
        else:
            self.model = model

        self.loss_function = loss_function or CombinedLoss(config=config)

        self.scaler = amp.GradScaler() if (self.use_mixed_precision and self.device.type == "cuda") else None

        self.visualization_dir = Path(visualization_dir) if visualization_dir else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.records_dir = Path(records_dir) if records_dir else None
        self.predictions_dir = Path(predictions_dir) if predictions_dir else None

        self.record_logger = (
            OutputManager(records_dir=self.records_dir, predictions_dir=self.predictions_dir)
            if (self.records_dir or self.predictions_dir)
            else None
        )

        self.current_epoch = 0
        self.global_step = 0

        self._rankk_cache = {1: 0.0, 2: 0.0, 3: 0.0}

        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _is_main_process(self) -> bool:
        return self.rank == 0

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                out[k] = v
                continue

            # Transfer both raw data and cached features to device
            if k in {"frames", "bboxes", "person_mask", "target_index", 
                     "vision_features", "geometry_features"}:
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def save_checkpoint(self, name: str = "last.pt") -> None:
        if self.checkpoint_dir is None or not self._is_main_process():
            return

        model_ref = self.model.module if hasattr(self.model, "module") else self.model

        trainable_state = {n: p.detach().cpu() for n, p in model_ref.named_parameters() if p.requires_grad}
        payload = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_trainable": trainable_state,
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            payload["scheduler"] = self.scheduler.state_dict()
        torch.save(payload, self.checkpoint_dir / name)

    def save_checkpoint_with_metrics(
        self,
        *,
        name: str,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        rank1: float,
        rank2: float,
        rank3: float,
    ) -> None:
        """Save a checkpoint plus the key epoch metrics.

        We keep this separate so both `last.pt` and `best.pt` share the same payload
        layout.
        """
        if self.checkpoint_dir is None or not self._is_main_process():
            return

        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        trainable_state = {n: p.detach().cpu() for n, p in model_ref.named_parameters() if p.requires_grad}
        payload: Dict[str, Any] = {
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            "model_trainable": trainable_state,
            "optimizer": self.optimizer.state_dict(),
            "metrics": {
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "rank@1": float(rank1),
                "rank@2": float(rank2),
                "rank@3": float(rank3),
            },
        }
        if self.scheduler is not None:
            payload["scheduler"] = self.scheduler.state_dict()
        torch.save(payload, self.checkpoint_dir / name)

    def fit(self) -> None:
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        rank1_vals = []
        rank2_vals = []
        rank3_vals = []

        train_imp_losses = []
        train_pref_losses = []
        train_pair_losses = []
        val_imp_losses = []
        val_pref_losses = []
        val_pair_losses = []

        num_epochs = int(self.config.training.num_epochs)
        early_stop_patience = getattr(self.config.training, "early_stop", None)
        early_stop_patience = int(early_stop_patience) if early_stop_patience is not None else None
        best_val_acc = float("-inf")
        best_epoch: Optional[int] = None
        epochs_no_improve = 0
        for epoch in range(1, num_epochs + 1):
            # Public-facing epoch numbers are 1-based (src-ref behavior).
            # Internally we keep current_epoch 0-based to avoid breaking checkpoint fields.
            self.current_epoch = epoch - 1

            t_epoch0 = time.perf_counter() if self._is_main_process() else 0.0
            train_loss, train_acc, train_imp_loss, train_pref_loss, train_pair_loss = train_epoch(self, log_every=10)
            val_loss, val_acc, r1, r2, r3, val_imp_loss, val_pref_loss, val_pair_loss = validate_epoch(self, log_every=10)
            t_epoch1 = time.perf_counter() if self._is_main_process() else 0.0

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            rank1_vals.append(r1)
            rank2_vals.append(r2)
            rank3_vals.append(r3)

            train_imp_losses.append(train_imp_loss)
            train_pref_losses.append(train_pref_loss)
            train_pair_losses.append(train_pair_loss)
            val_imp_losses.append(val_imp_loss)
            val_pref_losses.append(val_pref_loss)
            val_pair_losses.append(val_pair_loss)

            if self.record_logger and self._is_main_process():
                self.record_logger.log_losses(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_importance_loss": train_imp_loss,
                        "train_preference_loss": train_pref_loss,
                        "train_pairwise_loss": train_pair_loss,
                        "val_loss": val_loss,
                        "val_importance_loss": val_imp_loss,
                        "val_preference_loss": val_pref_loss,
                        "val_pairwise_loss": val_pair_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "rank@1": r1,
                        "rank@2": r2,
                        "rank@3": r3,
                    }
                )

                # Bucketed rank@k by number of valid persons (K) if available.
                # Expected format: {"k2": {"rank1":...,"rank2":...,"rank3":...,"n":...}, ...}
                rank_by_k = getattr(self, "_rank_by_k", None)
                if isinstance(rank_by_k, dict) and rank_by_k:
                    bucket_payload: Dict[str, float] = {}
                    for bk, stats in rank_by_k.items():
                        if not isinstance(stats, dict):
                            continue
                        n = int(stats.get("n", 0) or 0)
                        bucket_payload[f"rank_by_k-{bk}-n"] = float(n)
                        bucket_payload[f"rank_by_k-{bk}-rank1"] = float(stats.get("rank1", 0.0) or 0.0)
                        bucket_payload[f"rank_by_k-{bk}-rank2"] = float(stats.get("rank2", 0.0) or 0.0)
                        bucket_payload[f"rank_by_k-{bk}-rank3"] = float(stats.get("rank3", 0.0) or 0.0)
                    if bucket_payload:
                        self.record_logger.log_metrics({"epoch": epoch, **bucket_payload})


            if self._is_main_process():
                save_metric_plots(
                    self.visualization_dir,
                    int(epoch),
                    train_losses=train_losses,
                    val_losses=val_losses,
                    train_accs=train_accs,
                    val_accs=val_accs,
                    rank1_values=rank1_vals,
                    rank2_values=rank2_vals,
                    rank3_values=rank3_vals,
                    main_process=True,
                    logger=logger,
                    plot_validation=True,
                )

                # Always save latest.
                if self.config.training.save_model:
                    self.save_checkpoint_with_metrics(
                        name="last.pt",
                        train_loss=float(train_loss),
                        train_acc=float(train_acc),
                        val_loss=float(val_loss),
                        val_acc=float(val_acc),
                        rank1=float(r1),
                        rank2=float(r2),
                        rank3=float(r3),
                    )

                # Scheduler step at epoch end (recommended order: after optimizer steps).
                if self.scheduler is not None:
                    self.scheduler.step()

                # Epoch summary (keep it compact like src-ref)
                group_lrs = [float(pg.get("lr", 0.0)) for pg in self.optimizer.param_groups]
                group_lr_str = ", ".join(f"g{i}:{lr:.6f}" for i, lr in enumerate(group_lrs))
                epoch_dt = (t_epoch1 - t_epoch0) if t_epoch0 else float("nan")

                logger.info(
                    "Epoch %d: train_acc=%.4f train_loss=%.4f (imp=%.4f pref=%.4f) "
                    "val_acc=%.4f val_loss=%.4f (imp=%.4f pref=%.4f) "
                    "rank1=%.2f%% rank2=%.2f%% rank3=%.2f%% lr=%.6f group_lrs=[%s] dt=%.1fs",
                    int(epoch),
                    float(train_acc),
                    float(train_loss),
                    float(train_imp_loss),
                    float(train_pref_loss),
                    float(val_acc),
                    float(val_loss),
                    float(val_imp_loss),
                    float(val_pref_loss),
                    float(r1),
                    float(r2),
                    float(r3),
                    float(self.config.training.learning_rate),
                    group_lr_str,
                    float(epoch_dt),
                )

                rank_by_k = getattr(self, "_rank_by_k", None)
                if isinstance(rank_by_k, dict) and rank_by_k:
                    # Keep it single-line and stable for grepping.
                    parts = []
                    for bk in ("k2", "k3", "k4", "k5p"):
                        s = rank_by_k.get(bk)
                        if not isinstance(s, dict):
                            continue
                        parts.append(
                            f"{bk} n={int(s.get('n', 0) or 0)} "
                            f"r1={float(s.get('rank1', 0.0) or 0.0):.2f}% "
                            f"r2={float(s.get('rank2', 0.0) or 0.0):.2f}% "
                            f"r3={float(s.get('rank3', 0.0) or 0.0):.2f}%"
                        )
                    if parts:
                        logger.info("Epoch %d: val-rank_by_k: %s", int(epoch), " | ".join(parts))

                # Save best (by val_acc) after summary.
                if float(val_acc) > best_val_acc:
                    best_val_acc = float(val_acc)
                    best_epoch = int(epoch)
                    if self.config.training.save_model:
                        self.save_checkpoint_with_metrics(
                            name="best.pt",
                            train_loss=float(train_loss),
                            train_acc=float(train_acc),
                            val_loss=float(val_loss),
                            val_acc=float(val_acc),
                            rank1=float(r1),
                            rank2=float(r2),
                            rank3=float(r3),
                        )
                    logger.info("Updated best checkpoint: best.pt (epoch=%d best_val_acc=%.4f)", int(epoch), float(val_acc))

                # Early stopping decision happens after validation, records, plots,
                # checkpoint, and scheduler step have all completed for this epoch.
                should_stop = False
                if early_stop_patience is not None:
                    if float(val_acc) >= best_val_acc:
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    logger.info(
                        "EarlyStop: patience=%d best_val_acc=%.4f epochs_no_improve=%d",
                        int(early_stop_patience),
                        float(best_val_acc),
                        int(epochs_no_improve),
                    )

                    # Stop once we've reached or exceeded patience epochs without improvement.
                    if epochs_no_improve >= early_stop_patience:
                        logger.info(
                            "EarlyStop triggered at epoch %d (no improvement for %d epochs)",
                            int(epoch),
                            int(epochs_no_improve),
                        )
                        should_stop = True

            else:
                should_stop = False

            # Keep all ranks in sync: if rank0 early-stops, others should exit too.
            if self.is_distributed:
                stop_flag = torch.tensor(
                    [1 if (early_stop_patience is not None and should_stop) else 0],
                    device=self.device,
                )
                dist.broadcast(stop_flag, src=0)
                if int(stop_flag.item()) == 1:
                    break
            if (not self.is_distributed) and early_stop_patience is not None and should_stop:
                break

        # Training summary
        if self._is_main_process():
            epoch_val_acc_str = ", ".join(f"{i+1}:{acc:.4f}" for i, acc in enumerate(val_accs))
            logger.info("val_acc per epoch: %s", epoch_val_acc_str)
            if best_epoch is not None:
                logger.info("best_val_acc: %.4f (epoch %d)", best_val_acc, best_epoch)


