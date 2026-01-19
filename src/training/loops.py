"""Training loop helpers (progress bar + AMP + metrics accumulation)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import csv
import torch.distributed as dist
from torch import amp
from tqdm import tqdm

def compute_accuracy_metrics(
    importance_logits: torch.Tensor,
    target_index: torch.Tensor,
    person_mask: torch.Tensor,
) -> Dict[str, int]:
    valid_mask = person_mask.any(dim=1) if person_mask.dim() == 3 else person_mask.bool()
    masked_logits = importance_logits.masked_fill(~valid_mask, float("-inf"))
    pred = masked_logits.argmax(dim=1)
    correct = int((pred == target_index).sum().item())
    total = int(target_index.numel())
    return {"correct_predictions": correct, "total_samples": total}


@torch.no_grad()
def compute_topk_indices(
    importance_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Compute masked top-k indices on the same device as `importance_logits`.

    `valid_mask` is candidate mask (B,N) or (B,T,N) -> reduced to (B,N).
    Returns LongTensor of shape (B,k_eff).
    """

    mask = valid_mask.bool()
    if mask.dim() > 2:
        mask = mask.any(dim=1)
    masked_logits = importance_logits.masked_fill(~mask, float("-inf"))
    k_eff = min(int(k), int(masked_logits.shape[1]))
    return masked_logits.topk(k=k_eff, dim=1).indices


@torch.no_grad()
def compute_rankk_from_topk(topk_indices: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Compute rank@k from precomputed top-k indices.

    `topk_indices`: (B, K>=k)
    `targets`: (B,)
    Returns ratio in [0,1].
    """

    if topk_indices.numel() == 0:
        return 0.0
    tgt = targets.long()
    hit = (topk_indices[:, : int(k)] == tgt.unsqueeze(1)).any(dim=1).float()
    return float(hit.mean().item())

@torch.no_grad()
def compute_rankk_global(
    importance_logits: torch.Tensor,
    target_index: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int,
) -> float:
    targets = target_index.long()
    mask = valid_mask.bool()
    if mask.dim() > 2:
        mask = mask.any(dim=1)

    masked_logits = importance_logits.masked_fill(~mask, float("-inf"))
    k_eff = min(int(k), int(masked_logits.shape[1]))
    topk = masked_logits.topk(k=k_eff, dim=1).indices
    hit = (topk == targets.unsqueeze(1)).any(dim=1).float()
    return float(hit.mean().item() * 100.0)

logger = logging.getLogger(__name__)

def export_predictions_csv(
    *,
    trainer,
    split: str,
    epoch_1based: int,
    predicted_index: torch.Tensor,
    targets: torch.Tensor,
    video_ids: List[str],
    scene_categories: List[str],
    person_ids: List[List[int]],
) -> None:
    if trainer.record_logger is None or not trainer._is_main_process():
        return

    trainer.record_logger.export_predictions_csv(
        split=str(split),
        epoch=int(epoch_1based),
        predicted_index=predicted_index,
        targets=targets,
        video_ids=video_ids,
        scene_categories=scene_categories,
        person_ids=person_ids,
        rankk_cache=getattr(trainer, "_rankk_cache", {}) or {},
    )


def train_epoch(trainer, log_every: int = 10) -> Tuple[float, float, float, float, float]:
    trainer.model.train()

    epoch_loss = 0.0
    epoch_imp_loss = 0.0
    epoch_pref_loss = 0.0
    epoch_pair_loss = 0.0
    num_batches = 0
    running_correct = 0
    running_total = 0

    collect_predictions = trainer._is_main_process() and trainer.record_logger is not None
    if collect_predictions:
        pred_top3: List[torch.Tensor] = []
        pred_targets: List[torch.Tensor] = []
        pred_video_ids: List[str] = []
        pred_scene_categories: List[str] = []
        pred_person_ids: List[List[int]] = []

    is_main = trainer._is_main_process()
    iterator = trainer.train_dataloader
    if is_main:
        iterator = tqdm(iterator, desc=f"Train Epoch {trainer.current_epoch + 1}", leave=True, dynamic_ncols=True)

    accum_counter = 0
    for batch_idx, batch in enumerate(iterator):
        t0 = time.perf_counter()
        
        # Debug timing: track each step
        debug_timing = {} if trainer.debug else None
        
        batch = trainer._move_batch_to_device(batch)
        if debug_timing is not None:
            debug_timing['data_loading'] = time.perf_counter() - t0

        t_forward = time.perf_counter()
        with amp.autocast(device_type=trainer.device.type, enabled=trainer.use_mixed_precision):
            # Support both raw data and cached features mode
            if "vision_features" in batch and "geometry_features" in batch:
                outputs = trainer.model(
                    vision_features=batch["vision_features"],
                    geometry_features=batch["geometry_features"],
                    person_mask=batch["person_mask"],
                    target_index=batch["target_index"],
                    debug_epoch=trainer.current_epoch,
                    debug_batch_idx=batch_idx,
                )
            else:
                outputs = trainer.model(
                    frames=batch["frames"],
                    bboxes=batch["bboxes"],
                    person_mask=batch["person_mask"],
                    target_index=batch["target_index"],
                    debug_epoch=trainer.current_epoch,
                    debug_batch_idx=batch_idx,
                )
            
            if debug_timing is not None:
                debug_timing['forward'] = time.perf_counter() - t_forward
            
            t_loss = time.perf_counter()
            loss_components = trainer.loss_function.get_loss_components(
                importance_logits=outputs["importance_logits"],
                target_index=batch["target_index"],
                person_mask=batch["person_mask"],
                additional_outputs=outputs,
            )
            loss = loss_components["total_loss"] / trainer.accumulation_steps
            
            if debug_timing is not None:
                debug_timing['loss_compute'] = time.perf_counter() - t_loss

        if torch.isnan(loss):
            trainer.optimizer.zero_grad(set_to_none=True)
            continue

        t_backward = time.perf_counter()
        if trainer.scaler is not None:
            trainer.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if debug_timing is not None:
            debug_timing['backward'] = time.perf_counter() - t_backward

        accum_counter += 1
        if accum_counter % trainer.accumulation_steps == 0:
            t_optim = time.perf_counter()
            if trainer.scaler is not None:
                trainer.scaler.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.max_grad_norm)

            if trainer.scaler is not None:
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
            else:
                trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)

            # Global step is defined as optimizer steps (not micro-batches).
            trainer.global_step += 1
            
            if debug_timing is not None:
                debug_timing['optimizer'] = time.perf_counter() - t_optim

        loss_value = float(loss_components["total_loss"].item())
        epoch_loss += loss_value
        epoch_imp_loss += float(loss_components["importance_loss"].item())
        epoch_pref_loss += float(loss_components["preference_loss"].item())
        epoch_pair_loss += float(loss_components.get("pairwise_loss", 0.0).item())
        num_batches += 1

        acc_metrics = compute_accuracy_metrics(outputs["importance_logits"], batch["target_index"], batch["person_mask"])
        running_correct += acc_metrics["correct_predictions"]
        running_total += acc_metrics["total_samples"]

        batch_acc = acc_metrics["correct_predictions"] / max(1, acc_metrics["total_samples"])
        cum_acc = running_correct / max(1, running_total)

        if collect_predictions:
            pm = batch["person_mask"]
            valid_mask = pm.any(dim=1) if pm.dim() == 3 else pm
            top3 = compute_topk_indices(outputs["importance_logits"], valid_mask, k=3)
            pred_top3.append(top3.detach().long().cpu())
            pred_targets.append(batch["target_index"].detach().long().cpu())
            B = int(batch["target_index"].shape[0])
            videos = batch.get("video_id")
            if isinstance(videos, (list, tuple)):
                pred_video_ids.extend(str(v) for v in videos)
            else:
                pred_video_ids.extend([""] * B)

            scenes = batch.get("scene_category")
            if isinstance(scenes, (list, tuple)):
                pred_scene_categories.extend(str(s).replace(" ", "_").lower() for s in scenes)
            else:
                pred_scene_categories.extend([""] * B)

            original_ids = batch.get("original_ids")
            if isinstance(original_ids, torch.Tensor):
                for row in original_ids.detach():
                    pred_person_ids.append([int(x) for x in row.reshape(-1).tolist()])
            elif isinstance(original_ids, (list, tuple)):
                for row in original_ids:
                    if isinstance(row, torch.Tensor):
                        pred_person_ids.append([int(x) for x in row.detach().reshape(-1).tolist()])
                    elif isinstance(row, (list, tuple)):
                        pred_person_ids.append([int(x) for x in row])
                    else:
                        pred_person_ids.append([])
            else:
                for _ in range(B):
                    pred_person_ids.append([])

        if is_main:
            postfix = {"loss": f"{loss_value:.4f}", "acc": f"{cum_acc:.4f}"}
            
            # Debug timing: log detailed breakdown every N batches
            if debug_timing is not None:
                log_interval = getattr(trainer.config.training, 'debug_log_interval', 1)
                if batch_idx % log_interval == 0:
                    total_time = time.perf_counter() - t0
                    logger = logging.getLogger(__name__)
                    logger.info(
                        "Batch %4d/%d: data=%5.1fms fwd=%6.1fms loss=%5.1fms bwd=%6.1fms opt=%5.1fms total=%6.1fms | loss=%.4f acc=%.4f",
                        batch_idx + 1, len(trainer.train_dataloader),
                        debug_timing.get('data_loading', 0) * 1000,
                        debug_timing.get('forward', 0) * 1000,
                        debug_timing.get('loss_compute', 0) * 1000,
                        debug_timing.get('backward', 0) * 1000,
                        debug_timing.get('optimizer', 0) * 1000,
                        total_time * 1000,
                        loss_value, cum_acc,
                    )
            
            iterator.set_postfix(postfix)

    avg_loss = epoch_loss / max(1, num_batches)
    avg_acc = running_correct / max(1, running_total)

    if collect_predictions and pred_top3:
        top3_all = torch.cat(pred_top3, dim=0)
        targets_all = torch.cat(pred_targets, dim=0)
        # Compute rank@k from indices (ratio in [0,1]).
        trainer._rankk_cache = {
            1: compute_rankk_from_topk(top3_all, targets_all, 1),
            2: compute_rankk_from_topk(top3_all, targets_all, 2),
            3: compute_rankk_from_topk(top3_all, targets_all, 3),
        }

        export_predictions_csv(
            trainer=trainer,
            split="train",
            epoch_1based=int(trainer.current_epoch) + 1,
            targets=targets_all,
            predicted_index=top3_all[:, :1].squeeze(1),
            video_ids=pred_video_ids,
            scene_categories=pred_scene_categories,
            person_ids=pred_person_ids,
        )

    avg_imp_loss = epoch_imp_loss / max(1, num_batches)
    avg_pref_loss = epoch_pref_loss / max(1, num_batches)
    avg_pair_loss = epoch_pair_loss / max(1, num_batches)
    return avg_loss, avg_acc, avg_imp_loss, avg_pref_loss, avg_pair_loss


@torch.no_grad()
def validate_epoch(trainer, log_every: int = 10) -> Tuple[float, float, float, float, float, float, float, float]:
    trainer.model.eval()

    epoch_loss = 0.0
    epoch_imp_loss = 0.0
    epoch_pref_loss = 0.0
    epoch_pair_loss = 0.0
    num_batches = 0
    running_correct = 0
    running_total = 0

    # We always need top3/targets for rank@k and K-bucket metrics.
    pred_top3: List[torch.Tensor] = []
    pred_targets: List[torch.Tensor] = []

    # Only collect metadata if we will export predictions.
    collect_metadata = trainer.record_logger is not None
    if collect_metadata:
        pred_video_ids: List[str] = []
        pred_scene_categories: List[str] = []
        pred_person_ids: List[List[int]] = []

    # For bucketed metrics, we carry per-sample K (number of valid persons) to rank0.
    k_values_local: List[int] = []

    # Bucketed metrics by number of valid persons K.
    # Keys: "k2", "k3", "k4", "k5p".
    bucket_keys = ("k2", "k3", "k4", "k5p")
    bucket_totals = {k: 0 for k in bucket_keys}
    bucket_hits = {k: {1: 0, 2: 0, 3: 0} for k in bucket_keys}

    is_main = trainer._is_main_process()
    iterator = trainer.val_dataloader
    if is_main:
        iterator = tqdm(
            iterator,
            desc=f"Val Epoch {trainer.current_epoch + 1}",
            leave=True,
            dynamic_ncols=True,
        )

    for batch_idx, batch in enumerate(iterator):
        batch = trainer._move_batch_to_device(batch)
        with amp.autocast(device_type=trainer.device.type, enabled=trainer.use_mixed_precision):
            # Support both raw data and cached features mode
            if "vision_features" in batch and "geometry_features" in batch:
                outputs = trainer.model(
                    vision_features=batch["vision_features"],
                    geometry_features=batch["geometry_features"],
                    person_mask=batch["person_mask"],
                    target_index=batch["target_index"],
                )
            else:
                outputs = trainer.model(
                    frames=batch["frames"],
                    bboxes=batch["bboxes"],
                    person_mask=batch["person_mask"],
                    target_index=batch["target_index"],
                )
            loss_components = trainer.loss_function.get_loss_components(
                importance_logits=outputs["importance_logits"],
                target_index=batch["target_index"],
                person_mask=batch["person_mask"],
                additional_outputs=outputs,
            )

        loss_value = float(loss_components["total_loss"].item())
        epoch_loss += loss_value
        epoch_imp_loss += float(loss_components["importance_loss"].item())
        epoch_pref_loss += float(loss_components["preference_loss"].item())
        epoch_pair_loss += float(loss_components.get("pairwise_loss", 0.0).item())
        num_batches += 1

        acc_metrics = compute_accuracy_metrics(outputs["importance_logits"], batch["target_index"], batch["person_mask"])
        running_correct += acc_metrics["correct_predictions"]
        running_total += acc_metrics["total_samples"]

        pm = batch["person_mask"]
        valid_mask = pm.any(dim=1) if pm.dim() == 3 else pm

        # K per sample = number of valid persons in candidate set.
        # valid_mask: (B,N) bool
        k_batch = valid_mask.bool().sum(dim=1).detach().long().cpu().tolist()
        k_values_local.extend(int(x) for x in k_batch)

        # Precompute top3 indices once for both export and (future) per-batch analysis.
        top3_for_bucket = compute_topk_indices(outputs["importance_logits"], valid_mask, k=3)

        pred_top3.append(top3_for_bucket.detach().long().cpu())
        pred_targets.append(batch["target_index"].detach().long().cpu())

        if collect_metadata:
            B = int(batch["target_index"].shape[0])
            videos = batch.get("video_id")
            if isinstance(videos, (list, tuple)):
                pred_video_ids.extend(str(v) for v in videos)
            else:
                pred_video_ids.extend([""] * B)

            scenes = batch.get("scene_category")
            if isinstance(scenes, (list, tuple)):
                pred_scene_categories.extend(str(s).replace(" ", "_").lower() for s in scenes)
            elif scenes:
                pred_scene_categories.extend([str(scenes).replace(" ", "_").lower() for _ in range(B)])
            else:
                pred_scene_categories.extend([""] * B)

            original_ids = batch.get("original_ids")
            if isinstance(original_ids, torch.Tensor):
                for row in original_ids.detach():
                    pred_person_ids.append([int(x) for x in row.reshape(-1).tolist()])
            elif isinstance(original_ids, (list, tuple)):
                for row in original_ids:
                    if isinstance(row, torch.Tensor):
                        pred_person_ids.append([int(x) for x in row.detach().reshape(-1).tolist()])
                    elif isinstance(row, (list, tuple)):
                        pred_person_ids.append([int(x) for x in row])
                    else:
                        pred_person_ids.append([])
            else:
                for _ in range(B):
                    pred_person_ids.append([])

        if is_main:
            cum_acc = running_correct / max(1, running_total)
            iterator.set_postfix({"loss": f"{loss_value:.4f}", "acc": f"{cum_acc:.4f}"})

    avg_loss = epoch_loss / max(1, num_batches)
    avg_acc = running_correct / max(1, running_total)

    avg_imp_loss = epoch_imp_loss / max(1, num_batches)
    avg_pref_loss = epoch_pref_loss / max(1, num_batches)
    avg_pair_loss = epoch_pair_loss / max(1, num_batches)

    local_top3 = torch.cat(pred_top3, dim=0)
    local_targets = torch.cat(pred_targets, dim=0)

    # DDP: gather all ranks' predictions to rank0, then compute/export once.
    if dist.is_available() and dist.is_initialized() and trainer.world_size > 1:
        payload = {
            "targets": local_targets,
            "top3": local_top3,
            "video_ids": pred_video_ids if collect_metadata else [],
            "scene_categories": pred_scene_categories if collect_metadata else [],
            "person_ids": pred_person_ids if collect_metadata else [],
            "k_values": k_values_local,
        }

        gathered = [None for _ in range(int(trainer.world_size))] if trainer.rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)

        if trainer.rank != 0:
            # Non-main ranks don't compute global metrics/export.
            return avg_loss, avg_acc, 0.0, 0.0, 0.0, avg_imp_loss, avg_pref_loss, avg_pair_loss

        targets_all = torch.cat([g["targets"] for g in gathered if g is not None], dim=0)
        top3_all = torch.cat([g["top3"] for g in gathered if g is not None], dim=0)
        video_ids_all: List[str] = []
        scene_all: List[str] = []
        person_ids_all: List[List[int]] = []
        k_values_all: List[int] = []
        for g in gathered:
            if g is None:
                continue
            video_ids_all.extend(g.get("video_ids", []) or [])
            scene_all.extend(g.get("scene_categories", []) or [])
            person_ids_all.extend(g.get("person_ids", []) or [])
            k_values_all.extend([int(x) for x in (g.get("k_values", []) or [])])
    else:
        targets_all = local_targets
        top3_all = local_top3
        video_ids_all = pred_video_ids if collect_metadata else []
        scene_all = pred_scene_categories if collect_metadata else []
        person_ids_all = pred_person_ids if collect_metadata else []
        k_values_all = k_values_local

    # Build K-bucket metrics on rank0 using gathered targets/top3 and true K (from person_mask).
    bucket_totals = {k: 0 for k in bucket_keys}
    bucket_hits = {k: {1: 0, 2: 0, 3: 0} for k in bucket_keys}
    targets_cpu_for_bucket = targets_all.detach().long().cpu()
    top3_cpu_for_bucket = top3_all.detach().long().cpu()

    def _bucket_key(k_val: int) -> str:
        if k_val <= 2:
            return "k2"
        if k_val == 3:
            return "k3"
        if k_val == 4:
            return "k4"
        return "k5p"

    for i in range(int(top3_cpu_for_bucket.shape[0])):
        tgt_i = int(targets_cpu_for_bucket[i].item())
        k_val = int(k_values_all[i]) if i < len(k_values_all) else int(top3_cpu_for_bucket.shape[1])
        bkey = _bucket_key(k_val)
        bucket_totals[bkey] += 1
        row = top3_cpu_for_bucket[i].tolist()
        if tgt_i in row[:1]:
            bucket_hits[bkey][1] += 1
        if tgt_i in row[:2]:
            bucket_hits[bkey][2] += 1
        if tgt_i in row[:3]:
            bucket_hits[bkey][3] += 1

    rank_by_k = {}
    for bk in bucket_keys:
        total = bucket_totals[bk]
        if total <= 0:
            rank_by_k[bk] = {"rank1": 0.0, "rank2": 0.0, "rank3": 0.0, "n": 0}
            continue
        h = bucket_hits[bk]
        rank_by_k[bk] = {
            "rank1": (h[1] / total) * 100.0,
            "rank2": (h[2] / total) * 100.0,
            "rank3": (h[3] / total) * 100.0,
            "n": int(total),
        }

    r1_ratio = compute_rankk_from_topk(top3_all, targets_all, 1)
    r2_ratio = compute_rankk_from_topk(top3_all, targets_all, 2)
    r3_ratio = compute_rankk_from_topk(top3_all, targets_all, 3)
    trainer._rankk_cache = {1: r1_ratio, 2: r2_ratio, 3: r3_ratio}
    avg_acc = r1_ratio

    # Keep return values in percent (0-100) for trainer plots/logs compatibility.
    rank1 = r1_ratio * 100.0
    rank2 = r2_ratio * 100.0
    rank3 = r3_ratio * 100.0

    if collect_metadata:
        export_predictions_csv(
            trainer=trainer,
            split="val",
            epoch_1based=int(trainer.current_epoch) + 1,
            targets=targets_all,
            predicted_index=top3_all[:, :1].squeeze(1),
            video_ids=video_ids_all,
            scene_categories=scene_all,
            person_ids=person_ids_all,
        )

    classwise: Dict[str, Dict[str, float]] = {}
    totals: Dict[str, int] = {}
    hits: Dict[str, Dict[int, int]] = {}

    targets_cpu = targets_all.detach().long().cpu()
    top3_cpu = top3_all.detach().long().cpu()

    for i in range(int(top3_cpu.shape[0])):
        tgt_i = int(targets_cpu[i].item())
        scene = str(scene_all[i] if i < len(scene_all) else "").replace(" ", "_").lower()
        totals[scene] = totals.get(scene, 0) + 1
        if scene not in hits:
            hits[scene] = {1: 0, 2: 0, 3: 0}

        row_top3 = top3_cpu[i].tolist()
        if tgt_i in row_top3[:1]:
            hits[scene][1] += 1
        if tgt_i in row_top3[:2]:
            hits[scene][2] += 1
        if tgt_i in row_top3[:3]:
            hits[scene][3] += 1

    for scene, total in totals.items():
        if total <= 0:
            continue
        h = hits.get(scene, {1: 0, 2: 0, 3: 0})
        classwise[scene] = {
            "rank@1": (h[1] / total) * 100.0,
            "rank@2": (h[2] / total) * 100.0,
            "rank@3": (h[3] / total) * 100.0,
            "rank1_correct": float(h[1]),
            "total": float(total),
        }

    trainer.record_logger.log_classwise_metrics(int(trainer.current_epoch) + 1, classwise)
    # Attach bucketed rank@k for trainer logging if available.
    trainer._rank_by_k = rank_by_k
    return avg_loss, avg_acc, rank1, rank2, rank3, avg_imp_loss, avg_pref_loss, avg_pair_loss
