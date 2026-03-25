"""Train and validation epoch loops."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch import amp
from tqdm import tqdm

from src.engine.metrics import (
    compute_accuracy_metrics,
    compute_topk_indices,
    compute_rankk_from_topk,
    mean_abs_valid,
)

logger = logging.getLogger(__name__)


def train_epoch(trainer, log_every: int = 10) -> Tuple[float, float, float, float, Dict[str, float]]:
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)

    epoch_loss = 0.0
    epoch_imp_loss = 0.0
    epoch_pref_loss = 0.0
    num_batches = 0

    self_abs_sum = 0.0
    rel_abs_sum = 0.0

    running_correct = 0
    running_total = 0

    collect_predictions = (
        trainer._is_main_process()
        and trainer.record_logger is not None
        and bool(getattr(trainer.config.training, "export_train_predictions", False))
    )
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
        batch = trainer._move_batch_to_device(batch)

        with amp.autocast(device_type=trainer.device.type, enabled=trainer.use_mixed_precision):
            outputs = trainer.model(
                frames=batch["frames"],
                bboxes=batch["bboxes"],
                person_mask=batch["person_mask"],
                target_index=batch["target_index"],
            )

            scene_cat = batch.get("scene_category_idx")
            if scene_cat is not None and isinstance(scene_cat, torch.Tensor):
                scene_cat = scene_cat.to(trainer.device)

            loss_components = trainer.loss_function.get_loss_components(
                importance_logits=outputs["importance_logits"],
                target_index=batch["target_index"],
                person_mask=batch["person_mask"],
                model_outputs=outputs,
                scene_category=scene_cat,
            )
            loss = loss_components["total_loss"] / trainer.accumulation_steps

        pm = batch["person_mask"]
        valid_mask = pm.any(dim=1) if pm.dim() == 3 else pm
        self_abs_sum += mean_abs_valid(outputs.get("importance_logits_self"), valid_mask)
        rel_abs_sum += mean_abs_valid(outputs.get("importance_logits_rel"), valid_mask)

        if torch.isnan(loss):
            trainer.optimizer.zero_grad(set_to_none=True)
            continue

        if trainer.scaler is not None:
            trainer.scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_counter += 1
        if accum_counter % trainer.accumulation_steps == 0:
            if trainer.scaler is not None:
                trainer.scaler.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.max_grad_norm)

            if trainer.scaler is not None:
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
            else:
                trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.global_step += 1

        loss_value = float(loss_components["total_loss"].item())
        epoch_loss += loss_value
        epoch_imp_loss += float(loss_components["importance_loss"].item())
        epoch_pref_loss += float(loss_components["preference_loss"].item())
        num_batches += 1

        acc_metrics = compute_accuracy_metrics(outputs["importance_logits"], batch["target_index"], batch["person_mask"])
        running_correct += acc_metrics["correct_predictions"]
        running_total += acc_metrics["total_samples"]
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
            iterator.set_postfix({"loss": f"{loss_value:.4f}", "acc": f"{cum_acc:.4f}"})

    # Flush remaining micro-batches
    if accum_counter % trainer.accumulation_steps != 0:
        if trainer.scaler is not None:
            trainer.scaler.unscale_(trainer.optimizer)
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.max_grad_norm)

        if trainer.scaler is not None:
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        else:
            trainer.optimizer.step()
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.global_step += 1

    avg_loss = epoch_loss / max(1, num_batches)
    avg_acc = running_correct / max(1, running_total)

    if collect_predictions and pred_top3:
        top3_all = torch.cat(pred_top3, dim=0)
        targets_all = torch.cat(pred_targets, dim=0)
        trainer._rankk_cache = {
            1: compute_rankk_from_topk(top3_all, targets_all, 1),
            2: compute_rankk_from_topk(top3_all, targets_all, 2),
            3: compute_rankk_from_topk(top3_all, targets_all, 3),
        }
        if trainer.record_logger is not None and trainer._is_main_process():
            trainer.record_logger.export_predictions_csv(
                split="train",
                epoch=int(trainer.current_epoch) + 1,
                targets=targets_all,
                predicted_index=top3_all[:, :1].squeeze(1),
                video_ids=pred_video_ids,
                scene_categories=pred_scene_categories,
                person_ids=pred_person_ids,
                rankk_cache=getattr(trainer, "_rankk_cache", {}),
            )

    avg_imp_loss = epoch_imp_loss / max(1, num_batches)
    avg_pref_loss = epoch_pref_loss / max(1, num_batches)

    diagnostics = {
        "train_self_logit_abs": self_abs_sum / max(1, num_batches),
        "train_rel_logit_abs": rel_abs_sum / max(1, num_batches),
    }
    return avg_loss, avg_acc, avg_imp_loss, avg_pref_loss, diagnostics


@torch.no_grad()
def validate_epoch(trainer, log_every: int = 10) -> Tuple[float, float, float, float, float, float, float, Dict[str, float]]:
    trainer.model.eval()

    epoch_loss = 0.0
    epoch_imp_loss = 0.0
    epoch_pref_loss = 0.0
    num_batches = 0

    self_abs_sum = 0.0
    rel_abs_sum = 0.0

    running_correct = 0
    running_total = 0

    pred_top3: List[torch.Tensor] = []
    pred_targets: List[torch.Tensor] = []

    collect_metadata = trainer.record_logger is not None
    if collect_metadata:
        pred_video_ids: List[str] = []
        pred_scene_categories: List[str] = []
        pred_person_ids: List[List[int]] = []

    k_values_local: List[int] = []
    bucket_keys = ("k2", "k3", "k4", "k5p")

    is_main = trainer._is_main_process()
    iterator = trainer.val_dataloader
    if is_main:
        iterator = tqdm(iterator, desc=f"Val Epoch {trainer.current_epoch + 1}", leave=True, dynamic_ncols=True)

    for batch_idx, batch in enumerate(iterator):
        batch = trainer._move_batch_to_device(batch)

        scene_cat = batch.get("scene_category_idx")
        if scene_cat is not None and isinstance(scene_cat, torch.Tensor):
            scene_cat = scene_cat.to(trainer.device)

        with amp.autocast(device_type=trainer.device.type, enabled=trainer.use_mixed_precision):
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
                model_outputs=outputs,
                scene_category=scene_cat,
            )

        loss_value = float(loss_components["total_loss"].item())
        epoch_loss += loss_value
        epoch_imp_loss += float(loss_components["importance_loss"].item())
        epoch_pref_loss += float(loss_components["preference_loss"].item())
        num_batches += 1

        acc_metrics = compute_accuracy_metrics(outputs["importance_logits"], batch["target_index"], batch["person_mask"])
        running_correct += acc_metrics["correct_predictions"]
        running_total += acc_metrics["total_samples"]

        pm = batch["person_mask"]
        valid_mask = pm.any(dim=1) if pm.dim() == 3 else pm
        self_abs_sum += mean_abs_valid(outputs.get("importance_logits_self"), valid_mask)
        rel_abs_sum += mean_abs_valid(outputs.get("importance_logits_rel"), valid_mask)

        k_batch = valid_mask.bool().sum(dim=1).detach().long().cpu().tolist()
        k_values_local.extend(int(x) for x in k_batch)

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

    local_top3 = torch.cat(pred_top3, dim=0)
    local_targets = torch.cat(pred_targets, dim=0)

    # DDP: gather all ranks' predictions to rank0
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
            diagnostics = {
                "val_self_logit_abs": self_abs_sum / max(1, num_batches),
                "val_rel_logit_abs": rel_abs_sum / max(1, num_batches),
            }
            return avg_loss, avg_acc, 0.0, 0.0, 0.0, avg_imp_loss, avg_pref_loss, diagnostics

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

    # K-bucket metrics
    def _bucket_key(k_val: int) -> str:
        if k_val <= 2: return "k2"
        if k_val == 3: return "k3"
        if k_val == 4: return "k4"
        return "k5p"

    bucket_totals = {k: 0 for k in bucket_keys}
    bucket_hits = {k: {1: 0, 2: 0, 3: 0} for k in bucket_keys}
    targets_cpu_for_bucket = targets_all.detach().long().cpu()
    top3_cpu_for_bucket = top3_all.detach().long().cpu()

    for i in range(int(top3_cpu_for_bucket.shape[0])):
        tgt_i = int(targets_cpu_for_bucket[i].item())
        k_val = int(k_values_all[i]) if i < len(k_values_all) else int(top3_cpu_for_bucket.shape[1])
        bkey = _bucket_key(k_val)
        bucket_totals[bkey] += 1
        row = top3_cpu_for_bucket[i].tolist()
        if tgt_i in row[:1]: bucket_hits[bkey][1] += 1
        if tgt_i in row[:2]: bucket_hits[bkey][2] += 1
        if tgt_i in row[:3]: bucket_hits[bkey][3] += 1

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

    rank1 = r1_ratio * 100.0
    rank2 = r2_ratio * 100.0
    rank3 = r3_ratio * 100.0

    if collect_metadata and trainer.record_logger is not None and trainer._is_main_process():
        trainer.record_logger.export_predictions_csv(
            split="val",
            epoch=int(trainer.current_epoch) + 1,
            targets=targets_all,
            predicted_index=top3_all[:, :1].squeeze(1),
            video_ids=video_ids_all,
            scene_categories=scene_all,
            person_ids=person_ids_all,
            rankk_cache=getattr(trainer, "_rankk_cache", {}),
        )

    # Per-scene (classwise) metrics
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
        if tgt_i in row_top3[:1]: hits[scene][1] += 1
        if tgt_i in row_top3[:2]: hits[scene][2] += 1
        if tgt_i in row_top3[:3]: hits[scene][3] += 1

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

    if trainer.record_logger is not None:
        trainer.record_logger.log_classwise_metrics(int(trainer.current_epoch) + 1, classwise)
    trainer._rank_by_k = rank_by_k

    diagnostics = {
        "val_self_logit_abs": self_abs_sum / max(1, num_batches),
        "val_rel_logit_abs": rel_abs_sum / max(1, num_batches),
    }
    return avg_loss, avg_acc, rank1, rank2, rank3, avg_imp_loss, avg_pref_loss, diagnostics
