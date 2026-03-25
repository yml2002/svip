"""Evaluation metrics for importance ranking.

Pure functions — no trainer dependency, safe to call from loops or standalone.
"""

from __future__ import annotations

from typing import Dict

import torch


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
    mask = valid_mask.bool()
    if mask.dim() > 2:
        mask = mask.any(dim=1)
    masked_logits = importance_logits.masked_fill(~mask, float("-inf"))
    k_eff = min(int(k), int(masked_logits.shape[1]))
    return masked_logits.topk(k=k_eff, dim=1).indices


@torch.no_grad()
def compute_rankk_from_topk(topk_indices: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    if topk_indices.numel() == 0:
        return 0.0
    tgt = targets.long()
    hit = (topk_indices[:, :int(k)] == tgt.unsqueeze(1)).any(dim=1).float()
    return float(hit.mean().item())


def mean_abs_valid(logits: torch.Tensor | None, valid_mask: torch.Tensor) -> float:
    """Mean absolute value of logits at valid positions."""
    if logits is None:
        return 0.0
    mask = valid_mask.bool()
    if mask.dim() > 2:
        mask = mask.any(dim=1)
    if not bool(mask.any()):
        return 0.0
    vals = logits.masked_select(mask)
    return float(vals.abs().mean().item()) if vals.numel() > 0 else 0.0


def to_float_scalar(value: torch.Tensor | float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().reshape(-1)[0].item())
    return float(value)
