"""Loss functions.

Only the final fused logits are supervised — no independent per-branch CE.

Components:
- ImportanceLoss: CE on fused logits (with label smoothing)
- PreferenceOptimizationLoss: margin-based ranking loss to push target above negatives
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PreferenceOptimizationLoss(nn.Module):
    def __init__(self, beta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.beta = float(beta)
        self.reduction = str(reduction)

    def forward(
        self,
        importance_logits: torch.Tensor,
        target_index: torch.Tensor,
        person_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N = importance_logits.shape
        valid_mask = person_mask.any(dim=1)

        t = target_index.long()
        pos = importance_logits.gather(1, t.unsqueeze(1)).squeeze(1)

        neg_mask = valid_mask.clone()
        neg_mask.scatter_(1, t.unsqueeze(1), False)
        neg_logits = importance_logits.masked_fill(~neg_mask, -1e4)
        neg_lse = torch.logsumexp(neg_logits, dim=1)
        neg_count = neg_mask.sum(dim=1).to(dtype=neg_lse.dtype).clamp(min=1.0)
        neg_soft = neg_lse - neg_count.log()

        per_sample = F.softplus(self.beta * (neg_soft - pos))
        if self.reduction == "sum":
            return per_sample.sum()
        if self.reduction == "none":
            return per_sample
        return per_sample.mean()


class ImportanceLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, target_index: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        masked = logits.masked_fill(~valid_mask, -1e4)
        return self.criterion(masked, target_index.long())


class CombinedLoss(nn.Module):
    def __init__(self, config: Any, **kwargs) -> None:
        super().__init__()

        beta_cfg = float(config.model.loss.beta)
        self.importance = ImportanceLoss()
        self.preference = PreferenceOptimizationLoss(beta=beta_cfg, reduction="mean")

        self.importance_weight = float(config.model.loss.importance_weight)
        self.preference_weight = float(config.model.loss.preference_weight)

        logger.info(
            "CombinedLoss: imp=%.3f pref=%.3f beta=%.3f",
            self.importance_weight, self.preference_weight, beta_cfg,
        )

    def get_loss_components(
        self,
        *,
        importance_logits: torch.Tensor,
        target_index: torch.Tensor,
        person_mask: torch.Tensor,
        model_outputs: Optional[Dict[str, torch.Tensor]] = None,
        scene_category: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        valid_mask = person_mask.any(dim=1)

        imp_loss = (
            self.importance(importance_logits, target_index, valid_mask) * self.importance_weight
            if self.importance_weight > 0
            else importance_logits.new_tensor(0.0)
        )
        pref_loss = (
            self.preference(importance_logits, target_index, person_mask) * self.preference_weight
            if self.preference_weight > 0
            else importance_logits.new_tensor(0.0)
        )

        total = imp_loss + pref_loss

        return {
            "importance_loss": imp_loss.detach(),
            "preference_loss": pref_loss.detach(),
            "total_loss": total,
        }
