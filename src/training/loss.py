"""Loss functions.
- For each sample, the target person's logit should be higher than all other
    valid persons.
- We aggregate negatives with logsumexp (soft-hard-negative) and apply a
    logistic preference loss:

        neg_soft = logsumexp(neg_logits)
        loss = softplus(beta * (neg_soft - pos))

We keep a `CombinedLoss` facade for compatibility with trainer code.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PreferenceOptimizationLoss(nn.Module):
    """Preference optimization with soft-negative aggregation.

    Assumption/contract: person_mask is always 3D (B, T, N).
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.beta = float(beta)
        self.reduction = str(reduction)

    def forward(
        self,
        importance_logits: torch.Tensor,  # (B,N)
        target_index: torch.Tensor,  # (B,)
        person_mask: torch.Tensor,  # (B,T,N)
    ) -> torch.Tensor:
        if importance_logits.ndim != 2:
            raise ValueError("importance_logits 必须是二维张量 (B, N)")

        if person_mask.ndim != 3:
            raise ValueError("person_mask 必须是三维张量 (B, T, N)")

        B, N = importance_logits.shape
        valid_mask = person_mask.any(dim=1)  # (B,N)

        t = target_index.long()
        pos = importance_logits.gather(1, t.unsqueeze(1)).squeeze(1)  # (B,)

        neg_mask = valid_mask.clone()
        neg_mask.scatter_(1, t.unsqueeze(1), False)
        neg_logits = importance_logits.masked_fill(~neg_mask, -1e4)
        neg_lse = torch.logsumexp(neg_logits, dim=1)  # (B,)
        neg_count = neg_mask.sum(dim=1).to(dtype=neg_lse.dtype).clamp(min=1.0)  # (B,)
        neg_soft = neg_lse - neg_count.log()

        per_sample = F.softplus(self.beta * (neg_soft - pos))
        if self.reduction == "sum":
            return per_sample.sum()
        if self.reduction == "none":
            return per_sample
        return per_sample.mean()


class HardNegativePairwiseLoss(nn.Module):
    def __init__(self, beta: float = 1.0, margin: float = 0.5, reduction: str = "mean") -> None:
        super().__init__()
        self.beta = float(beta)
        self.margin = float(margin)
        self.reduction = str(reduction)

    def forward(
        self,
        importance_logits: torch.Tensor,  # (B,N)
        target_index: torch.Tensor,  # (B,)
        person_mask: torch.Tensor,  # (B,T,N)
    ) -> torch.Tensor:
        if importance_logits.ndim != 2:
            raise ValueError("importance_logits 必须是二维张量 (B, N)")
        if person_mask.ndim != 3:
            raise ValueError("person_mask 必须是三维张量 (B, T, N)")

        valid_mask = person_mask.any(dim=1)  # (B,N)
        t = target_index.long()
        pos = importance_logits.gather(1, t.unsqueeze(1)).squeeze(1)  # (B,)

        neg_mask = valid_mask.clone()
        neg_mask.scatter_(1, t.unsqueeze(1), False)
        neg_logits = importance_logits.masked_fill(~neg_mask, -1e4)
        neg_max = neg_logits.max(dim=1).values

        per_sample = F.softplus(self.beta * (neg_max - pos + self.margin))
        if self.reduction == "sum":
            return per_sample.sum()
        if self.reduction == "none":
            return per_sample
        return per_sample.mean()


class ImportanceLoss(nn.Module):
    """Optional cross entropy (kept for compatibility)."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, logits: torch.Tensor, target_index: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        masked = logits.masked_fill(~valid_mask, -1e4)
        return self.criterion(masked, target_index.long())


class CombinedLoss(nn.Module):
    """Compatibility facade for trainer.

    Exposes get_loss_components() returning a dict with keys expected by loops/trainer.
    """

    def __init__(self, config: Any, **kwargs) -> None:
        super().__init__()

        beta_cfg = float(config.model.loss.beta)
        pairwise_margin = float(getattr(config.model.loss, "pairwise_margin", 0.5))

        self.importance = ImportanceLoss()
        self.preference = PreferenceOptimizationLoss(beta=beta_cfg, reduction="mean")
        self.pairwise = HardNegativePairwiseLoss(beta=beta_cfg, margin=pairwise_margin, reduction="mean")

        self.importance_weight = float(config.model.loss.importance_weight)
        self.preference_weight = float(config.model.loss.preference_weight)
        self.pairwise_weight = float(getattr(config.model.loss, "pairwise_weight", 0.0))

        logger.info(
            "CombinedLoss: importance_weight=%.3f preference_weight=%.3f pairwise_weight=%.3f beta=%.3f",
            self.importance_weight,
            self.preference_weight,
            self.pairwise_weight,
            beta_cfg,
        )

    def get_loss_components(
        self,
        *,
        importance_logits: torch.Tensor,
        target_index: torch.Tensor,
        person_mask: torch.Tensor,
        additional_outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if person_mask.ndim != 3:
            raise ValueError("person_mask 必须是三维张量 (B, T, N)")
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

        pair_loss = (
            self.pairwise(importance_logits, target_index, person_mask) * self.pairwise_weight
            if self.pairwise_weight > 0
            else importance_logits.new_tensor(0.0)
        )

        total = imp_loss + pref_loss + pair_loss
        return {
            "importance_loss": imp_loss.detach(),
            "preference_loss": pref_loss.detach(),
            "pairwise_loss": pair_loss.detach(),
            "total_loss": total,
        }
