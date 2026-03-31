"""Loss functions.

The final rank score remains primary, with light auxiliary supervision for the
unary and relation branches plus weak prototype consistency for scene context.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _ranking_margin(logits: torch.Tensor, target_index: torch.Tensor, person_mask: torch.Tensor) -> torch.Tensor:
    valid_mask = person_mask.any(dim=1)
    t = target_index.long()
    pos = logits.gather(1, t.unsqueeze(1)).squeeze(1)
    neg_mask = valid_mask.clone()
    neg_mask.scatter_(1, t.unsqueeze(1), False)
    neg_logits = logits.masked_fill(~neg_mask, -1e4)
    neg_lse = torch.logsumexp(neg_logits, dim=1)
    neg_count = neg_mask.sum(dim=1).to(dtype=neg_lse.dtype).clamp(min=1.0)
    neg_soft = neg_lse - neg_count.log()
    return pos - neg_soft


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
        t = target_index.long()
        pos = importance_logits.gather(1, t.unsqueeze(1)).squeeze(1)

        neg_mask = person_mask.any(dim=1).clone()
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
        self.intrinsic_aux_weight = float(getattr(config.model.loss, "intrinsic_aux_weight", 0.0))
        self.relation_aux_weight = float(getattr(config.model.loss, "relation_aux_weight", 0.0))
        self.scene_consistency_weight = float(getattr(config.model.loss, "scene_consistency_weight", 0.0))
        logger.info(
            "CombinedLoss: imp=%.3f pref=%.3f intrinsic=%.3f relation=%.3f scene=%.3f beta=%.3f",
            self.importance_weight,
            self.preference_weight,
            self.intrinsic_aux_weight,
            self.relation_aux_weight,
            self.scene_consistency_weight,
            beta_cfg,
        )

    def get_loss_components(
        self,
        *,
        importance_logits: torch.Tensor,
        target_index: torch.Tensor,
        person_mask: torch.Tensor,
        model_outputs: Optional[Dict[str, torch.Tensor]] = None,
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

        intrinsic_logits = model_outputs.get("intrinsic_logits") if model_outputs else None
        intrinsic_aux_loss = (
            self.importance(intrinsic_logits, target_index, valid_mask) * self.intrinsic_aux_weight
            if intrinsic_logits is not None and self.intrinsic_aux_weight > 0
            else importance_logits.new_tensor(0.0)
        )

        relation_logits = model_outputs.get("importance_logits_rel") if model_outputs else None
        relation_aux_loss = (
            self.importance(relation_logits, target_index, valid_mask) * self.relation_aux_weight
            if relation_logits is not None and self.relation_aux_weight > 0
            else importance_logits.new_tensor(0.0)
        )

        scene_logits = model_outputs.get("scene_logits") if model_outputs else None
        scene_key_logits = model_outputs.get("scene_tokens") if model_outputs else None
        if scene_logits is not None and scene_key_logits is not None and self.scene_consistency_weight > 0:
            scene_target = torch.softmax(scene_logits, dim=-1).unsqueeze(1)
            scene_key_logprob = torch.log_softmax(scene_key_logits, dim=-1)
            scene_consistency = F.kl_div(scene_key_logprob, scene_target.expand_as(scene_key_logits), reduction="batchmean")
            proto_usage = torch.softmax(scene_logits, dim=-1).mean(dim=0)
            scene_diversity = -(proto_usage.clamp(min=1e-6).log() * proto_usage).sum()
            scene_loss = self.scene_consistency_weight * (scene_consistency - 0.05 * scene_diversity)
        else:
            scene_loss = importance_logits.new_tensor(0.0)

        total = imp_loss + pref_loss + intrinsic_aux_loss + relation_aux_loss + scene_loss

        return {
            "importance_loss": imp_loss.detach(),
            "preference_loss": pref_loss.detach(),
            "intrinsic_aux_loss": intrinsic_aux_loss.detach(),
            "relation_aux_loss": relation_aux_loss.detach(),
            "scene_loss": scene_loss.detach(),
            "total_loss": total,
        }
