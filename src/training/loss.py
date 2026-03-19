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


class ImportanceLoss(nn.Module):
    """Optional cross entropy (kept for compatibility)."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

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
        self.importance = ImportanceLoss()
        self.preference = PreferenceOptimizationLoss(beta=beta_cfg, reduction="mean")

        self.importance_weight = float(config.model.loss.importance_weight)
        self.preference_weight = float(config.model.loss.preference_weight)
        self.counterfactual_effect_weight = float(getattr(config.model.loss, "counterfactual_effect_weight", 0.0))
        self.counterfactual_margin = float(getattr(config.model.loss, "counterfactual_margin", 0.0))
        self.use_relation_branch = bool(getattr(config.model.relation, "enabled", True))
        self.use_counterfactual_branch = bool(getattr(config.model.counterfactual, "enabled", True))
        self.use_self_branch = bool(getattr(getattr(config.model, "self_branch", object()), "enabled", True))
        self.relation_residual_weight = float(getattr(config.model.loss, "relation_residual_weight", 1.0))
        self.counterfactual_residual_weight = float(getattr(config.model.loss, "counterfactual_residual_weight", 1.0))
        # Enforce that each incremental branch should provide at least this relative CE gain over its base.
        self.min_incremental_gain_ratio = 0.02
        logger.info(
            "CombinedLoss: imp=%.3f pref=%.3f rel_res=%.3f cf_res=%.3f cf_eff=%.3f cf_margin=%.3f min_inc_gain=%.3f beta=%.3f",
            self.importance_weight,
            self.preference_weight,
            self.relation_residual_weight,
            self.counterfactual_residual_weight,
            self.counterfactual_effect_weight,
            self.counterfactual_margin,
            self.min_incremental_gain_ratio,
            beta_cfg,
        )

    def _counterfactual_effect(
        self,
        counterfactual_delta: torch.Tensor,
        target_index: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Promote larger counterfactual event impact for the true target than negatives.
        effect_score = torch.linalg.vector_norm(counterfactual_delta, ord=2, dim=-1)
        t = target_index.long()
        pos = effect_score.gather(1, t.unsqueeze(1)).squeeze(1)

        neg_mask = valid_mask.clone()
        neg_mask.scatter_(1, t.unsqueeze(1), False)
        neg_scores = effect_score.masked_fill(~neg_mask, -1e4)
        neg_lse = torch.logsumexp(neg_scores, dim=1)
        neg_count = neg_mask.sum(dim=1).to(dtype=neg_lse.dtype).clamp(min=1.0)
        neg_soft = neg_lse - neg_count.log()

        per_sample = F.softplus(self.counterfactual_margin + (neg_soft - pos))
        return per_sample.mean()

    def _incremental_improvement_loss(
        self,
        *,
        base_logits: torch.Tensor,
        improved_logits: torch.Tensor,
        target_index: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize a branch when its increment fails to improve base enough.

        Let L_base = CE(base), L_inc = CE(base + increment).
        We require L_inc <= L_base * (1 - r), where r is min_incremental_gain_ratio.
        This gives a clean dual effect in minimization:
        - worse than base: penalized
        - slightly better but below target gain: still penalized
        - sufficiently better: zero auxiliary penalty
        """
        base_loss = self.importance(base_logits, target_index, valid_mask)
        improved_loss = self.importance(improved_logits, target_index, valid_mask)
        target_loss = base_loss.detach() * (1.0 - self.min_incremental_gain_ratio)
        return torch.relu(improved_loss - target_loss)

    def get_loss_components(
        self,
        *,
        importance_logits: torch.Tensor,
        target_index: torch.Tensor,
        person_mask: torch.Tensor,
        model_outputs: Optional[Dict[str, torch.Tensor]] = None,
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

        relation_residual_loss = importance_logits.new_tensor(0.0)
        counterfactual_residual_loss = importance_logits.new_tensor(0.0)

        counterfactual_effect_loss = importance_logits.new_tensor(0.0)

        # Residual losses are only meaningful when self branch is present as the base branch.
        rel_res_weight = self.relation_residual_weight if (self.use_relation_branch and self.use_self_branch) else 0.0
        cf_res_weight = self.counterfactual_residual_weight if (self.use_counterfactual_branch and self.use_self_branch) else 0.0

        if model_outputs is not None:
            self_logits = model_outputs.get("importance_logits_self")
            rel_logits = model_outputs.get("importance_logits_rel")
            cf_logits = model_outputs.get("importance_logits_counterfactual")

            if self_logits is not None and rel_logits is not None and rel_res_weight > 0:
                rel_base = self_logits.detach()
                rel_stacked = rel_base + rel_logits
                relation_residual_loss = (
                    self._incremental_improvement_loss(
                        base_logits=rel_base,
                        improved_logits=rel_stacked,
                        target_index=target_index,
                        valid_mask=valid_mask,
                    )
                    * rel_res_weight
                )

            if self_logits is not None and cf_logits is not None and cf_res_weight > 0:
                cf_base = self_logits.detach()
                if rel_logits is not None and self.use_relation_branch:
                    cf_base = cf_base + rel_logits.detach()
                cf_stacked = cf_base + cf_logits
                counterfactual_residual_loss = (
                    self._incremental_improvement_loss(
                        base_logits=cf_base,
                        improved_logits=cf_stacked,
                        target_index=target_index,
                        valid_mask=valid_mask,
                    )
                    * cf_res_weight
                )

            cf_delta = model_outputs.get("counterfactual_delta")
            if cf_delta is not None and self.counterfactual_effect_weight > 0:
                cf_effect = self._counterfactual_effect(
                    counterfactual_delta=cf_delta,
                    target_index=target_index,
                    valid_mask=valid_mask,
                )
                counterfactual_effect_loss = cf_effect * self.counterfactual_effect_weight

        total = (
            imp_loss
            + pref_loss
            + relation_residual_loss
            + counterfactual_residual_loss
            + counterfactual_effect_loss
        )
        return {
            "importance_loss": imp_loss.detach(),
            "preference_loss": pref_loss.detach(),
            "relation_residual_loss": relation_residual_loss.detach(),
            "counterfactual_residual_loss": counterfactual_residual_loss.detach(),
            "counterfactual_effect_loss": counterfactual_effect_loss.detach(),
            "total_loss": total,
        }
