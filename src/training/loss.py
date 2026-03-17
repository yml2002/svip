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
        self.importance_per_sample = ImportanceLoss(reduction="none")
        self.preference = PreferenceOptimizationLoss(beta=beta_cfg, reduction="mean")
        self.preference_per_sample = PreferenceOptimizationLoss(beta=beta_cfg, reduction="none")

        self.importance_weight = float(config.model.loss.importance_weight)
        self.preference_weight = float(config.model.loss.preference_weight)
        self.self_branch_weight = float(getattr(config.model.loss, "self_branch_weight", 0.0))
        self.rel_branch_weight = float(getattr(config.model.loss, "rel_branch_weight", 0.0))
        self.counterfactual_effect_weight = float(getattr(config.model.loss, "counterfactual_effect_weight", 0.0))
        self.counterfactual_margin = float(getattr(config.model.loss, "counterfactual_margin", 0.0))
        logger.info(
            "CombinedLoss: imp=%.3f pref=%.3f self=%.3f rel=%.3f cf_eff=%.3f cf_margin=%.3f beta=%.3f",
            self.importance_weight,
            self.preference_weight,
            self.self_branch_weight,
            self.rel_branch_weight,
            self.counterfactual_effect_weight,
            self.counterfactual_margin,
            beta_cfg,
        )

    def _weighted_importance(
        self,
        logits: torch.Tensor,
        target_index: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        per_sample = self.importance_per_sample(logits, target_index, valid_mask)
        return (per_sample * sample_weights).mean()

    def _weighted_preference(
        self,
        logits: torch.Tensor,
        target_index: torch.Tensor,
        person_mask: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        per_sample = self.preference_per_sample(logits, target_index, person_mask)
        return (per_sample * sample_weights).mean()

    def _weighted_counterfactual_effect(
        self,
        counterfactual_delta: torch.Tensor,
        target_index: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_weights: torch.Tensor,
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
        return (per_sample * sample_weights).mean()

    def _aux_sample_weights(
        self,
        importance_logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked = importance_logits.masked_fill(~valid_mask, -1e4)
        conf = torch.softmax(masked, dim=1).max(dim=1).values
        hard_ratio = torch.sigmoid((0.72 - conf) / 0.08)

        # Keep a non-zero floor so branches still receive weak supervision on easy samples.
        weights = 0.2 + hard_ratio
        return weights / weights.mean().clamp(min=1e-6)

    def _branch_usage_from_outputs(
        self,
        *,
        model_outputs: Optional[Dict[str, torch.Tensor]],
        branch_name: str,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if model_outputs is None:
            return ref.new_ones((ref.shape[0],))

        branch_weights = model_outputs.get("branch_weights")
        branch_names = model_outputs.get("branch_weight_names")
        if branch_weights is None or not isinstance(branch_names, list) or branch_name not in branch_names:
            return ref.new_ones((ref.shape[0],))

        idx = int(branch_names.index(branch_name))
        if idx >= int(branch_weights.shape[-1]):
            return ref.new_ones((ref.shape[0],))

        sample_usage = branch_weights[:, :, idx].mean(dim=1).detach().clamp(0.0, 1.0)
        # Preserve a floor to avoid zeroing gradient entirely.
        return (0.2 + sample_usage).clamp(max=1.2)

    @staticmethod
    def _branch_is_enabled(
        model_outputs: Optional[Dict[str, torch.Tensor]],
        branch_name: str,
    ) -> bool:
        if model_outputs is None:
            return False
        branch_names = model_outputs.get("branch_weight_names")
        return isinstance(branch_names, list) and branch_name in branch_names

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
        aux_weights = self._aux_sample_weights(importance_logits, valid_mask)
        rel_usage = self._branch_usage_from_outputs(model_outputs=model_outputs, branch_name="rel", ref=importance_logits)
        cf_usage = self._branch_usage_from_outputs(
            model_outputs=model_outputs,
            branch_name="counterfactual",
            ref=importance_logits,
        )

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

        self_branch_loss = importance_logits.new_tensor(0.0)
        rel_branch_loss = importance_logits.new_tensor(0.0)
        counterfactual_effect_loss = importance_logits.new_tensor(0.0)

        if model_outputs is not None:
            self_logits = model_outputs.get("importance_logits_self")
            if self_logits is not None and self.self_branch_weight > 0:
                self_imp = self._weighted_importance(self_logits, target_index, valid_mask, aux_weights)
                self_pref = self._weighted_preference(self_logits, target_index, person_mask, aux_weights)
                self_branch_loss = (
                    self_imp * self.importance_weight + self_pref * self.preference_weight
                ) * self.self_branch_weight

            rel_logits = model_outputs.get("importance_logits_rel")
            if (
                rel_logits is not None
                and self.rel_branch_weight > 0
                and self._branch_is_enabled(model_outputs, "rel")
            ):
                rel_weights = aux_weights * rel_usage
                rel_weights = rel_weights / rel_weights.mean().clamp(min=1e-6)
                rel_imp = self._weighted_importance(rel_logits, target_index, valid_mask, rel_weights)
                rel_pref = self._weighted_preference(rel_logits, target_index, person_mask, rel_weights)
                rel_branch_loss = (rel_imp * self.importance_weight + rel_pref * self.preference_weight) * self.rel_branch_weight

            cf_delta = model_outputs.get("counterfactual_delta")
            if (
                cf_delta is not None
                and self.counterfactual_effect_weight > 0
                and self._branch_is_enabled(model_outputs, "counterfactual")
            ):
                cf_effect_weights = aux_weights * cf_usage
                cf_effect_weights = cf_effect_weights / cf_effect_weights.mean().clamp(min=1e-6)
                cf_effect = self._weighted_counterfactual_effect(
                    cf_delta,
                    target_index,
                    valid_mask,
                    cf_effect_weights,
                )
                counterfactual_effect_loss = cf_effect * self.counterfactual_effect_weight

        total = (
            imp_loss
            + pref_loss
            + self_branch_loss
            + rel_branch_loss
            + counterfactual_effect_loss
        )
        return {
            "importance_loss": imp_loss.detach(),
            "preference_loss": pref_loss.detach(),
            "self_branch_loss": self_branch_loss.detach(),
            "rel_branch_loss": rel_branch_loss.detach(),
            "counterfactual_effect_loss": counterfactual_effect_loss.detach(),
            "total_loss": total,
        }
