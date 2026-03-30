"""Dynamic unary prior for person-centric importance.

This branch estimates how important a person looks before social reasoning.
It keeps only person-local evidence: appearance, geometry, visibility, and the
person's own motion pattern over time.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp(min=1.0)


def _masked_max(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    neg = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask, neg)
    out = masked.max(dim=dim).values
    return torch.where(mask.any(dim=dim), out, torch.zeros_like(out))


def _masked_temporal_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def _masked_temporal_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask, neg)
    out = masked.max(dim=1).values
    return torch.where(mask.any(dim=1), out, torch.zeros_like(out))


class IntrinsicImportanceModule(nn.Module):
    def __init__(
        self,
        *,
        fused_dim: int,
        hidden_dim: int,
        dropout: float,
        use_geom_priors: bool,
        use_motion_priors: bool,
        use_max_pool: bool,
        use_attention_pool: bool,
    ) -> None:
        super().__init__()
        self.use_geom_priors = bool(use_geom_priors)
        self.use_motion_priors = bool(use_motion_priors)
        self.use_max_pool = bool(use_max_pool)
        self.use_attention_pool = bool(use_attention_pool)
        self.hidden_dim = int(hidden_dim)

        base_prior_dim = 6 if self.use_geom_priors else 1
        motion_prior_dim = 4 if self.use_motion_priors else 1
        self.prior_dim = base_prior_dim + motion_prior_dim

        self.temporal_proj = nn.Sequential(
            nn.LayerNorm(int(fused_dim)),
            nn.Linear(int(fused_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.temporal_attn = None
        if self.use_attention_pool:
            self.temporal_attn = nn.Sequential(
                nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
                nn.GELU(),
                nn.Linear(int(hidden_dim) // 2, 1),
            )
        self.pool_selector = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) + self.prior_dim),
            nn.Linear(int(hidden_dim) + self.prior_dim, int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, 3),
        )
        self.encoder = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) + self.prior_dim),
            nn.Linear(int(hidden_dim) + self.prior_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def _build_priors(self, fused: torch.Tensor, bboxes: torch.Tensor, person_mask: torch.Tensor) -> torch.Tensor:
        valid = person_mask.bool()
        visible_ratio = valid.float().mean(dim=1, keepdim=False).unsqueeze(-1)

        feat_delta = torch.linalg.vector_norm(fused[:, 1:] - fused[:, :-1], dim=-1)
        valid_pair = valid[:, 1:] & valid[:, :-1]
        feat_delta_mean = _masked_temporal_mean(feat_delta, valid_pair).unsqueeze(-1)
        if not self.use_geom_priors:
            valid_person = valid.any(dim=1).unsqueeze(-1).to(dtype=fused.dtype)
            if self.use_motion_priors:
                zero = torch.zeros_like(feat_delta_mean)
                return torch.cat([visible_ratio, zero, zero, feat_delta_mean, feat_delta_mean], dim=-1) * valid_person
            return torch.cat([visible_ratio, feat_delta_mean], dim=-1) * valid_person

        x1, y1, x2, y2 = bboxes.unbind(dim=-1)
        w = (x2 - x1).clamp(min=0.0)
        h = (y2 - y1).clamp(min=0.0)
        area = w * h
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        center_dist = torch.sqrt((cx - 0.5).square() + (cy - 0.5).square())
        center_bias = (1.0 - center_dist / 0.70710678).clamp(min=0.0, max=1.0)

        area_mean = _masked_mean(area, valid, dim=1).unsqueeze(-1)
        area_max = _masked_max(area, valid, dim=1).unsqueeze(-1)
        center_mean = _masked_mean(center_bias, valid, dim=1).unsqueeze(-1)
        center_max = _masked_max(center_bias, valid, dim=1).unsqueeze(-1)
        height_mean = _masked_mean(h, valid, dim=1).unsqueeze(-1)

        static_priors = [visible_ratio, area_mean, area_max, center_mean, center_max, height_mean]
        center_dx = cx[:, 1:] - cx[:, :-1]
        center_dy = cy[:, 1:] - cy[:, :-1]
        motion_mag = torch.sqrt(center_dx.square() + center_dy.square())
        motion_mean = _masked_temporal_mean(motion_mag, valid_pair).unsqueeze(-1)
        motion_max = _masked_temporal_max(motion_mag, valid_pair).unsqueeze(-1)

        area_delta = (area[:, 1:] - area[:, :-1]).abs()
        area_delta_mean = _masked_temporal_mean(area_delta, valid_pair).unsqueeze(-1)
        valid_person = valid.any(dim=1).unsqueeze(-1).to(dtype=bboxes.dtype)
        priors = list(static_priors)
        if self.use_motion_priors:
            priors.extend([motion_mean, motion_max, area_delta_mean, feat_delta_mean])
        else:
            priors.append(feat_delta_mean)
        return torch.cat(priors, dim=-1) * valid_person

    def forward(
        self,
        fused: torch.Tensor,
        bboxes: torch.Tensor,
        person_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = person_mask.bool()
        valid_person = valid.any(dim=1)
        h = self.temporal_proj(fused)
        mask_f = valid.to(dtype=h.dtype).unsqueeze(-1)
        mean_pool = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        max_pool = _masked_max(h, valid.unsqueeze(-1), dim=1) if self.use_max_pool else torch.zeros_like(mean_pool)
        if self.use_attention_pool and self.temporal_attn is not None:
            attn_logits = self.temporal_attn(h).squeeze(-1)
            attn_logits = attn_logits.masked_fill(~valid, -1e4)
            attn = torch.softmax(attn_logits, dim=1)
            attn = attn * valid.to(dtype=attn.dtype)
            attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
            attn_pool = (h * attn.unsqueeze(-1)).sum(dim=1)
        else:
            attn_pool = torch.zeros_like(mean_pool)
        priors = self._build_priors(fused, bboxes, valid)
        selector_logits = self.pool_selector(torch.cat([mean_pool, priors.to(dtype=h.dtype)], dim=-1))
        pool_mask = selector_logits.new_tensor([
            1.0,
            1.0 if self.use_max_pool else 0.0,
            1.0 if self.use_attention_pool else 0.0,
        ])
        selector_logits = selector_logits.masked_fill(pool_mask.unsqueeze(0).eq(0.0), -1e4)
        selector = torch.softmax(selector_logits, dim=-1)
        pool_stack = torch.stack([mean_pool, max_pool, attn_pool], dim=2)
        mixed_pool = (pool_stack * selector.unsqueeze(-1)).sum(dim=2)
        intrinsic_feat = self.encoder(torch.cat([mixed_pool, priors.to(dtype=h.dtype)], dim=-1))
        intrinsic_logits = self.scorer(intrinsic_feat).squeeze(-1)
        intrinsic_logits = intrinsic_logits.masked_fill(~valid_person, -1e4)
        intrinsic_feat = intrinsic_feat.masked_fill(~valid_person.unsqueeze(-1), 0.0)
        return intrinsic_logits, intrinsic_feat, priors
