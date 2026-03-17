"""Relation and counterfactual branch reasoners."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    masked = logits.masked_fill(~mask, -1e4)
    weights = torch.softmax(masked, dim=dim)
    weights = weights * mask.to(dtype=weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp(min=1e-6)


class DenseAdapter(nn.Module):
    """Simple per-person adapter for branch feature refinement."""

    def __init__(self, *, in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
            nn.LayerNorm(int(out_dim)),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out * valid_mask.to(dtype=out.dtype).unsqueeze(-1)


class RelationReasoner(nn.Module):
    """Build person-discriminative relation features and logits."""

    def __init__(
        self,
        *,
        token_dim: int,
        out_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        in_dim = int(token_dim) * 4
        self.temporal_attn = nn.Sequential(
            nn.LayerNorm(int(token_dim) * 3),
            nn.Linear(int(token_dim) * 3, 1),
        )
        self.adapter = DenseAdapter(
            in_dim=in_dim,
            out_dim=int(out_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(int(out_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, tokens: torch.Tensor, person_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # tokens: (B,T,N,D), person_mask: (B,T,N)
        valid_mask = person_mask.any(dim=1)
        mask_f = person_mask.to(dtype=tokens.dtype)

        frame_sum = (tokens * mask_f.unsqueeze(-1)).sum(dim=2)  # (B,T,D)
        frame_count = mask_f.sum(dim=2, keepdim=True).clamp(min=1.0)
        frame_mean = frame_sum / frame_count

        other_sum = frame_sum.unsqueeze(2) - tokens * mask_f.unsqueeze(-1)
        other_count = (frame_count.unsqueeze(2) - mask_f.unsqueeze(-1)).clamp(min=1.0)
        other_mean = other_sum / other_count

        token_triplet = torch.cat([tokens, other_mean, tokens - other_mean], dim=-1)
        attn_logits = self.temporal_attn(token_triplet).squeeze(-1)
        attn = _masked_softmax(attn_logits, person_mask, dim=1)

        self_summary = (tokens * attn.unsqueeze(-1)).sum(dim=1)
        other_summary = (other_mean * attn.unsqueeze(-1)).sum(dim=1)
        event_summary = (frame_mean.unsqueeze(2) * attn.unsqueeze(-1)).sum(dim=1)

        relation_input = torch.cat(
            [self_summary, other_summary, event_summary, self_summary - other_summary],
            dim=-1,
        )
        relation_feat = self.adapter(relation_input, valid_mask)
        relation_logits = self.head(relation_feat).squeeze(-1).masked_fill(~valid_mask, -1e4)

        return {
            "relation_features": relation_feat,
            "relation_logits": relation_logits,
            "relation_attention": attn,
        }


class CounterfactualReasoner(nn.Module):
    """Estimate person-level counterfactual effect and logits."""

    def __init__(
        self,
        *,
        token_dim: int,
        out_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        in_dim = int(token_dim) * 4
        self.temporal_attn = nn.Sequential(
            nn.LayerNorm(int(token_dim) * 2),
            nn.Linear(int(token_dim) * 2, 1),
        )
        self.adapter = DenseAdapter(
            in_dim=in_dim,
            out_dim=int(out_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(int(out_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        self.delta_refiner = nn.Sequential(
            nn.Linear(int(out_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(token_dim)),
        )

    def forward(self, tokens: torch.Tensor, person_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # tokens: (B,T,N,D), person_mask: (B,T,N)
        valid_mask = person_mask.any(dim=1)
        mask_f = person_mask.to(dtype=tokens.dtype)

        frame_sum = (tokens * mask_f.unsqueeze(-1)).sum(dim=2)  # (B,T,D)
        frame_count = mask_f.sum(dim=2, keepdim=True).clamp(min=1.0)
        frame_mean = frame_sum / frame_count

        other_sum = frame_sum.unsqueeze(2) - tokens * mask_f.unsqueeze(-1)
        other_count = (frame_count.unsqueeze(2) - mask_f.unsqueeze(-1)).clamp(min=1.0)
        counterfactual_frame = other_sum / other_count

        attn_in = torch.cat([tokens, frame_mean.unsqueeze(2).expand_as(tokens)], dim=-1)
        attn_logits = self.temporal_attn(attn_in).squeeze(-1)
        attn = _masked_softmax(attn_logits, person_mask, dim=1)

        self_summary = (tokens * attn.unsqueeze(-1)).sum(dim=1)
        factual_event = (frame_mean.unsqueeze(2) * attn.unsqueeze(-1)).sum(dim=1)
        counterfactual_event = (counterfactual_frame * attn.unsqueeze(-1)).sum(dim=1)
        delta_raw = factual_event - counterfactual_event

        cf_input = torch.cat([self_summary, factual_event, counterfactual_event, delta_raw], dim=-1)
        cf_feat = self.adapter(cf_input, valid_mask)

        delta_refine = self.delta_refiner(cf_feat)
        delta = (delta_raw + delta_refine) * valid_mask.to(dtype=delta_raw.dtype).unsqueeze(-1)
        cf_logits = self.head(cf_feat).squeeze(-1).masked_fill(~valid_mask, -1e4)

        return {
            "counterfactual_features": cf_feat,
            "counterfactual_delta": delta,
            "counterfactual_delta_raw": delta_raw,
            "counterfactual_delta_refine": delta_refine,
            "counterfactual_logits": cf_logits,
            "counterfactual_attention": attn,
            "event_state": factual_event,
        }
