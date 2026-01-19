"""Video-level (per-person) temporal aggregation.

This module is adapted from `src0105/models/scoring_head.py`.
We use an attention pooling (query attends to per-person temporal tokens)
which is often better than simple mean pooling for "important person" tasks
because only a few key moments determine importance.

Input shape convention in this repo:
- temporal_features: (B, T, N, D)
- person_mask: (B, T, N) bool

Output:
- video_features: (B, N, out_dim)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VideoLevelAggregator(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        out_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_video_transformer: bool = False,
        pooling: str = "attention",
        transformer_layers: int = 2,
        transformer_ffn_dim: int = 512,
        transformer_dropout: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        self.use_video_transformer = bool(use_video_transformer)
        self.pooling = str(pooling)
        if self.use_video_transformer:
            td = float(transformer_dropout) if transformer_dropout is not None else self.dropout
            layer = nn.TransformerEncoderLayer(
                d_model=self.input_dim,
                nhead=self.num_heads,
                dim_feedforward=int(transformer_ffn_dim),
                dropout=td,
                batch_first=True,
                activation="gelu",
            )
            self.video_transformer = nn.TransformerEncoder(layer, num_layers=int(transformer_layers))
        else:
            self.video_transformer = None

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        self.aggregation_query = nn.Parameter(torch.randn(1, 1, self.input_dim))
        self.query_projection = nn.Linear(self.input_dim, self.input_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

    def forward(self, temporal_features: torch.Tensor, person_mask: torch.Tensor) -> torch.Tensor:
        if temporal_features.ndim != 4:
            raise ValueError("temporal_features must be (B,T,N,D)")
        if person_mask.ndim != 3:
            raise ValueError("person_mask must be (B,T,N)")

        B, T, N, D = temporal_features.shape
        if D != self.input_dim:
            raise ValueError(f"Last dim mismatch: got {D}, expected {self.input_dim}")

        flat_features = temporal_features.permute(0, 2, 1, 3).reshape(B * N, T, D)
        flat_mask = person_mask.permute(0, 2, 1).reshape(B * N, T)

        out = temporal_features.new_zeros((B * N, self.out_dim))

        valid_idx = torch.nonzero(flat_mask.any(dim=1), as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            return out.view(B, N, self.out_dim)

        feats_v = flat_features.index_select(0, valid_idx)
        mask_v = flat_mask.index_select(0, valid_idx)

        feats_v = self._apply_video_transformer(feats_v, mask_v)
        pooled = self._pooling(feats_v, mask_v)
        pooled = pooled.to(dtype=out.dtype)
        out.index_copy_(0, valid_idx, pooled)

        return out.view(B, N, self.out_dim)

    def _apply_video_transformer(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.video_transformer is None:
            return features
        key_padding_mask = ~mask.bool()
        return self.video_transformer(features, src_key_padding_mask=key_padding_mask)

    def _pooling(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # features: (BN,T,D), mask: (BN,T)
        key_padding_mask = ~mask.bool()

        if self.pooling == "mean":
            m = mask.to(dtype=features.dtype)
            valid_counts = m.sum(dim=1).clamp_min(1.0)
            summary = (features * m.unsqueeze(-1)).sum(dim=1) / valid_counts.unsqueeze(-1)
            pooled = self.output_projection(summary)
            return torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        
        m = mask.to(dtype=features.dtype)
        valid_counts = m.sum(dim=1).clamp_min(1.0)
        summary = (features * m.unsqueeze(-1)).sum(dim=1) / valid_counts.unsqueeze(-1)

        base_query = self.aggregation_query.expand(features.size(0), -1, -1)
        dynamic_query = self.query_projection(summary).unsqueeze(1)
        query = base_query + dynamic_query

        attended, _ = self.temporal_attention(query, features, features, key_padding_mask=key_padding_mask)
        combined = attended.squeeze(1) + summary
        pooled = self.output_projection(combined)
        return torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
