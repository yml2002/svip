"""Keyframe-Conditioned Global Context (KCGC).

Provides scene-level background knowledge to the relation branch before GAT
processes person interactions.  DINOv2 CLS tokens from K uniformly-sampled
full frames encode the overall scene type (wedding, press conf, classroom…),
giving GAT a scene-conditioned starting point for modelling interactions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GlobalContextModule(nn.Module):
    """Keyframe-Conditioned Global Context (KCGC).

    Flow:
      1. Uniformly sample K keyframe indices from T sampled frames
         (K = T//4, gap >= 4 frames so keyframes differ meaningfully)
      2. Encode each full frame with shared DINOv2 backbone (no_grad) → CLS token
      3. Project K CLS tokens → scene_ctx (B, K, context_dim)
      4. Each person cross-attends to scene_ctx
      5. Residual + LayerNorm → fused_enhanced
    """

    def __init__(
        self,
        fused_dim: int,
        num_keyframes: int,
        context_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_keyframes = num_keyframes
        self.kf_proj = nn.Linear(768, context_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        fused: torch.Tensor,
        frames: torch.Tensor,
        vision: nn.Module,
        person_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, N, D = fused.shape
        device = fused.device

        kf_indices = torch.linspace(0, T - 1, steps=self.num_keyframes,
                                    device=device).long()
        kf_frames = frames[:, kf_indices]
        K = kf_frames.shape[1]
        H, W = kf_frames.shape[3], kf_frames.shape[4]

        kf_flat = kf_frames.reshape(B * K, 3, H, W)
        with torch.no_grad():
            kf_cls = vision(kf_flat)                    # (B*K, 768)
        kf_ctx = self.kf_proj(kf_cls.reshape(B, K, 768))  # (B, K, context_dim)

        fused_flat = fused.reshape(B, T * N, D)
        attn_out, _ = self.cross_attn(fused_flat, kf_ctx, kf_ctx)
        enhanced = self.norm(fused_flat + self.dropout(attn_out)).reshape(B, T, N, D)
        return enhanced.masked_fill(~person_mask.unsqueeze(-1), 0.0)
