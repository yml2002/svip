"""Open-world scene context module.

Instead of classifying videos into a fixed list of scenes, this module learns a
latent scene context from keyframes and mixes it with a bank of learnable scene
prototypes. The output is a continuous scene token used only as conditioning.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class ScenePriorModule(nn.Module):
    def __init__(
        self,
        *,
        num_keyframes: int,
        context_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
        num_prototypes: int,
    ) -> None:
        super().__init__()
        self.num_keyframes = int(num_keyframes)
        self.context_dim = int(context_dim)
        self.frame_proj = nn.Linear(768, self.context_dim)
        self.position = nn.Parameter(torch.randn(1, self.num_keyframes, self.context_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.context_dim,
            nhead=int(num_heads),
            dim_feedforward=self.context_dim * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.query = nn.Parameter(torch.randn(1, 1, self.context_dim) * 0.02)
        self.pool = nn.MultiheadAttention(
            embed_dim=self.context_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.prototype_bank = nn.Parameter(torch.randn(int(num_prototypes), self.context_dim) * 0.02)
        self.prototype_gate = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, self.context_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.out_norm = nn.LayerNorm(self.context_dim)

    def forward(self, frames: torch.Tensor, vision: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, total_t = int(frames.shape[0]), int(frames.shape[1])
        device = frames.device
        key_idx = torch.linspace(0, total_t - 1, steps=self.num_keyframes, device=device).long()
        keyframes = frames[:, key_idx]
        _, num_kf, _, h, w = keyframes.shape

        with torch.no_grad():
            key_tokens = vision(keyframes.reshape(bsz * num_kf, 3, h, w))

        key_tokens = self.frame_proj(key_tokens.reshape(bsz, num_kf, -1))
        key_tokens = self.encoder(key_tokens + self.position[:, :num_kf])
        query = self.query.expand(bsz, -1, -1)
        pooled, _ = self.pool(query, key_tokens, key_tokens)
        scene_seed = pooled.squeeze(1)

        proto_logits = torch.matmul(self.prototype_gate(scene_seed), self.prototype_bank.t()) / math.sqrt(float(self.context_dim))
        proto_weight = torch.softmax(proto_logits, dim=-1)
        proto_context = torch.matmul(proto_weight, self.prototype_bank)
        scene_token = self.out_norm(scene_seed + proto_context)
        key_proto_logits = torch.matmul(key_tokens, self.prototype_bank.t()) / math.sqrt(float(self.context_dim))
        return scene_token, proto_logits, key_proto_logits
