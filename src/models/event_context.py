"""Event-level context modules.

Provides a learnable event token that summarizes per-frame person tokens and
feeds global context back to each person token.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Lightweight TransformerEncoder over time (batch_first)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        key_padding = ~time_mask
        return self.encoder(x, src_key_padding_mask=key_padding)


class EventTokenContext(nn.Module):
    """Global event context via a learnable event token.

    For each frame t, an event token attends over all persons' tokens to produce
    a frame-level context. Then we run a temporal encoder over the event sequence
    and fuse the event context back into each person's token.

    Args:
        d_model: token dim
        nhead: attention heads
        dim_feedforward: FFN dim for the event temporal encoder
        dropout: dropout
        event_num_layers: transformer layers for event temporal encoder
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        event_num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.event_token = nn.Parameter(torch.zeros(1, 1, int(d_model)))
        self.event_attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(nhead),
            dropout=float(dropout),
            batch_first=True,
        )
        self.event_temporal = TemporalEncoder(
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(event_num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
        )
        self.gate = nn.Linear(int(d_model) * 2, 1)

    def forward(self, tokens: torch.Tensor, pm: torch.Tensor) -> torch.Tensor:
        """Fuse event context into per-person tokens.

        Args:
            tokens: (B,T,N,D)
            pm: (B,T,N) bool

        Returns:
            (B,T,N,D)
        """
        B, T, N, D = tokens.shape

        persons_bt = tokens.reshape(B * T, N, D)
        mask_bt = pm.reshape(B * T, N)

        q = self.event_token.expand(B * T, 1, D)
        event_bt, _ = self.event_attn(
            q,
            persons_bt,
            persons_bt,
            key_padding_mask=~mask_bt,
            need_weights=False,
        )
        event = event_bt.reshape(B, T, D)

        event_mask = pm.any(dim=2)  # (B,T)
        event = self.event_temporal(event, event_mask)

        event_expand = event.unsqueeze(2).expand(B, T, N, D)
        g = torch.sigmoid(self.gate(torch.cat([tokens, event_expand], dim=-1)))  # (B,T,N,1)
        return (1.0 - g) * tokens + g * event_expand
