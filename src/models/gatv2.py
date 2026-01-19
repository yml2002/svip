"""GATv2 per-frame social modeling.

Hard requirement: `torch_geometric`.
Graph is built per (B,T): fully-connected among valid persons.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse

logger = logging.getLogger(__name__)


class GATv2Stack(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = bool(use_residual)
        self.in_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.layers.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )

        self.dropout = float(dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        person_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run GATv2 with spatial edges only (per-frame).

        Args:
            x: (B, T, N, D)
            person_mask: (B, T, N) bool
        """
        B, T, N, _ = x.shape
        h = self.in_proj(x)

        bt = B * T
        h_bt = h.reshape(bt, N, self.hidden_dim)
        m_bt = person_mask.reshape(bt, N)

        out_list = []
        for i in range(bt):
            mask = m_bt[i]
            if mask.sum() <= 1:
                out_list.append(h_bt[i])
                continue

            adj = torch.zeros(N, N, device=h_bt.device, dtype=torch.bool)
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            adj[idx[:, None], idx[None, :]] = True
            edge_index, _ = dense_to_sparse(adj)

            hi = h_bt[i]
            for li, layer in enumerate(self.layers):
                res = hi
                hi = layer(hi, edge_index)
                hi = F.elu(hi)
                hi = F.dropout(hi, p=self.dropout, training=self.training)
                hi = self.norms[li](hi)
                if self.use_residual:
                    hi = hi + res
                hi = hi.masked_fill(~mask.unsqueeze(-1), 0.0)
            out_list.append(hi)

        out = torch.stack(out_list, dim=0).reshape(B, T, N, self.hidden_dim)
        return out
