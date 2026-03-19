"""GATv2 per-frame social modeling.

Hard requirement: `torch_geometric`.
Graph is built per (B,T) with configurable sparsity/topology.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv

logger = logging.getLogger(__name__)


class GATv2Stack(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        topk_neighbors: int = 4,
    ) -> None:
        super().__init__()
        self.topk_neighbors = int(max(0, topk_neighbors))
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
                    add_self_loops=False,
                )
            )

        self.dropout = float(dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        person_mask: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run GATv2 with per-frame interaction edges.

        Args:
            x: (B, T, N, D)
            person_mask: (B, T, N) bool
            bboxes: optional (B, T, N, 4) normalized boxes for top-k graph
        """
        B, T, N, _ = x.shape
        h = self.in_proj(x)

        bt = B * T
        h_bt = h.reshape(bt, N, self.hidden_dim)
        m_bt = person_mask.reshape(bt, N)
        centers_bt = None
        if bboxes is not None:
            centers = torch.stack(
                [
                    (bboxes[..., 0] + bboxes[..., 2]) * 0.5,
                    (bboxes[..., 1] + bboxes[..., 3]) * 0.5,
                ],
                dim=-1,
            )
            centers_bt = centers.reshape(bt, N, 2)

        out_list = []
        for i in range(bt):
            mask = m_bt[i]
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            num_valid = int(idx.numel())
            if num_valid <= 1:
                out_list.append(h_bt[i].new_zeros((N, self.hidden_dim)))
                continue

            edge_index = self._build_edge_index(
                idx=idx,
                num_nodes=N,
                centers=centers_bt[i] if centers_bt is not None else None,
                device=h_bt.device,
            )
            if edge_index.numel() == 0:
                out_list.append(h_bt[i].new_zeros((N, self.hidden_dim)))
                continue

            hi = h_bt[i]
            for li, layer in enumerate(self.layers):
                hi = layer(hi, edge_index)
                hi = F.elu(hi)
                hi = F.dropout(hi, p=self.dropout, training=self.training)
                hi = self.norms[li](hi)
                hi = hi.masked_fill(~mask.unsqueeze(-1), 0.0)
            out_list.append(hi)

        out = torch.stack(out_list, dim=0).reshape(B, T, N, self.hidden_dim)
        return out

    def _build_edge_index(
        self,
        *,
        idx: torch.Tensor,
        num_nodes: int,
        centers: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        num_valid = int(idx.numel())
        if num_valid <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        if centers is None or self.topk_neighbors <= 0 or num_valid <= self.topk_neighbors:
            src = idx.repeat_interleave(num_valid)
            dst = idx.repeat(num_valid)
            keep = src != dst
            src = src[keep]
            dst = dst[keep]
            if src.numel() == 0:
                return torch.empty((2, 0), dtype=torch.long, device=device)
            return torch.stack([src, dst], dim=0)

        pos = centers.index_select(0, idx).to(dtype=torch.float32)
        dist = torch.cdist(pos, pos, p=2)
        dist.fill_diagonal_(float("inf"))
        k = min(self.topk_neighbors, num_valid - 1)
        if k <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        nn_local = torch.topk(dist, k=k, dim=1, largest=False).indices
        src = idx.repeat_interleave(k)
        dst = idx.index_select(0, nn_local.reshape(-1))
        return torch.stack([src, dst], dim=0)
