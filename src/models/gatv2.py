"""Spatio-temporal GATv2 graph network.

Builds a joint graph where each node is (person_i, frame_t).
Two edge types:
  - Spatial (intra-frame): between persons in the same frame
  - Temporal (inter-frame): same person across adjacent frames

Edge features are computed from bbox geometry and projected into the
attention mechanism via GATv2Conv's edge_attr support.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from src.models.bbox_geom import (
    compute_spatial_edge_features_from_bboxes,
    compute_temporal_edge_features_from_bboxes,
)

logger = logging.getLogger(__name__)

SPATIAL_EDGE_RAW_DIM = 6   # delta_cx, delta_cy, delta_w, delta_h, center_dist, iou
TEMPORAL_EDGE_RAW_DIM = 4  # delta_cx, delta_cy, delta_area, speed


class SpatioTemporalGATv2(nn.Module):
    """GATv2 on a joint spatio-temporal person graph.

    Args:
        in_dim: input node feature dimension
        hidden_dim: hidden dimension for GAT layers
        num_layers: number of GAT layers
        heads: number of attention heads per layer
        dropout: dropout rate
        topk_neighbors: for spatial edges, top-k nearest by bbox center distance (0 = all)
        temporal_window: connect same person across ±window frames (1 = adjacent only)
        spatial_edge_dim: projected spatial edge feature dim
        temporal_edge_dim: projected temporal edge feature dim
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        topk_neighbors: int = 4,
        temporal_window: int = 1,
        spatial_edge_dim: int = 32,
        temporal_edge_dim: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.topk_neighbors = int(max(0, topk_neighbors))
        self.temporal_window = int(max(1, temporal_window))
        self.dropout = float(dropout)

        self.in_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        # Edge feature projections: both types → same edge_dim for GATv2
        edge_dim = int(spatial_edge_dim)
        self.spatial_edge_proj = nn.Sequential(
            nn.Linear(SPATIAL_EDGE_RAW_DIM, edge_dim),
            nn.ReLU(inplace=True),
        )
        self.temporal_edge_proj = nn.Sequential(
            nn.Linear(TEMPORAL_EDGE_RAW_DIM, edge_dim),
            nn.ReLU(inplace=True),
        )

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=False,
                    edge_dim=edge_dim,
                )
            )

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        person_mask: torch.Tensor,
        bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, N, D) node features
            person_mask: (B, T, N) bool
            bboxes: (B, T, N, 4) normalized

        Returns:
            (B, T, N, hidden_dim) updated node features
        """
        B, T, N, _ = x.shape
        device = x.device

        h = self.in_proj(x)

        # Build the joint spatio-temporal graph for the entire batch.
        # Each valid (b, t, n) becomes a node.
        # We map (b, t, n) → flat node index.
        valid = person_mask.bool()  # (B, T, N)
        valid_flat = valid.reshape(-1)  # (B*T*N,)

        # Create mapping: flat index → node index (only for valid slots)
        node_indices = torch.full((B * T * N,), -1, dtype=torch.long, device=device)
        valid_positions = valid_flat.nonzero(as_tuple=False).squeeze(1)
        num_nodes = int(valid_positions.numel())

        if num_nodes == 0:
            return h.new_zeros((B, T, N, self.hidden_dim))

        node_indices[valid_positions] = torch.arange(num_nodes, device=device)

        # Gather node features
        h_flat = h.reshape(B * T * N, -1)
        all_x = h_flat[valid_positions]  # (num_nodes, hidden_dim)

        # Gather bboxes for valid nodes
        bboxes_flat = bboxes.reshape(B * T * N, 4).to(dtype=torch.float32)
        node_bboxes = bboxes_flat[valid_positions]  # (num_nodes, 4)

        # Decode (b, t, n) from valid_positions
        node_n = valid_positions % N
        node_t = (valid_positions // N) % T
        node_b = valid_positions // (T * N)

        # --- Build spatial edges (intra-frame) ---
        spatial_src, spatial_dst = self._build_spatial_edges(
            node_b, node_t, node_n, node_bboxes, num_nodes, device
        )

        # --- Build temporal edges (inter-frame, same person) ---
        temporal_src, temporal_dst = self._build_temporal_edges(
            node_b, node_t, node_n, node_indices, B, T, N, device
        )

        # --- Compute edge features ---
        all_edge_src = []
        all_edge_dst = []
        all_edge_feat = []

        if spatial_src.numel() > 0:
            s_feat_raw = compute_spatial_edge_features_from_bboxes(
                node_bboxes[spatial_src], node_bboxes[spatial_dst]
            )
            s_feat = self.spatial_edge_proj(s_feat_raw)
            all_edge_src.append(spatial_src)
            all_edge_dst.append(spatial_dst)
            all_edge_feat.append(s_feat)

        if temporal_src.numel() > 0:
            t_feat_raw = compute_temporal_edge_features_from_bboxes(
                node_bboxes[temporal_src], node_bboxes[temporal_dst]
            )
            t_feat = self.temporal_edge_proj(t_feat_raw)
            all_edge_src.append(temporal_src)
            all_edge_dst.append(temporal_dst)
            all_edge_feat.append(t_feat)

        if all_edge_src:
            edge_index = torch.stack([
                torch.cat(all_edge_src),
                torch.cat(all_edge_dst),
            ], dim=0)
            edge_attr = torch.cat(all_edge_feat, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0, self.spatial_edge_proj[0].out_features), device=device)

        # --- GATv2 message passing ---
        hi = all_x
        for li, layer in enumerate(self.layers):
            residual = hi
            hi = layer(hi, edge_index, edge_attr=edge_attr)
            hi = F.elu(hi)
            hi = F.dropout(hi, p=self.dropout, training=self.training)
            hi = self.norms[li](hi + residual)

        # Scatter back to (B, T, N, hidden_dim)
        out = h.new_zeros((B * T * N, self.hidden_dim))
        out[valid_positions] = hi.to(dtype=out.dtype)
        return out.reshape(B, T, N, self.hidden_dim)

    def _build_spatial_edges(
        self,
        node_b: torch.Tensor,
        node_t: torch.Tensor,
        node_n: torch.Tensor,
        node_bboxes: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build intra-frame spatial edges between persons."""
        # Group nodes by (b, t)
        frame_key = node_b * 10000 + node_t  # unique per (b, t)
        unique_frames = frame_key.unique()

        src_list = []
        dst_list = []

        for fk in unique_frames:
            frame_nodes = (frame_key == fk).nonzero(as_tuple=False).squeeze(1)
            n_valid = int(frame_nodes.numel())
            if n_valid <= 1:
                continue

            if self.topk_neighbors <= 0 or n_valid <= self.topk_neighbors:
                # Fully connected
                arange = torch.arange(n_valid, device=device)
                s = arange.repeat_interleave(n_valid)
                d = arange.repeat(n_valid)
                keep = s != d
                src_list.append(frame_nodes[s[keep]])
                dst_list.append(frame_nodes[d[keep]])
            else:
                # Top-k by bbox center distance
                fb = node_bboxes[frame_nodes]
                cx = (fb[:, 0] + fb[:, 2]) * 0.5
                cy = (fb[:, 1] + fb[:, 3]) * 0.5
                pos = torch.stack([cx, cy], dim=-1)
                dist = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)
                dist.fill_diagonal_(float("inf"))
                k = min(self.topk_neighbors, n_valid - 1)
                nn_idx = dist.topk(k=k, dim=1, largest=False).indices
                arange = torch.arange(n_valid, device=device)
                s = arange.repeat_interleave(k)
                d = nn_idx.reshape(-1)
                src_list.append(frame_nodes[s])
                dst_list.append(frame_nodes[d])

        if src_list:
            return torch.cat(src_list), torch.cat(dst_list)
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

    def _build_temporal_edges(
        self,
        node_b: torch.Tensor,
        node_t: torch.Tensor,
        node_n: torch.Tensor,
        node_indices: torch.Tensor,
        B: int,
        T: int,
        N: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build inter-frame temporal edges (same person across frames)."""
        src_list = []
        dst_list = []

        num_nodes = int(node_b.numel())
        for dt in range(1, self.temporal_window + 1):
            # For each node, try to connect to same person dt frames later
            future_t = node_t + dt
            valid_future = future_t < T

            if not valid_future.any():
                continue

            # Compute flat index for future node
            future_flat = node_b * (T * N) + future_t * N + node_n
            # Clamp to valid range for indexing
            future_flat_clamped = future_flat.clamp(0, B * T * N - 1)
            future_node_idx = node_indices[future_flat_clamped]

            # Both current and future must be valid nodes
            both_valid = valid_future & (future_node_idx >= 0)

            if both_valid.any():
                curr_node_idx = torch.arange(num_nodes, device=device)
                valid_curr = curr_node_idx[both_valid]
                valid_future_nodes = future_node_idx[both_valid]

                # Bidirectional
                src_list.append(valid_curr)
                dst_list.append(valid_future_nodes)
                src_list.append(valid_future_nodes)
                dst_list.append(valid_curr)

        if src_list:
            return torch.cat(src_list), torch.cat(dst_list)
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
