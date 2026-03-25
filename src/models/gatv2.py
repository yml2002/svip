"""Spatio-temporal GATv2 graph network.

Builds a joint graph where each node is (person_i, frame_t).
Two edge types:
  - Spatial  (intra-frame): between persons in the same frame
  - Temporal (inter-frame): same person across adjacent frames

Spatial edge features: bbox geometry (delta_cx, delta_cy, delta_w, delta_h, dist, iou).
Temporal edge features: vis_feats difference vector projected via two-layer MLP with
  LayerNorm, capturing how a person's DINOv2 appearance shifts between sampled frames.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv

from src.models.bbox_geom import compute_spatial_edge_features_from_bboxes

logger = logging.getLogger(__name__)

SPATIAL_EDGE_RAW_DIM  = 6  # delta_cx, delta_cy, delta_w, delta_h, center_dist, iou
TEMPORAL_EDGE_RAW_DIM = 768  # vis_feats difference vector (DINOv2 dim)


class SpatioTemporalGATv2(nn.Module):
    """GATv2 on a joint spatio-temporal person graph."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        topk_neighbors: int = 8,
        temporal_window: int = 2,
        spatial_edge_dim: int = 32,
        temporal_edge_dim: int = 32,
        vis_feat_dim: int = 768,
        use_spatial_edges: bool = True,
        use_temporal_edges: bool = True,
        use_edge_features: bool = True,
        graph_type: str = "gatv2",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.topk_neighbors = int(max(0, topk_neighbors))
        self.temporal_window = int(max(1, temporal_window))
        self.dropout = float(dropout)
        self.use_spatial_edges = bool(use_spatial_edges)
        self.use_temporal_edges = bool(use_temporal_edges)
        self.use_edge_features = bool(use_edge_features)
        self.graph_type = str(graph_type)

        self.in_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        edge_dim = int(spatial_edge_dim) if (self.use_edge_features and self.graph_type == "gatv2") else None
        if self.use_edge_features and self.graph_type == "gatv2":
            self.spatial_edge_proj = nn.Sequential(
                nn.Linear(SPATIAL_EDGE_RAW_DIM, int(spatial_edge_dim)),
                nn.ReLU(inplace=True),
            )
            # Temporal: vis_feats diff (768-dim) → spatial_edge_dim via bottleneck
            self.temporal_edge_proj = nn.Sequential(
                nn.LayerNorm(int(vis_feat_dim)),
                nn.Linear(int(vis_feat_dim), int(temporal_edge_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(temporal_edge_dim), int(spatial_edge_dim)),
                nn.ReLU(inplace=True),
            )
        else:
            self.spatial_edge_proj = None
            self.temporal_edge_proj = None

        self.layers = nn.ModuleList()
        # GCN: reduced hidden_dim as a genuinely weak baseline
        gcn_dim = max(64, hidden_dim // 4)
        self.gcn_out_proj = nn.Linear(gcn_dim, hidden_dim) if self.graph_type == "gcn" else None
        for _ in range(self.num_layers):
            if self.graph_type == "gcn":
                self.layers.append(GCNConv(hidden_dim, gcn_dim, add_self_loops=False))
            else:
                self.layers.append(
                    GATv2Conv(
                        hidden_dim,
                        hidden_dim // heads,
                        heads=heads,
                        dropout=dropout,
                        add_self_loops=True,
                        edge_dim=edge_dim,
                    )
                )

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        person_mask: torch.Tensor,
        bboxes: torch.Tensor,
        vis_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, N, _ = x.shape
        device = x.device

        h = self.in_proj(x)

        valid = person_mask.bool()
        valid_flat = valid.reshape(-1)
        node_indices = torch.full((B * T * N,), -1, dtype=torch.long, device=device)
        valid_positions = valid_flat.nonzero(as_tuple=False).squeeze(1)
        num_nodes = int(valid_positions.numel())

        if num_nodes == 0:
            return h.new_zeros((B, T, N, self.hidden_dim))

        node_indices[valid_positions] = torch.arange(num_nodes, device=device)
        h_flat = h.reshape(B * T * N, -1)
        all_x = h_flat[valid_positions]

        bboxes_flat = bboxes.reshape(B * T * N, 4).to(dtype=torch.float32)
        node_bboxes = bboxes_flat[valid_positions]

        node_n = valid_positions % N
        node_t = (valid_positions // N) % T
        node_b = valid_positions // (T * N)

        if self.use_spatial_edges:
            spatial_src, spatial_dst = self._build_spatial_edges(
                node_b, node_t, node_n, node_bboxes, num_nodes, device
            )
        else:
            spatial_src = torch.empty(0, dtype=torch.long, device=device)
            spatial_dst = torch.empty(0, dtype=torch.long, device=device)

        if self.use_temporal_edges:
            temporal_src, temporal_dst = self._build_temporal_edges(
                node_b, node_t, node_n, node_indices, B, T, N, device
            )
        else:
            temporal_src = torch.empty(0, dtype=torch.long, device=device)
            temporal_dst = torch.empty(0, dtype=torch.long, device=device)

        all_edge_src = []
        all_edge_dst = []
        all_edge_feat = []

        if spatial_src.numel() > 0:
            all_edge_src.append(spatial_src)
            all_edge_dst.append(spatial_dst)
            if self.spatial_edge_proj is not None:
                s_feat_raw = compute_spatial_edge_features_from_bboxes(
                    node_bboxes[spatial_src], node_bboxes[spatial_dst]
                )
                all_edge_feat.append(self.spatial_edge_proj(s_feat_raw))

        if temporal_src.numel() > 0:
            all_edge_src.append(temporal_src)
            all_edge_dst.append(temporal_dst)
            if self.temporal_edge_proj is not None:
                assert vis_feats is not None
                vis_flat = vis_feats.reshape(B * T * N, -1).to(dtype=torch.float32)
                node_vis = vis_flat[valid_positions]
                t_feat_raw = node_vis[temporal_dst] - node_vis[temporal_src]
                all_edge_feat.append(self.temporal_edge_proj(t_feat_raw))

        if all_edge_src:
            edge_index = torch.stack([
                torch.cat(all_edge_src),
                torch.cat(all_edge_dst),
            ], dim=0)
            edge_attr = torch.cat(all_edge_feat, dim=0) if all_edge_feat else None
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = None

        hi = all_x
        for li, layer in enumerate(self.layers):
            residual = hi
            if self.graph_type == "gcn":
                hi = self.gcn_out_proj(layer(hi, edge_index))
            else:
                hi = layer(hi, edge_index, edge_attr=edge_attr)
            hi = F.elu(hi)
            hi = F.dropout(hi, p=self.dropout, training=self.training)
            hi = self.norms[li](hi + residual)

        out = h.new_zeros((B * T * N, self.hidden_dim))
        out[valid_positions] = hi.to(dtype=out.dtype)
        return out.reshape(B, T, N, self.hidden_dim)

    def _build_spatial_edges(self, node_b, node_t, node_n, node_bboxes, num_nodes, device):
        frame_key = node_b * 10000 + node_t
        unique_frames = frame_key.unique()
        src_list, dst_list = [], []

        for fk in unique_frames:
            frame_nodes = (frame_key == fk).nonzero(as_tuple=False).squeeze(1)
            n_valid = int(frame_nodes.numel())
            if n_valid <= 1:
                continue
            if self.topk_neighbors <= 0 or n_valid <= self.topk_neighbors:
                arange = torch.arange(n_valid, device=device)
                s = arange.repeat_interleave(n_valid)
                d = arange.repeat(n_valid)
                keep = s != d
                src_list.append(frame_nodes[s[keep]])
                dst_list.append(frame_nodes[d[keep]])
            else:
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

    def _build_temporal_edges(self, node_b, node_t, node_n, node_indices, B, T, N, device):
        src_list, dst_list = [], []
        num_nodes = int(node_b.numel())
        curr_idx = torch.arange(num_nodes, device=device)

        for dt in range(1, self.temporal_window + 1):
            future_t = node_t + dt
            valid_future = future_t < T
            if not valid_future.any():
                continue
            future_flat = node_b * (T * N) + future_t * N + node_n
            future_flat_clamped = future_flat.clamp(0, B * T * N - 1)
            future_node_idx = node_indices[future_flat_clamped]
            both_valid = valid_future & (future_node_idx >= 0)
            if both_valid.any():
                src_list.append(curr_idx[both_valid])
                dst_list.append(future_node_idx[both_valid])
                src_list.append(future_node_idx[both_valid])
                dst_list.append(curr_idx[both_valid])

        if src_list:
            return torch.cat(src_list), torch.cat(dst_list)
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
