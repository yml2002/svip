"""Temporal social memory relation reasoning.

This branch refines person-centric importance with history-aware social cues.
At each frame, pairwise spatial relations are built from current features and
the previous social memory state, so historical interaction patterns can affect
how the current frame is interpreted.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def _bbox_pair_features(bboxes: torch.Tensor) -> torch.Tensor:
    src = bboxes[:, :, None, :]
    dst = bboxes[:, None, :, :]

    sx1, sy1, sx2, sy2 = src.unbind(dim=-1)
    dx1, dy1, dx2, dy2 = dst.unbind(dim=-1)

    scx = (sx1 + sx2) * 0.5
    scy = (sy1 + sy2) * 0.5
    dcx = (dx1 + dx2) * 0.5
    dcy = (dy1 + dy2) * 0.5
    sw = (sx2 - sx1).clamp(min=1e-6)
    sh = (sy2 - sy1).clamp(min=1e-6)
    dw = (dx2 - dx1).clamp(min=1e-6)
    dh = (dy2 - dy1).clamp(min=1e-6)

    inter_x1 = torch.maximum(sx1, dx1)
    inter_y1 = torch.maximum(sy1, dy1)
    inter_x2 = torch.minimum(sx2, dx2)
    inter_y2 = torch.minimum(sy2, dy2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter = inter_w * inter_h
    union = sw * sh + dw * dh - inter
    iou = inter / union.clamp(min=1e-6)

    dx = dcx - scx
    dy = dcy - scy
    dist = torch.sqrt(dx.square() + dy.square())
    area_ratio = ((dw * dh) / (sw * sh)).clamp(min=1e-2, max=1e2)
    log_scale = torch.log(area_ratio)
    pair = torch.stack([dx, dy, dist, iou, dw - sw, log_scale], dim=-1)
    pair = torch.nan_to_num(pair, nan=0.0, posinf=4.0, neginf=-4.0)
    pair[..., 0] = pair[..., 0].clamp(min=-1.0, max=1.0)
    pair[..., 1] = pair[..., 1].clamp(min=-1.0, max=1.0)
    pair[..., 2] = pair[..., 2].clamp(min=0.0, max=1.5)
    pair[..., 3] = pair[..., 3].clamp(min=0.0, max=1.0)
    pair[..., 4] = pair[..., 4].clamp(min=-1.0, max=1.0)
    pair[..., 5] = pair[..., 5].clamp(min=-4.0, max=4.0)
    return pair


class SocialMemoryLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        dropout: float,
        topk_neighbors: int,
        graph_type: str,
        use_edge_features: bool,
        scene_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = max(1, int(heads))
        self.head_dim = int(hidden_dim) // self.heads
        if self.heads * self.head_dim != int(hidden_dim):
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by heads={heads}")
        self.topk_neighbors = int(max(0, topk_neighbors))
        self.graph_type = str(graph_type)
        self.use_edge_features = bool(use_edge_features)

        self.pre_norm = nn.LayerNorm(self.hidden_dim)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.self_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scene_q = nn.Linear(int(scene_dim), self.hidden_dim)
        self.scene_v = nn.Linear(int(scene_dim), self.hidden_dim)
        self.edge_bias = nn.Linear(6, self.heads) if self.use_edge_features else None
        self.stats_proj = nn.Linear(4, self.hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 5),
            nn.Linear(self.hidden_dim * 5, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.memory_cell = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.dropout = nn.Dropout(float(dropout))

    def _neighbor_mask(self, bboxes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        centers = (bboxes[..., :2] + bboxes[..., 2:]) * 0.5
        dist = torch.cdist(centers, centers)
        n_person = int(bboxes.shape[1])
        eye = torch.eye(n_person, device=bboxes.device, dtype=torch.bool).unsqueeze(0)
        valid_pairs = valid[:, :, None] & valid[:, None, :] & ~eye

        if self.topk_neighbors <= 0 or n_person <= self.topk_neighbors:
            return valid_pairs

        masked_dist = dist.masked_fill(~valid_pairs, float("inf"))
        k = min(self.topk_neighbors, max(1, n_person - 1))
        nn_idx = masked_dist.topk(k=k, dim=-1, largest=False).indices
        topk_mask = torch.zeros_like(valid_pairs)
        topk_mask.scatter_(dim=-1, index=nn_idx, value=True)
        return topk_mask & valid_pairs

    def forward(
        self,
        x_t: torch.Tensor,
        prev_memory: torch.Tensor,
        bboxes_t: torch.Tensor,
        valid_t: torch.Tensor,
        scene_token: torch.Tensor,
        *,
        use_temporal_memory: bool,
        use_spatial_edges: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n_person, _ = x_t.shape
        memory_in = prev_memory if use_temporal_memory else torch.zeros_like(prev_memory)
        node = self.pre_norm(x_t + memory_in)

        if not valid_t.any():
            zeros = torch.zeros_like(node)
            stats = node.new_zeros((bsz, n_person, 4))
            return zeros, zeros, stats

        neighbor_mask = self._neighbor_mask(bboxes_t, valid_t)
        if not use_spatial_edges or not neighbor_mask.any():
            edge_gate = node.new_zeros((bsz, self.heads, n_person, n_person))
            incoming_msg = torch.zeros_like(node)
            outgoing_msg = torch.zeros_like(node)
            stats = node.new_zeros((bsz, n_person, 4))
        else:
            q = self.q_proj(node).view(bsz, n_person, self.heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k_proj(node).view(bsz, n_person, self.heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_proj(node).view(bsz, n_person, self.heads, self.head_dim).permute(0, 2, 1, 3)
            scene_shift = self.scene_q(scene_token).view(bsz, self.heads, self.head_dim).unsqueeze(2)
            q = q + scene_shift

            if self.graph_type == "gcn":
                edge_gate = neighbor_mask.unsqueeze(1).to(dtype=node.dtype)
            else:
                scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(float(self.head_dim))
                if self.edge_bias is not None:
                    pair_feat = _bbox_pair_features(bboxes_t).to(dtype=torch.float32)
                    pair_bias = self.edge_bias(pair_feat).to(dtype=scores.dtype).permute(0, 3, 1, 2)
                    scores = scores + pair_bias
                scores = scores.masked_fill(~neighbor_mask.unsqueeze(1), -20.0)
                edge_gate = torch.sigmoid(scores) * neighbor_mask.unsqueeze(1).to(dtype=scores.dtype)

            recv_norm = edge_gate / edge_gate.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            send_norm = edge_gate.transpose(-1, -2) / edge_gate.transpose(-1, -2).sum(dim=-1, keepdim=True).clamp(min=1e-6)

            incoming_msg = torch.einsum("bhij,bhjd->bhid", recv_norm, v)
            incoming_msg = incoming_msg.permute(0, 2, 1, 3).reshape(bsz, n_person, self.hidden_dim)
            outgoing_msg = torch.einsum("bhij,bhjd->bhid", send_norm, v)
            outgoing_msg = outgoing_msg.permute(0, 2, 1, 3).reshape(bsz, n_person, self.hidden_dim)

            gate_mean = edge_gate.mean(dim=1)
            deg = neighbor_mask.to(dtype=node.dtype).sum(dim=-1).clamp(min=1.0)
            incoming_strength = gate_mean.sum(dim=-2) / deg
            outgoing_strength = gate_mean.sum(dim=-1) / deg
            reciprocal = (gate_mean * gate_mean.transpose(1, 2)).sum(dim=-1) / deg
            focality = incoming_strength - outgoing_strength
            stats = torch.stack([incoming_strength, outgoing_strength, reciprocal, focality], dim=-1)

        scene_value = self.scene_v(scene_token).unsqueeze(1).expand(-1, n_person, -1)
        update_input = torch.cat(
            [
                self.self_proj(node),
                incoming_msg,
                outgoing_msg,
                self.stats_proj(stats),
                scene_value,
            ],
            dim=-1,
        )
        update_hidden = self.update_mlp(update_input)
        next_memory = self.memory_cell(update_hidden.reshape(bsz * n_person, self.hidden_dim), memory_in.reshape(bsz * n_person, self.hidden_dim))
        next_memory = next_memory.reshape(bsz, n_person, self.hidden_dim)
        out = self.out_norm(next_memory + self.dropout(self.ffn(next_memory)))
        out = out.masked_fill(~valid_t.unsqueeze(-1), 0.0)
        next_memory = next_memory.masked_fill(~valid_t.unsqueeze(-1), 0.0)
        stats = stats.masked_fill(~valid_t.unsqueeze(-1), 0.0)
        return out, next_memory, stats


class TemporalSocialMemoryEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
        topk_neighbors: int,
        use_spatial_edges: bool,
        use_temporal_edges: bool,
        use_edge_features: bool,
        graph_type: str,
        scene_dim: int,
    ) -> None:
        super().__init__()
        self.use_spatial_edges = bool(use_spatial_edges)
        self.use_temporal_edges = bool(use_temporal_edges)
        self.out_dim = int(hidden_dim)
        self.stat_dim = 4

        self.in_proj = nn.Linear(int(in_dim), int(hidden_dim))
        self.layers = nn.ModuleList([
            SocialMemoryLayer(
                hidden_dim=int(hidden_dim),
                heads=int(heads),
                dropout=float(dropout),
                topk_neighbors=int(topk_neighbors),
                graph_type=str(graph_type),
                use_edge_features=bool(use_edge_features),
                scene_dim=int(scene_dim),
            )
            for _ in range(int(num_layers))
        ])
        self.temporal_attn = nn.Sequential(
            nn.Linear(int(hidden_dim) + self.stat_dim + int(scene_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, 1),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) + self.stat_dim + int(scene_dim)),
            nn.Linear(int(hidden_dim) + self.stat_dim + int(scene_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        fused: torch.Tensor,
        person_mask: torch.Tensor,
        bboxes: torch.Tensor,
        scene_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, total_t, n_person, _ = fused.shape
        h = self.in_proj(fused).masked_fill(~person_mask.unsqueeze(-1), 0.0)
        stats_seq = h.new_zeros((bsz, total_t, n_person, self.stat_dim))

        for layer in self.layers:
            memory = h.new_zeros((bsz, n_person, self.out_dim))
            outputs = []
            layer_stats = []
            for t in range(total_t):
                out_t, memory, stats_t = layer(
                    h[:, t],
                    memory,
                    bboxes[:, t].to(dtype=h.dtype),
                    person_mask[:, t],
                    scene_token.to(dtype=h.dtype),
                    use_temporal_memory=self.use_temporal_edges,
                    use_spatial_edges=self.use_spatial_edges,
                )
                outputs.append(out_t)
                layer_stats.append(stats_t)
            h = torch.stack(outputs, dim=1)
            stats_seq = stats_seq + torch.stack(layer_stats, dim=1)

        stats_seq = stats_seq / max(1, len(self.layers))
        scene_expand = scene_token[:, None, None, :].expand(bsz, total_t, n_person, -1).to(dtype=h.dtype)
        attn_input = torch.cat([h, stats_seq.to(dtype=h.dtype), scene_expand], dim=-1)
        attn_logits = self.temporal_attn(attn_input).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~person_mask, -1e4)
        attn = torch.softmax(attn_logits.permute(0, 2, 1), dim=-1)
        attn = attn * person_mask.permute(0, 2, 1).to(dtype=attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        pooled_feat = (h.permute(0, 2, 1, 3) * attn.unsqueeze(-1)).sum(dim=2)
        pooled_stats = (stats_seq.permute(0, 2, 1, 3).to(dtype=attn.dtype) * attn.unsqueeze(-1)).sum(dim=2)
        delta_input = torch.cat(
            [pooled_feat, pooled_stats.to(dtype=pooled_feat.dtype), scene_token[:, None, :].expand(-1, n_person, -1).to(dtype=pooled_feat.dtype)],
            dim=-1,
        )
        relation_delta = self.delta_head(delta_input).squeeze(-1)
        valid_person = person_mask.any(dim=1)
        relation_delta = relation_delta.masked_fill(~valid_person, 0.0)
        pooled_feat = pooled_feat.masked_fill(~valid_person.unsqueeze(-1), 0.0)
        return relation_delta, pooled_feat


class UnaryRelationHead(nn.Module):
    def __init__(self, *, in_dim: int, hidden_dim: int, dropout: float, scene_dim: int) -> None:
        super().__init__()
        self.out_dim = int(hidden_dim)
        self.proj = nn.Linear(int(in_dim), int(hidden_dim))
        self.temporal_attn = nn.Sequential(
            nn.Linear(int(hidden_dim) + int(scene_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, 1),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) + int(scene_dim)),
            nn.Linear(int(hidden_dim) + int(scene_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, fused: torch.Tensor, person_mask: torch.Tensor, scene_token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, total_t, n_person, _ = fused.shape
        h = self.proj(fused).masked_fill(~person_mask.unsqueeze(-1), 0.0)
        scene_expand = scene_token[:, None, None, :].expand(bsz, total_t, n_person, -1).to(dtype=h.dtype)
        attn_logits = self.temporal_attn(torch.cat([h, scene_expand], dim=-1)).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~person_mask, -1e4)
        attn = torch.softmax(attn_logits.permute(0, 2, 1), dim=-1)
        attn = attn * person_mask.permute(0, 2, 1).to(dtype=attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        person_feat = (h.permute(0, 2, 1, 3) * attn.unsqueeze(-1)).sum(dim=2)
        out_input = torch.cat([person_feat, scene_token[:, None, :].expand(-1, n_person, -1).to(dtype=person_feat.dtype)], dim=-1)
        relation_delta = self.out(out_input).squeeze(-1)
        valid_person = person_mask.any(dim=1)
        relation_delta = relation_delta.masked_fill(~valid_person, 0.0)
        person_feat = person_feat.masked_fill(~valid_person.unsqueeze(-1), 0.0)
        return relation_delta, person_feat
