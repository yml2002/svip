"""Importance ranker model.

Two components:
  Self     — per-person appearance + geometry scoring (independent per person)
  Relation — spatio-temporal graph capturing person interactions (GAT)

Information flow:
  frames/bboxes → DINOv2 ROI + BBoxGeom → fused features
    → Self branch: temporal mean pool → self_scores
    → Relation branch: spatio-temporal GAT → temporal attn pool → rel_scores
  final = self_scores + rel_scores
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from src.models.bbox_geom import BBoxGeomEncoder
from src.models.vision_encoder import VisionEncoder
from src.models.gatv2 import SpatioTemporalGATv2

logger = logging.getLogger(__name__)


def roi_crop_valid_batch(
    frames: torch.Tensor,
    bboxes: torch.Tensor,
    person_mask: torch.Tensor,
    frame_mask: torch.Tensor,
    out_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ROI crop via grid_sample for valid slots only."""
    device = frames.device
    valid = person_mask & frame_mask.unsqueeze(-1)
    valid_idx = valid.nonzero(as_tuple=False)
    if valid_idx.numel() == 0:
        return valid_idx, frames.new_zeros((0, 3, out_size, out_size))

    u = torch.linspace(0, 1, out_size, device=device, dtype=frames.dtype)
    v = torch.linspace(0, 1, out_size, device=device, dtype=frames.dtype)
    grid_y, grid_x = torch.meshgrid(v, u, indexing="ij")
    base = torch.stack([grid_x, grid_y], dim=-1)

    b = valid_idx[:, 0]
    t = valid_idx[:, 1]
    n = valid_idx[:, 2]
    boxes = bboxes[b, t, n].to(dtype=frames.dtype)
    frames_sel = frames[b, t]

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    gx = x1[:, None, None] + base[None, :, :, 0] * w[:, None, None]
    gy = y1[:, None, None] + base[None, :, :, 1] * h[:, None, None]
    grid = torch.stack([gx * 2 - 1, gy * 2 - 1], dim=-1)
    crops = F.grid_sample(frames_sel, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return valid_idx, crops


class ImportanceRanker(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

        feat_cfg = config.model.features
        dino_cfg = feat_cfg.dino
        geom_cfg = feat_cfg.bbox_geom
        gat_cfg = config.model.gatv2
        sc_cfg = config.model.scoring

        self.use_self_branch = bool(getattr(config.model.self_branch, "enabled", True))
        self.use_relation_branch = bool(getattr(config.model.relation, "enabled", True))
        self.use_geom = bool(getattr(geom_cfg, "enabled", True))

        # --- Feature extraction (shared backbone) ---
        self.vision = VisionEncoder(
            model_dir=str(dino_cfg.model_dir),
            out_dim=int(dino_cfg.feature_dim),
            image_size=int(dino_cfg.image_size),
            freeze=bool(dino_cfg.freeze),
            unfreeze_layers=int(dino_cfg.unfreeze_layers),
        )

        self.geom = BBoxGeomEncoder(out_dim=int(geom_cfg.feature_dim), hidden_dim=int(geom_cfg.hidden_dim)) if self.use_geom else None

        fused_in = int(dino_cfg.feature_dim) + (int(geom_cfg.feature_dim) if self.use_geom else 0)
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, int(feat_cfg.fused_dim)),
            nn.LayerNorm(int(feat_cfg.fused_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(config.model.dropout.features)),
        )

        # --- Self branch ---
        if self.use_self_branch:
            self.self_scoring = nn.Sequential(
                nn.LayerNorm(int(feat_cfg.fused_dim)),
                nn.Linear(int(feat_cfg.fused_dim), int(sc_cfg.hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(config.model.dropout.scoring)),
                nn.Linear(int(sc_cfg.hidden_dim), 1),
            )

        # --- Relation branch ---
        if self.use_relation_branch:
            self.use_social_gat = bool(getattr(gat_cfg, "enabled", True))
            if self.use_social_gat:
                self.gat = SpatioTemporalGATv2(
                    in_dim=int(feat_cfg.fused_dim),
                    hidden_dim=int(gat_cfg.hidden_dim),
                    num_layers=int(gat_cfg.num_layers),
                    heads=int(gat_cfg.heads),
                    dropout=float(config.model.dropout.gatv2),
                    topk_neighbors=int(getattr(gat_cfg, "topk_neighbors", 4)),
                    temporal_window=int(getattr(gat_cfg, "temporal_window", 3)),
                    spatial_edge_dim=int(getattr(geom_cfg, "spatial_edge_dim", 32)),
                    temporal_edge_dim=int(getattr(geom_cfg, "temporal_edge_dim", 16)),
                    use_temporal_edges=bool(getattr(gat_cfg, "use_temporal_edges", True)),
                    use_edge_features=bool(getattr(gat_cfg, "use_edge_features", True)),
                )
            else:
                self.gat = None
                self.no_gat_proj = nn.Sequential(
                    nn.Linear(int(feat_cfg.fused_dim), int(gat_cfg.hidden_dim)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(config.model.dropout.gatv2)),
                )

            self.rel_temporal_attn = nn.Sequential(
                nn.Linear(int(gat_cfg.hidden_dim), 1),
            )
            self.rel_scoring = nn.Sequential(
                nn.LayerNorm(int(gat_cfg.hidden_dim)),
                nn.Linear(int(gat_cfg.hidden_dim), int(sc_cfg.hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(config.model.dropout.scoring)),
                nn.Linear(int(sc_cfg.hidden_dim), 1),
            )

        self.activation_checkpointing = bool(getattr(config.training, "activation_checkpointing", False))
        self.temperature = float(sc_cfg.temperature)

        branches = []
        if self.use_self_branch: branches.append("self")
        if self.use_relation_branch: branches.append("relation")
        logger.info("Initialized ImportanceRanker: branches=%s", branches)

    def _maybe_checkpoint(self, fn, *args):
        if self.training and self.activation_checkpointing:
            return torch_checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(
        self,
        frames: torch.Tensor,
        bboxes: torch.Tensor,
        person_mask: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
        target_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        B, T, N = person_mask.shape

        fm = frame_mask if frame_mask is not None else person_mask.any(dim=-1)
        pm = person_mask & fm.unsqueeze(-1)

        # --- Person ROI features ---
        roi_chunk = int(getattr(self.config.training, "roi_chunk", 512))
        dino_dim = int(self.config.model.features.dino.feature_dim)
        vis_feats = frames.new_zeros((B, T, N, dino_dim))
        valid_idx, crops_valid = roi_crop_valid_batch(
            frames, bboxes, pm, fm,
            out_size=int(self.config.model.features.dino.image_size),
        )
        if valid_idx.numel() > 0:
            b_idx, t_idx, n_idx = valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]
            for s in range(0, int(crops_valid.shape[0]), roi_chunk):
                e = min(int(crops_valid.shape[0]), s + roi_chunk)
                vis_chunk = self.vision(crops_valid[s:e])
                vis_feats[b_idx[s:e], t_idx[s:e], n_idx[s:e]] = vis_chunk.to(dtype=vis_feats.dtype)

        geom_feats = self.geom(bboxes, pm) if self.geom is not None else None
        if geom_feats is not None:
            fused = self.fuse(torch.cat([vis_feats, geom_feats], dim=-1))
        else:
            fused = self.fuse(vis_feats)
        fused = fused.masked_fill(~pm.unsqueeze(-1), 0.0)

        valid_mask = pm.any(dim=1)
        mask_f = pm.to(dtype=fused.dtype).unsqueeze(-1)

        # --- Self branch ---
        if self.use_self_branch:
            self_pooled = (fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            self_scores = self.self_scoring(self_pooled).squeeze(-1)
        else:
            self_scores = fused.new_zeros((B, N))

        # --- Relation branch ---
        if self.use_relation_branch:
            if self.use_social_gat and self.gat is not None:
                graph_feats = self._maybe_checkpoint(self.gat, fused, pm, bboxes)
            else:
                graph_feats = self.no_gat_proj(fused).masked_fill(~pm.unsqueeze(-1), 0.0)

            attn_logits = self.rel_temporal_attn(graph_feats).squeeze(-1)
            attn_logits = attn_logits.masked_fill(~pm, -1e4)
            attn_perm = attn_logits.permute(0, 2, 1)
            attn_w = torch.softmax(attn_perm, dim=-1)
            attn_w = attn_w * pm.permute(0, 2, 1).to(dtype=attn_w.dtype)
            attn_w = attn_w / attn_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)

            graph_perm = graph_feats.permute(0, 2, 1, 3)
            rel_pooled = (graph_perm * attn_w.unsqueeze(-1)).sum(dim=2)
            rel_scores = self.rel_scoring(rel_pooled).squeeze(-1)
        else:
            rel_scores = fused.new_zeros((B, N))

        # --- Fusion ---
        logits = self_scores + rel_scores
        logits = logits.masked_fill(~valid_mask, -1e4)

        scores = torch.softmax(logits / self.temperature, dim=1) * valid_mask.float()

        return {
            "importance_logits": logits,
            "importance_scores": scores,
            "importance_logits_self": self_scores.masked_fill(~valid_mask, -1e4),
            "importance_logits_rel": rel_scores.masked_fill(~valid_mask, -1e4),
        }
