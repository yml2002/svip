"""Top-level person importance ranking model.

Assembles all sub-modules and defines the end-to-end forward pass:

  frames / bboxes
      → VisionEncoder (DINOv2 ROI crops)      — vis_feats        (B,T,N,D_vis)
      → BBoxGeomEncoder (static geometry)      — geom_feats       (B,T,N,D_geom)
      → fuse MLP                               — fused            (B,T,N,D_fused)
      → Self branch   : temporal mean of fused → self_scores      (B,N)
      → KCGC (optional): cross-attend fused to keyframe context
                         → fused_ctx           (B,T,N,D_fused)
      → Relation branch: SpatioTemporalGATv2(fused_ctx) → rel_scores (B,N)
  logits = self_scores + rel_scores

  KCGC is applied between the two branches deliberately:
  - Self branch sees plain fused (individual appearance, no scene bias)
  - Relation branch sees scene-enriched fused (GAT benefits from global context)
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
from src.models.global_context import GlobalContextModule
from src.models.roi import roi_crop_valid_batch

logger = logging.getLogger(__name__)



class PersonRanker(nn.Module):
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


        # --- Self branch: temporal mean of fused features ---
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
                    topk_neighbors=int(getattr(gat_cfg, "topk_neighbors", 0)),
                    temporal_window=int(getattr(gat_cfg, "temporal_window", 2)),
                    spatial_edge_dim=int(getattr(geom_cfg, "spatial_edge_dim", 32)),
                    temporal_edge_dim=int(getattr(geom_cfg, "temporal_edge_dim", 32)),
                    use_spatial_edges=bool(getattr(gat_cfg, "use_spatial_edges", True)),
                    use_temporal_edges=bool(getattr(gat_cfg, "use_temporal_edges", True)),
                    use_edge_features=bool(getattr(gat_cfg, "use_edge_features", True)),
                    graph_type=str(getattr(gat_cfg, "graph_type", "gatv2")),
                )
            else:
                self.gat = None
                self.no_gat_proj = nn.Sequential(
                    nn.Linear(int(feat_cfg.fused_dim), int(gat_cfg.hidden_dim)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(config.model.dropout.gatv2)),
                )

            # rel_temporal_attn: weights which frames matter for each person.
            # Input = graph_feats + temporal_deviation (how different this frame
            # is from the person's own average — frames with unusual behaviour
            # should get higher attention weight).
            self.rel_temporal_attn = nn.Sequential(
                nn.Linear(int(gat_cfg.hidden_dim) + int(feat_cfg.fused_dim), 1),
            )
            self.rel_scoring = nn.Sequential(
                nn.LayerNorm(int(gat_cfg.hidden_dim)),
                nn.Linear(int(gat_cfg.hidden_dim), int(sc_cfg.hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(config.model.dropout.scoring)),
                nn.Linear(int(sc_cfg.hidden_dim), 1),
            )

        # --- Global Context (KCGC) ---
        gc_cfg = config.model.global_context
        if bool(getattr(gc_cfg, "enabled", True)):
            # num_keyframes must stay well below sampled_frames so keyframes
            # are spread far enough apart to carry different scene context.
            # Rule: K = min(config value, T//4), guaranteeing gap >= 4 frames.
            T_sampled = int(config.data.sampled_frames)
            effective_kf = max(2, min(int(gc_cfg.num_keyframes), T_sampled // 4))
            self.global_ctx = GlobalContextModule(
                fused_dim=int(feat_cfg.fused_dim),
                num_keyframes=effective_kf,
                context_dim=int(gc_cfg.context_dim),
                num_heads=int(gc_cfg.num_heads),
                dropout=float(gc_cfg.dropout),
            )
        else:
            self.global_ctx = None

        self.activation_checkpointing = bool(getattr(config.training, "activation_checkpointing", False))
        self.temperature = float(sc_cfg.temperature)

        branches = []
        if self.use_self_branch: branches.append("self")
        if self.use_relation_branch: branches.append("relation")
        if self.global_ctx is not None: branches.append("kcgc")
        logger.info("Initialized PersonRanker: branches=%s", branches)

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

        # --- Self branch: temporal mean of fused (appearance + geometry) ---
        # Uses fused before KCGC — self branch scores individual appearance,
        # independent of scene-level context.
        if self.use_self_branch:
            self_pooled = (fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            self_scores = self.self_scoring(self_pooled).squeeze(-1)
        else:
            self_scores = fused.new_zeros((B, N))

        # --- Global Context (KCGC): inject scene-level background knowledge ---
        # Applied after self branch so only the relation branch sees scene context.
        # DINOv2 CLS tokens of K keyframes encode objective scene semantics
        # (what kind of scene this is), conditioning GAT on scene type.
        if self.global_ctx is not None:
            fused = self.global_ctx(fused, frames, self.vision, pm)

        # --- Relation branch: spatio-temporal graph over KCGC-enriched features ---
        if self.use_relation_branch:
            if self.use_social_gat and self.gat is not None:
                graph_feats = self._maybe_checkpoint(self.gat, fused, pm, bboxes, vis_feats)
            else:
                graph_feats = self.no_gat_proj(fused).masked_fill(~pm.unsqueeze(-1), 0.0)

            # Temporal deviation: how different is this frame from the person's
            # own average across the video — captures "unusual" moments.
            # fused_mean: (B, N, fused_dim), temporal mean per person
            fused_mean = (fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            temporal_dev = fused - fused_mean.unsqueeze(1)  # (B, T, N, fused_dim)

            # Concat graph features with temporal deviation for attention scoring
            attn_input = torch.cat([graph_feats, temporal_dev], dim=-1)  # (B,T,N,hidden+fused)
            attn_logits = self.rel_temporal_attn(attn_input).squeeze(-1)
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
