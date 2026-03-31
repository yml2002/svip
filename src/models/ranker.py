"""Top-level person importance ranking model.

Responsibilities are explicit:
- unary prior: person-centric importance from appearance, geometry, and motion
- open-world scene context: latent environment token from keyframes
- temporal social memory: history-aware social refinement conditioned on scene

Final score:
    final = unary_prior + gate(scene, person) * relation_delta
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from src.models.bbox_geom import BBoxGeomEncoder
from src.models.intrinsic import IntrinsicImportanceModule
from src.models.relation import TemporalSocialMemoryEncoder, UnaryRelationHead
from src.models.roi import roi_crop_from_indices, roi_valid_indices
from src.models.scene_prior import ScenePriorModule
from src.models.vision_encoder import VisionEncoder

logger = logging.getLogger(__name__)



class PersonRanker(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

        feat_cfg = config.model.features
        dino_cfg = feat_cfg.dino
        geom_cfg = feat_cfg.bbox_geom
        gat_cfg = config.model.gatv2
        gc_cfg = config.model.global_context
        sc_cfg = config.model.scoring
        intrinsic_cfg = config.model.intrinsic

        self.use_relation_branch = bool(getattr(config.model.relation, "enabled", True))
        self.relation_delta_scale = float(getattr(config.model.relation, "delta_scale", 1.0))
        self.use_adaptive_gate = bool(getattr(config.model.relation, "use_adaptive_gate", True))
        self.relation_gate_bias = float(getattr(config.model.relation, "gate_bias", 0.90))
        self.use_geom = bool(getattr(geom_cfg, "enabled", True))
        self.geom_fuse_scale = float(getattr(geom_cfg, "fuse_scale", 0.5))
        self.geom_dropout_prob = float(getattr(geom_cfg, "dropout_prob", 0.0))

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
            nn.GELU(),
            nn.Dropout(float(config.model.dropout.features)),
        )

        self.intrinsic = IntrinsicImportanceModule(
            fused_dim=int(feat_cfg.fused_dim),
            hidden_dim=int(sc_cfg.hidden_dim),
            dropout=float(config.model.dropout.scoring),
            use_geom_priors=self.use_geom,
            use_motion_priors=bool(getattr(intrinsic_cfg, "use_motion_priors", True)),
            use_max_pool=bool(getattr(intrinsic_cfg, "use_max_pool", True)),
            use_attention_pool=bool(getattr(intrinsic_cfg, "use_attention_pool", True)),
        )

        if self.use_relation_branch:
            self.use_social_gat = bool(getattr(gat_cfg, "enabled", True))
            if self.use_social_gat:
                self.relation = TemporalSocialMemoryEncoder(
                    in_dim=int(feat_cfg.fused_dim),
                    hidden_dim=int(gat_cfg.hidden_dim),
                    num_layers=int(gat_cfg.num_layers),
                    heads=int(gat_cfg.heads),
                    dropout=float(config.model.dropout.gatv2),
                    topk_neighbors=int(getattr(gat_cfg, "topk_neighbors", 0)),
                    use_spatial_edges=bool(getattr(gat_cfg, "use_spatial_edges", True)),
                    use_temporal_edges=bool(getattr(gat_cfg, "use_temporal_edges", True)),
                    use_edge_features=bool(getattr(gat_cfg, "use_edge_features", True)) and self.use_geom,
                    graph_type=str(getattr(gat_cfg, "graph_type", "gatv2")),
                    scene_dim=int(gc_cfg.context_dim),
                )
            else:
                self.relation = UnaryRelationHead(
                    in_dim=int(feat_cfg.fused_dim),
                    hidden_dim=int(gat_cfg.hidden_dim),
                    dropout=float(config.model.dropout.gatv2),
                    scene_dim=int(gc_cfg.context_dim),
                )
            relation_feat_dim = int(getattr(self.relation, "out_dim", int(gat_cfg.hidden_dim)))
            self.relation_gate_rel_proj = nn.Linear(relation_feat_dim, int(sc_cfg.hidden_dim))
            self.scene_gate_proj = nn.Sequential(
                nn.LayerNorm(int(gc_cfg.context_dim)),
                nn.Linear(int(gc_cfg.context_dim), int(sc_cfg.hidden_dim)),
                nn.GELU(),
                nn.Linear(int(sc_cfg.hidden_dim), int(sc_cfg.hidden_dim)),
            )
            self.relation_gate = nn.Sequential(
                nn.LayerNorm(int(sc_cfg.hidden_dim) * 3),
                nn.Linear(int(sc_cfg.hidden_dim) * 3, int(sc_cfg.hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(config.model.dropout.scoring)),
                nn.Linear(int(sc_cfg.hidden_dim), 1),
            )
        else:
            self.relation = None
            self.relation_gate = None
            self.relation_gate_rel_proj = None
            self.scene_gate_proj = None

        if bool(getattr(gc_cfg, "enabled", True)):
            T_sampled = int(config.data.sampled_frames)
            effective_kf = max(2, min(int(gc_cfg.num_keyframes), T_sampled // 4))
            self.global_ctx = ScenePriorModule(
                num_keyframes=effective_kf,
                context_dim=int(gc_cfg.context_dim),
                num_heads=int(gc_cfg.num_heads),
                dropout=float(gc_cfg.dropout),
                num_layers=int(getattr(gc_cfg, "num_layers", 1)),
                num_prototypes=int(getattr(gc_cfg, "num_prototypes", 8)),
            )
        else:
            self.global_ctx = None

        self.activation_checkpointing = bool(getattr(config.training, "activation_checkpointing", False))

        branches = ["intrinsic"]
        if self.use_relation_branch:
            branches.append("relation")
        if self.global_ctx is not None:
            branches.append("scene_prior")
        logger.info("Initialized PersonRanker: branches=%s", branches)

    def _maybe_checkpoint(self, fn, *args):
        if self.training and self.activation_checkpointing:
            return torch_checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def configure_train_stage(self, epoch: int) -> None:
        train_cfg = self.config.training
        self.vision.configure_train_stage(
            int(epoch),
            warmup_epochs=int(getattr(train_cfg, "backbone_warmup_epochs", 0)),
            train_mode=str(getattr(train_cfg, "backbone_train_mode", "attn_ln")),
        )

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
        valid_idx = roi_valid_indices(pm, fm)
        if valid_idx.numel() > 0:
            out_size = int(self.config.model.features.dino.image_size)
            for s in range(0, int(valid_idx.shape[0]), roi_chunk):
                e = min(int(valid_idx.shape[0]), s + roi_chunk)
                idx_chunk = valid_idx[s:e]
                crops_chunk = roi_crop_from_indices(frames, bboxes, idx_chunk, out_size=out_size)
                vis_chunk = self.vision(crops_chunk)
                vis_feats[idx_chunk[:, 0], idx_chunk[:, 1], idx_chunk[:, 2]] = vis_chunk.to(dtype=vis_feats.dtype)

        geom_feats = self.geom(bboxes, pm) if self.geom is not None else None
        if geom_feats is not None:
            geom_main = geom_feats * self.geom_fuse_scale
            if self.training and self.geom_dropout_prob > 0:
                geom_keep = torch.rand((B, 1, N, 1), device=geom_main.device) > self.geom_dropout_prob
                geom_main = geom_main * geom_keep.to(dtype=geom_main.dtype)
            fused = self.fuse(torch.cat([vis_feats, geom_main], dim=-1))
        else:
            fused = self.fuse(vis_feats)
        fused = fused.masked_fill(~pm.unsqueeze(-1), 0.0)


        valid_mask = pm.any(dim=1)
        if self.global_ctx is not None:
            scene_token, scene_logits, scene_tokens = self.global_ctx(frames, self.vision)
        else:
            scene_dim = int(self.config.model.global_context.context_dim)
            scene_token = fused.new_zeros((B, scene_dim)).to(dtype=fused.dtype)
            scene_logits = None
            scene_tokens = None
        intrinsic_logits, intrinsic_feat, intrinsic_priors = self.intrinsic(fused, bboxes.to(dtype=fused.dtype), pm)

        if self.use_relation_branch:
            if self.use_social_gat:
                relation_delta, relation_feat = self._maybe_checkpoint(self.relation, fused, pm, bboxes.to(dtype=fused.dtype), scene_token)
            else:
                relation_delta, relation_feat = self._maybe_checkpoint(self.relation, fused, pm, scene_token)
            if self.global_ctx is not None and self.relation_gate is not None and self.use_adaptive_gate:
                rel_gate_feat = self.relation_gate_rel_proj(relation_feat)
                scene_gate_feat = self.scene_gate_proj(scene_token)
                gate_input = torch.cat([
                    intrinsic_feat,
                    rel_gate_feat,
                    scene_gate_feat[:, None, :].expand(-1, N, -1).to(dtype=intrinsic_feat.dtype),
                ], dim=-1)
                relation_gate = self.relation_gate_bias + 0.15 * torch.tanh(self.relation_gate(gate_input).squeeze(-1))
                relation_gate = relation_gate.masked_fill(~valid_mask, 0.0)
            else:
                relation_gate = fused.new_ones((B, N)) * valid_mask.float()
            rel_scores = relation_gate * (self.relation_delta_scale * relation_delta)
        else:
            rel_feat_dim = int(getattr(self.relation, "out_dim", self.config.model.gatv2.hidden_dim)) if self.relation is not None else int(self.config.model.gatv2.hidden_dim)
            relation_feat = fused.new_zeros((B, N, rel_feat_dim))
            relation_delta = fused.new_zeros((B, N))
            relation_gate = fused.new_zeros((B, N))
            rel_scores = fused.new_zeros((B, N))

        logits = intrinsic_logits + rel_scores
        logits = logits.masked_fill(~valid_mask, -1e4)

        scores = torch.softmax(logits, dim=1) * valid_mask.float()

        return {
            "importance_logits": logits,
            "importance_scores": scores,
            "importance_logits_self": intrinsic_logits.masked_fill(~valid_mask, -1e4),
            "importance_logits_rel": rel_scores.masked_fill(~valid_mask, -1e4),
            "intrinsic_logits": intrinsic_logits.masked_fill(~valid_mask, -1e4),
            "relation_logits": relation_delta.masked_fill(~valid_mask, 0.0),
            "relation_gate": relation_gate.masked_fill(~valid_mask, 0.0),
            "intrinsic_features": intrinsic_feat,
            "relation_features": relation_feat,
            "intrinsic_priors": intrinsic_priors,
            "scene_logits": scene_logits,
            "scene_token": scene_token,
            "scene_tokens": scene_tokens,
        }
