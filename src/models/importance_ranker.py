"""Importance ranker model."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from src.models.bbox_geom import BBoxGeomEncoder
from src.models.counterfactual_reasoner import CounterfactualReasoner, RelationReasoner
from src.models.event_context import EventTokenContext
from src.models.vision_encoder import VisionEncoder
from src.models.gatv2 import GATv2Stack
from src.models.video_aggregator import VideoLevelAggregator

logger = logging.getLogger(__name__)


def roi_crop_valid_batch(
    frames: torch.Tensor,  # (B,T,3,H,W)
    bboxes: torch.Tensor,  # (B,T,N,4) normalized
    person_mask: torch.Tensor,
    frame_mask: torch.Tensor,
    out_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ROI crop via grid_sample for valid slots only.

    Returns:
        valid_idx: (K,3) [b,t,n]
        valid_crops: (K,3,out_size,out_size)
    """

    device = frames.device
    valid = person_mask & frame_mask.unsqueeze(-1)  # (B,T,N)
    valid_idx = valid.nonzero(as_tuple=False)  # (K,3)
    if valid_idx.numel() == 0:
        return valid_idx, frames.new_zeros((0, 3, out_size, out_size))

    # Keep grid computations in the same dtype as frames to avoid AMP dtype mismatch.
    u = torch.linspace(0, 1, out_size, device=device, dtype=frames.dtype)
    v = torch.linspace(0, 1, out_size, device=device, dtype=frames.dtype)
    grid_y, grid_x = torch.meshgrid(v, u, indexing="ij")
    base = torch.stack([grid_x, grid_y], dim=-1)  # (S,S,2)

    b = valid_idx[:, 0]
    t = valid_idx[:, 1]
    n = valid_idx[:, 2]
    boxes = bboxes[b, t, n].to(dtype=frames.dtype)  # (K,4)
    frames_sel = frames[b, t]  # (K,3,H,W)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    gx = x1[:, None, None] + base[None, :, :, 0] * w[:, None, None]
    gy = y1[:, None, None] + base[None, :, :, 1] * h[:, None, None]
    grid = torch.stack([gx * 2 - 1, gy * 2 - 1], dim=-1)  # (K,S,S,2)
    crops = F.grid_sample(
        frames_sel,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return valid_idx, crops


class TemporalEncoder(nn.Module):
    """Per-person temporal encoder.

    We keep a lightweight TransformerEncoder over time and then apply an
    attention pooling (VideoLevelAggregator) to focus on key moments.
    """

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
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        key_padding = ~time_mask
        return self.encoder(x, src_key_padding_mask=key_padding)


class ImportanceRanker(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

        feat_cfg = config.model.features
        dino_cfg = feat_cfg.dino
        geom_cfg = feat_cfg.bbox_geom
        gat_cfg = config.model.gatv2
        tmp_cfg = config.model.temporal
        sc_cfg = config.model.scoring
        rel_cfg = config.model.relation
        cf_cfg = config.model.counterfactual
        fusion_cfg = config.model.fusion
        self.use_relation_branch = bool(getattr(rel_cfg, "enabled", True))
        self.use_counterfactual_branch = bool(getattr(cf_cfg, "enabled", True))

        if not bool(dino_cfg.enabled):
            raise ValueError("Vision backbone is disabled. Set config.model.features.dino.enabled=True")

        self.vision = VisionEncoder(
            model_dir=str(dino_cfg.model_dir),
            out_dim=int(dino_cfg.feature_dim),
            image_size=int(dino_cfg.image_size),
            freeze=bool(dino_cfg.freeze),
            unfreeze_layers=int(dino_cfg.unfreeze_layers),
        )

        self.geom = BBoxGeomEncoder(out_dim=int(geom_cfg.feature_dim), hidden_dim=int(geom_cfg.hidden_dim))

        fused_in = int(dino_cfg.feature_dim) + int(geom_cfg.feature_dim)
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, int(feat_cfg.fused_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(config.model.dropout.features)),
        )

        self.gat = GATv2Stack(
            in_dim=int(feat_cfg.fused_dim),
            hidden_dim=int(gat_cfg.hidden_dim),
            num_layers=int(gat_cfg.num_layers),
            heads=int(gat_cfg.heads),
            dropout=float(config.model.dropout.gatv2),
            use_residual=bool(gat_cfg.use_residual),
        )

        self.temporal_encoder = TemporalEncoder(
            d_model=int(tmp_cfg.d_model),
            nhead=int(tmp_cfg.nhead),
            num_layers=int(tmp_cfg.num_layers),
            dim_feedforward=int(tmp_cfg.dim_feedforward),
            dropout=float(config.model.dropout.temporal),
        )

        agg_heads = int(getattr(tmp_cfg, "agg_heads", 8))
        agg_out = int(getattr(tmp_cfg, "agg_out_dim", int(tmp_cfg.d_model)))
        use_video_transformer = bool(getattr(tmp_cfg, "use_video_transformer", False))
        transformer_layers = int(getattr(tmp_cfg, "transformer_layers", 0))
        pooling_cfg = getattr(tmp_cfg, "pooling", None)
        pooling = str(pooling_cfg)
        self.temporal_agg = VideoLevelAggregator(
            input_dim=int(tmp_cfg.d_model),
            out_dim=agg_out,
            num_heads=agg_heads,
            dropout=float(config.model.dropout.temporal),
            use_video_transformer=use_video_transformer,
            pooling=pooling,
            transformer_layers=max(1, transformer_layers) if use_video_transformer else 1,
        )
        agg_out_dim = int(agg_out)

        self.use_event_token = bool(getattr(tmp_cfg, "use_event_token", True))
        event_layers = int(getattr(tmp_cfg, "event_num_layers", 1))
        if self.use_event_token:
            self.event_ctx = EventTokenContext(
                d_model=int(tmp_cfg.d_model),
                nhead=int(tmp_cfg.nhead),
                dim_feedforward=int(tmp_cfg.dim_feedforward),
                dropout=float(config.model.dropout.temporal),
                event_num_layers=event_layers,
            )

        self.to_dmodel = nn.Linear(int(gat_cfg.hidden_dim), int(tmp_cfg.d_model))

        # Keep self branch independent from graph-enhanced relation features.
        self.dual_head = bool(getattr(config.training, "enable_dual_head", False))
        if not self.dual_head:
            raise ValueError("config.training.enable_dual_head must be True to keep self/rel paths disentangled")
        self.self_temporal_attn = nn.Sequential(
            nn.LayerNorm(int(feat_cfg.fused_dim)),
            nn.Linear(int(feat_cfg.fused_dim), 1),
        )
        self.self_scoring = nn.Sequential(
            nn.Linear(int(feat_cfg.fused_dim), int(sc_cfg.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(config.model.dropout.scoring)),
            nn.Linear(int(sc_cfg.hidden_dim), 1),
        )

        fusion_dim = int(getattr(fusion_cfg, "feature_dim", int(agg_out_dim)))
        if fusion_dim <= 0:
            raise ValueError(f"fusion.feature_dim must be > 0, got {fusion_dim}")

        self.self_feature_proj = nn.Linear(int(feat_cfg.fused_dim), fusion_dim)
        self.rel_feature_proj = nn.Linear(int(agg_out_dim), fusion_dim)
        self.cf_feature_proj = nn.Linear(int(agg_out_dim), fusion_dim)

        fusion_heads = int(getattr(fusion_cfg, "interaction_heads", 8))
        if fusion_dim % max(1, fusion_heads) != 0:
            raise ValueError(
                f"fusion.feature_dim ({fusion_dim}) must be divisible by fusion.interaction_heads ({fusion_heads})"
            )
        fusion_layers = max(1, int(getattr(fusion_cfg, "interaction_layers", 1)))
        fusion_dropout = float(getattr(fusion_cfg, "dropout", 0.1))

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=fusion_heads,
            dim_feedforward=max(2 * fusion_dim, 512),
            dropout=fusion_dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.feature_interaction = nn.TransformerEncoder(fusion_layer, num_layers=fusion_layers)
        self.feature_pool = nn.Linear(fusion_dim, 1)
        self.final_scoring = nn.Sequential(
            nn.Linear(fusion_dim, int(sc_cfg.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(config.model.dropout.scoring)),
            nn.Linear(int(sc_cfg.hidden_dim), 1),
        )

        if self.use_relation_branch:
            self.relation_reasoner = RelationReasoner(
                token_dim=int(tmp_cfg.d_model),
                out_dim=agg_out_dim,
                hidden_dim=int(rel_cfg.hidden_dim),
                dropout=float(rel_cfg.dropout),
            )

        if self.use_counterfactual_branch:
            self.counterfactual_reasoner = CounterfactualReasoner(
                token_dim=int(tmp_cfg.d_model),
                out_dim=agg_out_dim,
                hidden_dim=int(cf_cfg.hidden_dim),
                dropout=float(cf_cfg.dropout),
            )

        self.branch_names = ["self"]
        if self.use_relation_branch:
            self.branch_names.append("rel")
        if self.use_counterfactual_branch:
            self.branch_names.append("counterfactual")

        self.activation_checkpointing = bool(getattr(config.training, "activation_checkpointing", True))

        logger.info("Initialized ImportanceRanker (vision_dir=%s)", str(dino_cfg.model_dir))

    def _maybe_checkpoint(self, fn, *args):
        if self.training and self.activation_checkpointing:
            return torch_checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        masked = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(masked, dim=dim)
        weights = weights * mask.to(dtype=weights.dtype)
        return weights / weights.sum(dim=dim, keepdim=True).clamp(min=1e-6)

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

        roi_chunk = int(getattr(self.config.training, "roi_chunk", 256))
        vis_feats = frames.new_zeros((B, T, N, int(self.config.model.features.dino.feature_dim)))
        valid_idx, crops_valid = roi_crop_valid_batch(
            frames,
            bboxes,
            pm,
            fm,
            out_size=int(self.config.model.features.dino.image_size),
        )
        if valid_idx.numel() > 0:
            chunk = roi_chunk
            b_idx = valid_idx[:, 0]
            t_idx = valid_idx[:, 1]
            n_idx = valid_idx[:, 2]
            for s in range(0, int(crops_valid.shape[0]), chunk):
                e = min(int(crops_valid.shape[0]), s + chunk)
                vis_chunk = self.vision(crops_valid[s:e])
                vis_feats[b_idx[s:e], t_idx[s:e], n_idx[s:e]] = vis_chunk.to(dtype=vis_feats.dtype)

        geom_feats = self.geom(bboxes, pm)

        fused = self.fuse(torch.cat([vis_feats, geom_feats], dim=-1)).masked_fill(~pm.unsqueeze(-1), 0.0)

        social = self.gat(fused, pm)

        # Transformer 的时间 mask 是“该 person 在该帧是否有效”
        pm_bt = pm.permute(0, 2, 1).reshape(B * N, T)

        rel_tokens = self._maybe_checkpoint(
            self.temporal_encoder,
            self.to_dmodel(social).permute(0, 2, 1, 3).reshape(B * N, T, -1),
            pm_bt,
        ).reshape(B, N, T, -1).permute(0, 2, 1, 3)  # (B,T,N,D)

        if self.use_event_token:
            rel_tokens = self._maybe_checkpoint(self.event_ctx, rel_tokens, pm).masked_fill(~pm.unsqueeze(-1), 0.0)

        rel_pooled = self._maybe_checkpoint(self.temporal_agg, rel_tokens, pm)  # (B,N,D)

        valid_mask = pm.any(dim=1)

        if self.use_relation_branch:
            relation_out = self.relation_reasoner(rel_tokens, pm)
            rel_features = rel_pooled + relation_out["relation_features"]
            rel_logits = relation_out["relation_logits"]
        else:
            relation_out = None
            rel_features = rel_pooled
            rel_logits = rel_pooled.new_zeros((B, N))

        self_attn_logits = self.self_temporal_attn(fused).squeeze(-1)
        self_attn = self._masked_softmax(self_attn_logits, pm, dim=1)
        self_pooled = (fused * self_attn.unsqueeze(-1)).sum(dim=1)
        self_logits = self.self_scoring(self_pooled).squeeze(-1)
        self_logits = self_logits.masked_fill(~valid_mask, -1e4)

        if self.use_counterfactual_branch:
            cf_out = self.counterfactual_reasoner(rel_tokens, pm)
            counterfactual_logits = cf_out["counterfactual_logits"]
            counterfactual_delta = cf_out["counterfactual_delta"]
            event_state = cf_out["event_state"]
        else:
            cf_out = None
            counterfactual_logits = None
            counterfactual_delta = None
            event_state = None

        branch_features = [self.self_feature_proj(self_pooled)]
        branch_names = ["self"]

        if self.use_relation_branch:
            branch_features.append(self.rel_feature_proj(rel_features))
            branch_names.append("rel")

        if self.use_counterfactual_branch and cf_out is not None:
            branch_features.append(self.cf_feature_proj(cf_out["counterfactual_features"]))
            branch_names.append("counterfactual")

        branch_stack = torch.stack(branch_features, dim=2)  # (B,N,M,D)
        branch_mask = valid_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=branch_stack.dtype)
        branch_stack = branch_stack * branch_mask

        M = int(branch_stack.shape[2])
        D = int(branch_stack.shape[3])
        branch_tokens = branch_stack.reshape(B * N, M, D)
        branch_tokens = self.feature_interaction(branch_tokens)

        pool_logits = self.feature_pool(branch_tokens).squeeze(-1)  # (B*N, M)
        pool_weights = torch.softmax(pool_logits, dim=1)
        fused_features = (branch_tokens * pool_weights.unsqueeze(-1)).sum(dim=1)
        fused_features = fused_features.reshape(B, N, D)

        logits = self.final_scoring(fused_features).squeeze(-1)
        logits = logits.masked_fill(~valid_mask, -1e4)

        branch_weight_tensor = pool_weights.reshape(B, N, M) * valid_mask.unsqueeze(-1).to(dtype=pool_weights.dtype)
        branch_weight_tensor = branch_weight_tensor / branch_weight_tensor.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        scores = torch.softmax(logits / float(self.config.model.scoring.temperature), dim=1) * valid_mask.float()

        out = {
            "importance_logits": logits,
            "importance_scores": scores,
            "video_features": fused_features,
            "branch_weights": branch_weight_tensor,
            "branch_weight_names": branch_names,
        }

        out["importance_logits_self"] = self_logits
        out["self_attention"] = self_attn
        out["importance_logits_rel"] = rel_logits
        out["relation_features"] = rel_features
        if relation_out is not None:
            out["relation_attention"] = relation_out["relation_attention"]
        if cf_out is not None:
            out["importance_logits_counterfactual"] = counterfactual_logits
            out["counterfactual_delta"] = counterfactual_delta
            out["event_state"] = event_state
        return out
