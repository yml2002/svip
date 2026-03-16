"""Importance ranker model."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from src.models.bbox_geom import BBoxGeomEncoder
from src.models.counterfactual_reasoner import CounterfactualMoEReasoner, RelationMoEReasoner
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
        self.relation_dispatch_mode = str(getattr(rel_cfg, "dispatch_mode", "dense")).strip().lower()
        self.counterfactual_dispatch_mode = str(getattr(cf_cfg, "dispatch_mode", "dense")).strip().lower()

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
        if not self.use_relation_branch:
            self.rel_scoring = nn.Sequential(
                nn.Linear(agg_out_dim, int(sc_cfg.hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(config.model.dropout.scoring)),
                nn.Linear(int(sc_cfg.hidden_dim), 1),
            )
        else:
            self.rel_scoring = None

        self.dual_head = bool(getattr(config.training, "enable_dual_head", False))
        if self.dual_head:
            self.self_to_dmodel = nn.Linear(int(feat_cfg.fused_dim), int(tmp_cfg.d_model))
            self.self_scoring = nn.Sequential(
                nn.Linear(int(tmp_cfg.agg_out_dim), int(sc_cfg.hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(config.model.dropout.scoring)),
                nn.Linear(int(sc_cfg.hidden_dim), 1),
            )

        if self.use_relation_branch:
            self.relation_reasoner = RelationMoEReasoner(
                token_dim=int(tmp_cfg.d_model),
                out_dim=agg_out_dim,
                hidden_dim=int(rel_cfg.hidden_dim),
                num_experts=int(rel_cfg.num_experts),
                topk_experts=int(rel_cfg.topk_experts),
                dispatch_mode=self.relation_dispatch_mode,
                router_temperature=float(rel_cfg.router_temperature),
                router_noise_std=float(getattr(rel_cfg, "router_noise_std", 0.0)),
                capacity_factor=float(getattr(rel_cfg, "capacity_factor", 1.25)),
                drop_tokens=bool(getattr(rel_cfg, "drop_tokens", True)),
                dropout=float(rel_cfg.dropout),
            )

        if self.use_counterfactual_branch:
            self.counterfactual_reasoner = CounterfactualMoEReasoner(
                token_dim=int(tmp_cfg.d_model),
                out_dim=agg_out_dim,
                hidden_dim=int(cf_cfg.hidden_dim),
                num_experts=int(cf_cfg.num_experts),
                topk_experts=int(cf_cfg.topk_experts),
                dispatch_mode=self.counterfactual_dispatch_mode,
                router_temperature=float(cf_cfg.router_temperature),
                router_noise_std=float(getattr(cf_cfg, "router_noise_std", 0.0)),
                capacity_factor=float(getattr(cf_cfg, "capacity_factor", 1.25)),
                drop_tokens=bool(getattr(cf_cfg, "drop_tokens", True)),
                dropout=float(cf_cfg.dropout),
            )

        self.fusion_mode = str(getattr(fusion_cfg, "mode", "moe_residual"))
        self.aux_residual_scale = float(getattr(fusion_cfg, "aux_residual_scale", 0.75))
        self.self_hard_conf_threshold = float(getattr(fusion_cfg, "self_hard_conf_threshold", 0.72))
        self.self_hard_margin_threshold = float(getattr(fusion_cfg, "self_hard_margin_threshold", 1.0))
        self.hard_temperature = float(getattr(fusion_cfg, "hard_temperature", 0.12))
        self.branch_conf_threshold = float(getattr(fusion_cfg, "branch_conf_threshold", 0.56))
        self.branch_margin_threshold = float(getattr(fusion_cfg, "branch_margin_threshold", 0.25))
        self.use_branch_rejection = bool(getattr(fusion_cfg, "use_branch_rejection", True))
        self.delta_logit_scale = float(getattr(fusion_cfg, "delta_logit_scale", 1.0))

        self.branch_names = []
        if self.dual_head:
            self.branch_names.append("self")
        # Relation branch is always present (either MoE reasoner or simple scorer fallback).
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
    def _calibrate_branch_logits(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mask_f = valid_mask.to(dtype=logits.dtype)
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (logits * mask_f).sum(dim=1, keepdim=True) / denom
        centered = (logits - mean) * mask_f
        var = (centered * centered).sum(dim=1, keepdim=True) / denom
        std = torch.sqrt(var + 1e-6)
        return centered / std.clamp(min=1e-3)

    @staticmethod
    def _branch_conf_margin(logits: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = logits.masked_fill(~valid_mask, -1e4)
        probs = torch.softmax(masked, dim=1)
        conf = probs.max(dim=1).values
        topk = masked.topk(k=min(2, int(masked.shape[1])), dim=1).values
        if int(topk.shape[1]) < 2:
            margin = conf.new_zeros(conf.shape)
        else:
            margin = topk[:, 0] - topk[:, 1]
        return conf, margin

    def _hard_case_ratio(self, base_logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        conf, margin = self._branch_conf_margin(base_logits, valid_mask)
        hard_by_conf = torch.sigmoid(
            (self.self_hard_conf_threshold - conf) / max(self.hard_temperature, 1e-6)
        )
        hard_by_margin = torch.sigmoid(
            (self.self_hard_margin_threshold - margin) / max(self.hard_temperature, 1e-6)
        )
        return 0.5 * (hard_by_conf + hard_by_margin)

    def _branch_trust(self, branch_logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        conf, margin = self._branch_conf_margin(branch_logits, valid_mask)
        if not self.use_branch_rejection:
            return torch.ones_like(conf)
        conf_ok = conf >= self.branch_conf_threshold
        margin_ok = margin >= self.branch_margin_threshold
        return (conf_ok & margin_ok).to(dtype=branch_logits.dtype)

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

        relation_router_entropy = rel_pooled.new_tensor(0.0)
        counterfactual_router_entropy = rel_pooled.new_tensor(0.0)
        relation_load_balance_loss = rel_pooled.new_tensor(0.0)
        counterfactual_load_balance_loss = rel_pooled.new_tensor(0.0)
        relation_router_z_loss = rel_pooled.new_tensor(0.0)
        counterfactual_router_z_loss = rel_pooled.new_tensor(0.0)
        fusion_router_entropy = rel_pooled.new_tensor(0.0)

        if self.use_relation_branch:
            if self.relation_dispatch_mode == "sparse":
                relation_out = self.relation_reasoner(rel_tokens, pm)
            else:
                relation_out = self._maybe_checkpoint(self.relation_reasoner, rel_tokens, pm)
            rel_features = rel_pooled + relation_out["relation_features"]
            rel_logits = relation_out["relation_logits"]
            relation_router_entropy = relation_out["relation_router_entropy"]
            relation_load_balance_loss = relation_out["relation_load_balance_loss"]
            relation_router_z_loss = relation_out["relation_router_z_loss"]
        else:
            relation_out = None
            rel_features = rel_pooled
            rel_logits = self.rel_scoring(rel_features).squeeze(-1) if self.rel_scoring is not None else rel_pooled.new_zeros((B, N))

        if self.dual_head:
            self_tokens = self._maybe_checkpoint(
                self.temporal_encoder,
                self.self_to_dmodel(fused).permute(0, 2, 1, 3).reshape(B * N, T, -1),
                pm_bt,
            ).reshape(B, N, T, -1).permute(0, 2, 1, 3)

            self_pooled = self._maybe_checkpoint(self.temporal_agg, self_tokens, pm)
            self_logits = self.self_scoring(self_pooled).squeeze(-1)
        else:
            self_logits = None

        if self.use_counterfactual_branch:
            if self.counterfactual_dispatch_mode == "sparse":
                cf_out = self.counterfactual_reasoner(rel_tokens, pm)
            else:
                cf_out = self._maybe_checkpoint(self.counterfactual_reasoner, rel_tokens, pm)
            counterfactual_logits = cf_out["counterfactual_logits"]
            counterfactual_delta = cf_out["counterfactual_delta"]
            event_state = cf_out["event_state"]
            counterfactual_router_entropy = cf_out["counterfactual_router_entropy"]
            counterfactual_load_balance_loss = cf_out["counterfactual_load_balance_loss"]
            counterfactual_router_z_loss = cf_out["counterfactual_router_z_loss"]
        else:
            cf_out = None
            counterfactual_logits = None
            counterfactual_delta = None
            event_state = None

        rel_delta_logits = self.delta_logit_scale * torch.tanh(
            self._calibrate_branch_logits(rel_logits, valid_mask)
        )
        rel_branch_logits = rel_delta_logits
        rel_weight = rel_delta_logits.new_zeros((B,))

        if self.dual_head:
            base_logits = self_logits
        else:
            base_logits = rel_delta_logits

        hard_ratio = self._hard_case_ratio(base_logits, valid_mask)

        if self.dual_head:
            rel_branch_logits = base_logits + rel_delta_logits
            rel_trust = self._branch_trust(rel_branch_logits, valid_mask)
            rel_score = hard_ratio * rel_trust
            rel_weight = rel_score
        else:
            rel_trust = rel_delta_logits.new_ones((B,))

        cf_delta_logits = None
        cf_branch_logits = None
        cf_weight = rel_weight.new_zeros((B,))
        if counterfactual_logits is not None:
            cf_delta_logits = self.delta_logit_scale * torch.tanh(
                self._calibrate_branch_logits(counterfactual_logits, valid_mask)
            )
            if self.dual_head:
                cf_branch_logits = base_logits + cf_delta_logits
            else:
                cf_branch_logits = cf_delta_logits
            cf_trust = self._branch_trust(cf_branch_logits, valid_mask)
            cf_score = hard_ratio * cf_trust
            cf_weight = cf_score

        score_sum = (rel_weight + cf_weight).clamp(min=1e-6)
        rel_norm = rel_weight / score_sum
        cf_norm = cf_weight / score_sum

        if self.dual_head:
            delta_mix = rel_norm.unsqueeze(1) * rel_delta_logits
            if cf_delta_logits is not None:
                delta_mix = delta_mix + cf_norm.unsqueeze(1) * cf_delta_logits
            logits = base_logits + self.aux_residual_scale * hard_ratio.unsqueeze(1) * delta_mix
            self_weight = (1.0 - hard_ratio).clamp(min=0.0, max=1.0)
        else:
            logits = rel_branch_logits
            self_weight = rel_weight.new_zeros((B,))

        rel_effective = hard_ratio * rel_norm
        cf_effective = hard_ratio * cf_norm if counterfactual_logits is not None else rel_weight.new_zeros((B,))

        weight_cols = []
        if self.dual_head:
            weight_cols.append(self_weight)
        weight_cols.append(rel_effective)
        if counterfactual_logits is not None:
            weight_cols.append(cf_effective)
        branch_weight_tensor = torch.stack(weight_cols, dim=1).unsqueeze(1).expand(-1, N, -1)

        logits = logits.masked_fill(~valid_mask, -1e4)

        scores = torch.softmax(logits / float(self.config.model.scoring.temperature), dim=1) * valid_mask.float()

        out = {
            "importance_logits": logits,
            "importance_scores": scores,
            "video_features": rel_features,
            "branch_weights": branch_weight_tensor,
            "branch_weight_names": list(self.branch_names[: branch_weight_tensor.shape[-1]]),
            "fusion_router_entropy": fusion_router_entropy,
            "relation_router_entropy": relation_router_entropy,
            "counterfactual_router_entropy": counterfactual_router_entropy,
            "relation_load_balance_loss": relation_load_balance_loss,
            "counterfactual_load_balance_loss": counterfactual_load_balance_loss,
            "relation_router_z_loss": relation_router_z_loss,
            "counterfactual_router_z_loss": counterfactual_router_z_loss,
        }

        if self.dual_head:
            out["importance_logits_self"] = self_logits
        out["importance_logits_rel"] = rel_branch_logits
        out["relation_delta_logits"] = rel_delta_logits
        out["hard_case_ratio"] = hard_ratio
        out["relation_features"] = rel_features
        if relation_out is not None:
            out["relation_attention"] = relation_out["relation_attention"]
            out["relation_router_probs"] = relation_out["relation_router_probs"]
        if cf_out is not None:
            out["importance_logits_counterfactual"] = cf_branch_logits
            out["counterfactual_delta_logits"] = cf_delta_logits
            out["counterfactual_delta"] = counterfactual_delta
            out["counterfactual_router_probs"] = cf_out["counterfactual_router_probs"]
            out["event_state"] = event_state
        return out
