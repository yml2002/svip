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
    """Per-person temporal encoder (Transformer over time)."""

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
        self_cfg = getattr(config.model, "self_branch", None)
        rel_cfg = config.model.relation
        cf_cfg = config.model.counterfactual
        self.use_self_branch = bool(getattr(self_cfg, "enabled", True)) if self_cfg is not None else True
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

        self.use_social_gat = bool(getattr(gat_cfg, "enabled", True))
        if self.use_social_gat:
            self.gat = GATv2Stack(
                in_dim=int(feat_cfg.fused_dim),
                hidden_dim=int(gat_cfg.hidden_dim),
                num_layers=int(gat_cfg.num_layers),
                heads=int(gat_cfg.heads),
                dropout=float(config.model.dropout.gatv2),
                topk_neighbors=int(getattr(gat_cfg, "topk_neighbors", 4)),
            )
            self.no_gat_proj = None
        else:
            self.gat = None
            # No-GAT ablation: keep per-person projection only, without inter-person message passing.
            self.no_gat_proj = nn.Sequential(
                nn.Linear(int(feat_cfg.fused_dim), int(gat_cfg.hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(config.model.dropout.gatv2)),
            )

        self.temporal_encoder = TemporalEncoder(
            d_model=int(tmp_cfg.d_model),
            nhead=int(tmp_cfg.nhead),
            num_layers=int(tmp_cfg.num_layers),
            dim_feedforward=int(tmp_cfg.dim_feedforward),
            dropout=float(config.model.dropout.temporal),
        )

        agg_out = int(getattr(tmp_cfg, "agg_out_dim", int(tmp_cfg.d_model)))
        agg_out_dim = int(agg_out)
        self.rel_feature_dim = agg_out_dim

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

        # Additive scoring: self is the base signal, rel/cf provide explicit increments.
        self.rel_scoring = nn.Sequential(
            nn.LayerNorm(agg_out_dim),
            nn.Linear(agg_out_dim, int(sc_cfg.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(config.model.dropout.scoring)),
            nn.Linear(int(sc_cfg.hidden_dim), 1),
        )
        self.cf_scoring = nn.Sequential(
            nn.LayerNorm(agg_out_dim),
            nn.Linear(agg_out_dim, int(sc_cfg.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(config.model.dropout.scoring)),
            nn.Linear(int(sc_cfg.hidden_dim), 1),
        )
        self.normalize_branch_logits = bool(getattr(sc_cfg, "normalize_branch_logits", True))
        self.branch_gain_floor = float(getattr(sc_cfg, "gain_floor", 0.05))
        self.use_confidence_gate = bool(getattr(sc_cfg, "use_confidence_gate", True))
        self.confidence_gate_floor = float(getattr(sc_cfg, "confidence_gate_floor", 0.05))

        self._self_gain_param = nn.Parameter(self._inv_softplus(1.0 - self.branch_gain_floor))
        self._rel_gain_param = nn.Parameter(self._inv_softplus(1.0 - self.branch_gain_floor))
        self._cf_gain_param = nn.Parameter(self._inv_softplus(1.0 - self.branch_gain_floor))

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

    @staticmethod
    def _inv_softplus(x: float) -> torch.Tensor:
        # Stable inverse for positive initialization values.
        x_t = torch.tensor(float(max(x, 1e-6)), dtype=torch.float32)
        return torch.log(torch.expm1(x_t))

    def _branch_gain(self, key: str) -> torch.Tensor:
        if key == "self":
            p = self._self_gain_param
        elif key == "rel":
            p = self._rel_gain_param
        elif key == "cf":
            p = self._cf_gain_param
        else:
            raise ValueError(f"Unknown branch key: {key}")
        return F.softplus(p) + self.branch_gain_floor

    @staticmethod
    def _normalize_logits_over_valid(logits: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        mask = valid_mask.bool()
        mask_f = mask.to(dtype=logits.dtype)
        count = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (logits * mask_f).sum(dim=1, keepdim=True) / count
        centered = (logits - mean) * mask_f
        var = (centered * centered).sum(dim=1, keepdim=True) / count
        std = torch.sqrt(var + float(eps))
        return (centered / std) * mask_f

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

        if self.use_social_gat:
            assert self.gat is not None
            social = self.gat(fused, pm, bboxes=bboxes)
        else:
            assert self.no_gat_proj is not None
            social = self.no_gat_proj(fused).masked_fill(~pm.unsqueeze(-1), 0.0)

        # Transformer 的时间 mask 是“该 person 在该帧是否有效”
        pm_bt = pm.permute(0, 2, 1).reshape(B * N, T)

        rel_tokens = self._maybe_checkpoint(
            self.temporal_encoder,
            self.to_dmodel(social).permute(0, 2, 1, 3).reshape(B * N, T, -1),
            pm_bt,
        ).reshape(B, N, T, -1).permute(0, 2, 1, 3)  # (B,T,N,D)

        if self.use_event_token:
            rel_tokens = self._maybe_checkpoint(self.event_ctx, rel_tokens, pm).masked_fill(~pm.unsqueeze(-1), 0.0)

        valid_mask = pm.any(dim=1)

        if self.use_relation_branch:
            relation_out = self.relation_reasoner(rel_tokens, pm)
            rel_features = relation_out["relation_features"]
            rel_logits_raw = self.rel_scoring(rel_features).squeeze(-1)
        else:
            relation_out = None
            rel_features = fused.new_zeros((B, N, self.rel_feature_dim))
            rel_logits_raw = fused.new_zeros((B, N))

        if self.use_self_branch:
            self_attn_logits = self.self_temporal_attn(fused).squeeze(-1)
            self_attn = self._masked_softmax(self_attn_logits, pm, dim=1)
            self_pooled = (fused * self_attn.unsqueeze(-1)).sum(dim=1)
            self_logits_raw = self.self_scoring(self_pooled).squeeze(-1)
        else:
            self_attn = pm.new_zeros(pm.shape, dtype=fused.dtype)
            self_logits_raw = fused.new_zeros((B, N))

        if self.use_counterfactual_branch:
            cf_out = self.counterfactual_reasoner(rel_tokens, pm)
            counterfactual_logits_raw = self.cf_scoring(cf_out["counterfactual_features"]).squeeze(-1)
            counterfactual_delta = cf_out["counterfactual_delta"]
            event_state = cf_out["event_state"]
        else:
            cf_out = None
            counterfactual_logits_raw = fused.new_zeros((B, N))
            counterfactual_delta = None
            event_state = None

        if self.normalize_branch_logits:
            self_logits_norm = self._normalize_logits_over_valid(self_logits_raw, valid_mask)
            rel_logits_norm = self._normalize_logits_over_valid(rel_logits_raw, valid_mask)
            cf_logits_norm = self._normalize_logits_over_valid(counterfactual_logits_raw, valid_mask)
        else:
            mask_f = valid_mask.to(dtype=self_logits_raw.dtype)
            self_logits_norm = self_logits_raw * mask_f
            rel_logits_norm = rel_logits_raw * mask_f
            cf_logits_norm = counterfactual_logits_raw * mask_f

        self_gain = self._branch_gain("self")
        rel_gain = self._branch_gain("rel")
        cf_gain = self._branch_gain("cf")

        self_logits = self_logits_norm * self_gain if self.use_self_branch else self_logits_norm
        rel_logits = rel_logits_norm * rel_gain if self.use_relation_branch else rel_logits_norm
        counterfactual_logits = cf_logits_norm * cf_gain if self.use_counterfactual_branch else cf_logits_norm

        branch_gate = self_logits.new_ones((B, 1))
        if self.use_confidence_gate and self.use_self_branch:
            self_for_conf = self_logits.masked_fill(~valid_mask, -1e4)
            self_prob = torch.softmax(self_for_conf / float(self.config.model.scoring.temperature), dim=1)
            self_conf = self_prob.max(dim=1, keepdim=True).values
            branch_gate = (1.0 - self_conf).clamp(min=self.confidence_gate_floor, max=1.0)
            if self.use_relation_branch:
                rel_logits = rel_logits * branch_gate
            if self.use_counterfactual_branch:
                counterfactual_logits = counterfactual_logits * branch_gate

        # Final score is explicit base + increments.
        logits = self_logits + rel_logits
        if self.use_counterfactual_branch:
            logits = logits + counterfactual_logits
        logits = logits.masked_fill(~valid_mask, -1e4)

        scores = torch.softmax(logits / float(self.config.model.scoring.temperature), dim=1) * valid_mask.float()

        out = {
            "importance_logits": logits,
            "importance_scores": scores,
            "video_features": rel_features,
        }

        out["importance_logits_self"] = self_logits.masked_fill(~valid_mask, -1e4)
        out["importance_logits_self_raw"] = self_logits_raw.masked_fill(~valid_mask, -1e4)
        out["self_attention"] = self_attn
        out["importance_logits_rel"] = rel_logits.masked_fill(~valid_mask, -1e4)
        out["importance_logits_rel_raw"] = rel_logits_raw.masked_fill(~valid_mask, -1e4)
        out["branch_gain_self"] = self_gain
        out["branch_gain_rel"] = rel_gain
        out["branch_gain_cf"] = cf_gain
        out["branch_confidence_gate"] = branch_gate
        out["relation_features"] = rel_features
        if relation_out is not None:
            out["relation_attention"] = relation_out["relation_attention"]
        if cf_out is not None:
            out["importance_logits_counterfactual"] = counterfactual_logits.masked_fill(~valid_mask, -1e4)
            out["importance_logits_counterfactual_raw"] = counterfactual_logits_raw.masked_fill(~valid_mask, -1e4)
            out["counterfactual_delta"] = counterfactual_delta
            out["event_state"] = event_state
        return out
