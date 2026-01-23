"""Importance ranker model."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bbox_geom import BBoxGeomEncoder
from src.models.event_context import EventTokenContext
from src.models.segment_event_context import SegmentEventContext
from src.models.vision_encoder import VisionEncoder
from src.models.gatv2 import GATv2Stack
from src.models.video_aggregator import VideoLevelAggregator
from src.models.spatiotemporal_graph import SpatiotemporalGraphTransformer

logger = logging.getLogger(__name__)


def roi_crop_batch(
    frames: torch.Tensor,  # (B,T,3,H,W)
    bboxes: torch.Tensor,  # (B,T,N,4) normalized
    person_mask: torch.Tensor,
    frame_mask: torch.Tensor,
    out_size: int,
    roi_chunk: int = 1024,
) -> torch.Tensor:
    """ROI crop via grid_sample. Returns (B,T,N,3,out_size,out_size)."""

    B, T, _, H, W = frames.shape
    _, _, N, _ = bboxes.shape
    device = frames.device

    valid = person_mask & frame_mask.unsqueeze(-1)  # (B,T,N)
    crops = frames.new_zeros((B, T, N, 3, out_size, out_size))
    valid_idx = valid.nonzero(as_tuple=False)  # (K,3)
    if valid_idx.numel() == 0:
        return crops

    # Keep grid computations in the same dtype as frames to avoid AMP dtype mismatch.
    u = torch.linspace(0, 1, out_size, device=device, dtype=frames.dtype)
    v = torch.linspace(0, 1, out_size, device=device, dtype=frames.dtype)
    grid_y, grid_x = torch.meshgrid(v, u, indexing="ij")
    base = torch.stack([grid_x, grid_y], dim=-1)  # (S,S,2)

    chunk = max(1, int(roi_chunk))

    # Flatten index for easier slicing.
    b = valid_idx[:, 0]
    t = valid_idx[:, 1]
    n = valid_idx[:, 2]
    boxes = bboxes[b, t, n].to(dtype=frames.dtype)  # (K,4)
    frames_sel = frames[b, t]  # (K,3,H,W)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)

    for s in range(0, int(valid_idx.shape[0]), chunk):
        e = min(int(valid_idx.shape[0]), s + chunk)
        gx = x1[s:e, None, None] + base[None, :, :, 0] * w[s:e, None, None]
        gy = y1[s:e, None, None] + base[None, :, :, 1] * h[s:e, None, None]
        grid = torch.stack([gx * 2 - 1, gy * 2 - 1], dim=-1)  # (C,S,S,2)
        crop_chunk = F.grid_sample(
            frames_sel[s:e],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        crops[b[s:e], t[s:e], n[s:e]] = crop_chunk.to(dtype=crops.dtype)

    return crops


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
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        key_padding = ~time_mask
        return self.encoder(x, src_key_padding_mask=key_padding)


"""Importance ranker model."""


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

        if not bool(dino_cfg.enabled):
            raise ValueError("Vision backbone is disabled. Set config.model.features.dino.enabled=True")

        self.vision = VisionEncoder(
            model_dir=str(dino_cfg.model_dir),
            out_dim=int(dino_cfg.feature_dim),
            image_size=int(dino_cfg.image_size),
            freeze=bool(dino_cfg.freeze),
            finetune_layers=int(getattr(dino_cfg, "finetune_layers", 0)),
        )

        self.geom = BBoxGeomEncoder(out_dim=int(geom_cfg.feature_dim), hidden_dim=int(geom_cfg.hidden_dim))

        fused_in = int(dino_cfg.feature_dim) + int(geom_cfg.feature_dim)
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, int(feat_cfg.fused_dim)),
            nn.LayerNorm(int(feat_cfg.fused_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(config.model.dropout.features)),
        )

        # Choose between GATv2 (old) or Spatiotemporal Graph (new)
        st_cfg = getattr(config.model, "st_graph", None)
        use_st_graph = st_cfg and bool(st_cfg.enabled)
        
        if use_st_graph:
            logger.info("Using Spatiotemporal Graph Transformer")
            self.st_graph = SpatiotemporalGraphTransformer(
                input_dim=int(feat_cfg.fused_dim),
                hidden_dim=int(st_cfg.hidden_dim),
                num_layers=int(st_cfg.num_layers),
                num_heads=int(st_cfg.num_heads),
                dropout=float(config.model.dropout.st_graph),  # 从统一的DropoutConfig读取
                temporal_window=int(st_cfg.temporal_window),
                use_chunked_attention=bool(getattr(st_cfg, 'use_chunked_attention', True)),
                chunk_size=int(getattr(st_cfg, 'chunk_size', 30)),
                chunk_threshold=int(getattr(st_cfg, 'chunk_threshold', 1000)),
            )
            # Output from ST graph goes directly to aggregation
            self.to_dmodel = nn.Linear(int(st_cfg.hidden_dim), int(tmp_cfg.d_model))
            self.use_st_graph = True
        else:
            logger.info("Using legacy GATv2 + Temporal Transformer")
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
            self.to_dmodel = nn.Linear(int(gat_cfg.hidden_dim), int(tmp_cfg.d_model))
            self.use_st_graph = False

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

        self.use_event_token = bool(getattr(tmp_cfg, "use_event_token", True))
        self.event_type = str(getattr(tmp_cfg, "event_type", "segment"))
        
        if self.use_event_token:
            if self.event_type == "segment":
                # Use hierarchical segment-level event modeling (recommended)
                num_segments = int(getattr(tmp_cfg, "num_segments", 6))
                event_dim = int(getattr(tmp_cfg, "event_dim", 512))
                event_layers = int(getattr(tmp_cfg, "event_num_layers", 1))
                
                logger.info(
                    "Using SegmentEventContext: num_segments=%d, event_dim=%d, layers=%d",
                    num_segments, event_dim, event_layers
                )
                
                self.event_ctx = SegmentEventContext(
                    d_model=int(tmp_cfg.d_model),
                    event_dim=event_dim,
                    num_segments=num_segments,
                    nhead=int(tmp_cfg.nhead),
                    dropout=float(config.model.dropout.temporal),
                    event_num_layers=event_layers,
                )
            else:
                # Legacy frame-level event modeling
                event_layers = int(getattr(tmp_cfg, "event_num_layers", 1))
                logger.info("Using frame-level EventTokenContext (legacy)")
                
                self.event_ctx = EventTokenContext(
                    d_model=int(tmp_cfg.d_model),
                    nhead=int(tmp_cfg.nhead),
                    dim_feedforward=int(tmp_cfg.dim_feedforward),
                    dropout=float(config.model.dropout.temporal),
                    event_num_layers=event_layers,
                )

        self.dual_head = bool(getattr(config.training, "enable_dual_head", False))
        if self.dual_head and not use_st_graph:
            self.self_to_dmodel = nn.Linear(int(feat_cfg.fused_dim), int(tmp_cfg.d_model))
            self.self_scoring = nn.Sequential(
                nn.Linear(int(tmp_cfg.agg_out_dim), int(sc_cfg.hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(config.model.dropout.scoring)),
                nn.Linear(int(sc_cfg.hidden_dim), 1),
            )
            gate_hidden = int(getattr(config.training, "gate_hidden_dim", int(sc_cfg.hidden_dim)))
            gate_dropout = float(config.model.dropout.gate)
            self.gate_mlp = nn.Sequential(
                nn.Linear(int(tmp_cfg.agg_out_dim) * 2, gate_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(gate_dropout),
                nn.Linear(gate_hidden, 1),
            )

        self.scoring = nn.Sequential(
            nn.Linear(int(tmp_cfg.agg_out_dim), int(sc_cfg.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(config.model.dropout.scoring)),
            nn.Linear(int(sc_cfg.hidden_dim), 1),
        )

        logger.info("Initialized ImportanceRanker (vision_dir=%s)", str(dino_cfg.model_dir))

    def forward(
        self,
        frames: Optional[torch.Tensor] = None,
        bboxes: Optional[torch.Tensor] = None,
        person_mask: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
        target_index: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        geometry_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Support two modes: 1) raw data, 2) pre-extracted features
        use_cached = vision_features is not None and geometry_features is not None
        
        if use_cached:
            # Use pre-extracted features
            # Features are saved as (T,N,D) per sample, DataLoader stacks to (B,T,N,D)
            vis_feats = vision_features
            geom_feats = geometry_features
            pm = person_mask
            
            # Get dimensions - features should already be (B, T, N, D) after DataLoader collate
            if vis_feats.ndim == 4 and geom_feats.ndim == 4:
                B, T, N = vis_feats.shape[:3]
            else:
                raise ValueError(f"Unexpected feature shapes: vis_feats={vis_feats.shape}, geom_feats={geom_feats.shape}")
            
            # Ensure person_mask is (B, T, N)
            if pm.ndim == 2:
                # Single sample case: (T, N) -> (1, T, N)
                pm = pm.unsqueeze(0)
                B, T, N = 1, pm.shape[1], pm.shape[2]
            elif pm.shape != (B, T, N):
                raise ValueError(f"person_mask shape {pm.shape} doesn't match features (B={B}, T={T}, N={N})")
        else:
            # Extract features from raw data
            if frames is None or bboxes is None or person_mask is None:
                raise ValueError("Either provide (frames, bboxes, person_mask) or (vision_features, geometry_features)")
            
            B, T, N = person_mask.shape
            fm = frame_mask if frame_mask is not None else person_mask.any(dim=-1)
            pm = person_mask & fm.unsqueeze(-1)

            roi_chunk = int(getattr(self.config.training, "roi_chunk", 256))
            crops = roi_crop_batch(
                frames,
                bboxes,
                pm,
                fm,
                out_size=int(self.config.model.features.dino.image_size),
                roi_chunk=roi_chunk,
            )
            valid_idx = pm.nonzero(as_tuple=False)  # (K,3) => b,t,n

            vis_feats = crops.new_zeros((B, T, N, int(self.config.model.features.dino.feature_dim)))
            if valid_idx.numel() > 0:
                crops_valid = crops[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]  # (K,3,S,S)
                chunk = roi_chunk
                outs = []
                for s in range(0, int(crops_valid.shape[0]), chunk):
                    outs.append(self.vision(crops_valid[s : s + chunk]))
                vis_valid = torch.cat(outs, dim=0)
                vis_feats[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]] = vis_valid.to(dtype=vis_feats.dtype)

            geom_feats = self.geom(bboxes, pm)

        fused = self.fuse(torch.cat([vis_feats, geom_feats], dim=-1)).masked_fill(~pm.unsqueeze(-1), 0.0)

        if self.use_st_graph:
            # New: Spatiotemporal Graph Transformer
            # Unified space-time interaction modeling
            st_features = self.st_graph(fused, pm)  # (B, T, N, hidden_dim)
            
            # Project to d_model
            st_proj = self.to_dmodel(st_features)  # (B, T, N, d_model)
            
            # Optional: Global event context
            if self.use_event_token:
                st_proj = self.event_ctx(st_proj, pm).masked_fill(~pm.unsqueeze(-1), 0.0)
            
            # Video-level aggregation
            rel_pooled = self.temporal_agg(st_proj, pm)  # (B, N, agg_out_dim)
            
            valid_mask = pm.any(dim=1)
            logits = self.scoring(rel_pooled).squeeze(-1)
            
        else:
            # Legacy: GATv2 (spatial) + Temporal Transformer (separate)
            social = self.gat(fused, pm)

            # Transformer 的时间 mask 是"该 person 在该帧是否有效"
            pm_bt = pm.permute(0, 2, 1).reshape(B * N, T)

            rel_tokens = self.temporal_encoder(
                self.to_dmodel(social).permute(0, 2, 1, 3).reshape(B * N, T, -1),
                pm_bt,
            ).reshape(B, N, T, -1).permute(0, 2, 1, 3)  # (B,T,N,D)

            if self.use_event_token:
                rel_tokens = self.event_ctx(rel_tokens, pm).masked_fill(~pm.unsqueeze(-1), 0.0)

            rel_pooled = self.temporal_agg(rel_tokens, pm)  # (B,N,D)

            valid_mask = pm.any(dim=1)

            if self.dual_head:
                self_tokens = self.temporal_encoder(
                    self.self_to_dmodel(fused).permute(0, 2, 1, 3).reshape(B * N, T, -1),
                    pm_bt,
                ).reshape(B, N, T, -1).permute(0, 2, 1, 3)

                self_pooled = self.temporal_agg(self_tokens, pm)
                self_logits = self.self_scoring(self_pooled).squeeze(-1)
                rel_logits = self.scoring(rel_pooled).squeeze(-1)
                gate = torch.sigmoid(self.gate_mlp(torch.cat([self_pooled, rel_pooled], dim=-1)).squeeze(-1))
                logits = (1.0 - gate) * self_logits + gate * rel_logits
            else:
                logits = self.scoring(rel_pooled).squeeze(-1)

        logits = logits.masked_fill(~valid_mask, -1e4)

        scores = torch.softmax(logits / float(self.config.model.scoring.temperature), dim=1) * valid_mask.float()

        out = {
            "importance_logits": logits,
            "importance_scores": scores,
            "video_features": rel_pooled,
        }

        # Only add dual_head outputs if they were computed (legacy branch with dual_head enabled)
        if not self.use_st_graph and self.dual_head:
            out["importance_logits_self"] = self_logits
            out["importance_logits_rel"] = rel_logits
        return out
