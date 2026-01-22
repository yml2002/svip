"""Segment-level event context module.

Hierarchical event modeling: video → segments → events
Provides global event-level semantic understanding as complement to
local spatiotemporal interactions from ST-Graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SegmentEventContext(nn.Module):
    """Hierarchical segment-level event modeling.
    
    Design philosophy:
    - ST-Graph provides bottom-up local spatiotemporal interactions
    - Segment Events provide top-down global semantic understanding
    - Fusion: individual features + event context = context-aware importance
    
    Architecture:
    1. Split video into K segments (e.g., 120 frames → 6 segments of 20 frames)
    2. Per-segment event extraction via cross-attention
    3. Temporal event modeling to capture event evolution
    4. Event-aware feature enhancement for each person
    
    Args:
        d_model: person feature dimension (from ST-Graph output)
        event_dim: event token dimension
        num_segments: number of segments to split video into
        nhead: number of attention heads for event extraction
        dropout: dropout rate
        event_num_layers: number of transformer layers for event temporal modeling
    """
    
    def __init__(
        self,
        d_model: int,
        event_dim: int = 512,
        num_segments: int = 6,
        nhead: int = 8,
        dropout: float = 0.1,
        event_num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.event_dim = event_dim
        self.num_segments = num_segments
        self.nhead = nhead
        
        # Learnable segment event queries
        self.event_queries = nn.Parameter(torch.zeros(1, num_segments, event_dim))
        nn.init.normal_(self.event_queries, std=0.02)
        
        # Cross-attention for event extraction (query events from person features)
        self.event_extraction = nn.MultiheadAttention(
            embed_dim=event_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        
        # Project person features to event_dim for cross-attention
        self.person_to_event = nn.Linear(d_model, event_dim)
        
        # Temporal transformer for event sequence modeling
        event_layer = nn.TransformerEncoderLayer(
            d_model=event_dim,
            nhead=nhead,
            dim_feedforward=event_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.event_temporal = nn.TransformerEncoder(
            event_layer,
            num_layers=event_num_layers,
        )
        
        # Fusion: combine person features with event context
        # Use adaptive fusion with learned attention weights
        self.fusion = nn.Sequential(
            nn.Linear(d_model + event_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Attention-based fusion gate (learn importance of event context per person)
        self.fusion_gate = nn.Linear(d_model + event_dim, 1)
        # Gate初始化为0: sigmoid(0) = 0.5, 平衡event和residual
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)
    
    def forward(
        self,
        person_features: torch.Tensor,  # (B, T, N, d_model) from ST-Graph
        person_mask: torch.Tensor,      # (B, T, N) bool
    ) -> torch.Tensor:
        """Apply segment-level event context to person features.
        
        Args:
            person_features: (B, T, N, d_model) person features from ST-Graph
            person_mask: (B, T, N) bool mask indicating valid persons
            
        Returns:
            enhanced_features: (B, T, N, d_model) event-aware person features
        """
        B, T, N, D = person_features.shape
        K = self.num_segments
        orig_T = T  # ← 修复: 记录原始T
        
        # Calculate frames per segment
        T_seg = T // K
        if T % K != 0:
            # Handle uneven division by padding
            pad_frames = K - (T % K)
            person_features = torch.cat([
                person_features,
                person_features.new_zeros(B, pad_frames, N, D)
            ], dim=1)
            person_mask = torch.cat([
                person_mask,
                person_mask.new_zeros(B, pad_frames, N)
            ], dim=1)
            T = T + pad_frames
            T_seg = T // K
        
        # Step 1: Reshape to segments (B, K, T_seg, N, D)
        person_seg = person_features.reshape(B, K, T_seg, N, D)
        mask_seg = person_mask.reshape(B, K, T_seg, N)
        
        # Step 2: Extract segment events via cross-attention
        # For each segment, attend to all person features in that segment
        events = []
        for k in range(K):
            # Flatten persons and frames in this segment
            seg_features = person_seg[:, k].reshape(B, T_seg * N, D)  # (B, T_seg*N, D)
            seg_mask = mask_seg[:, k].reshape(B, T_seg * N)  # (B, T_seg*N)
            
            # Project to event dimension
            seg_features_proj = self.person_to_event(seg_features)  # (B, T_seg*N, event_dim)
            
            # Query: learnable event query for this segment
            query = self.event_queries[:, k:k+1, :].expand(B, 1, self.event_dim)  # (B, 1, event_dim)
            
            # Cross-attention: event queries attend to person features
            event_k, _ = self.event_extraction(
                query,
                seg_features_proj,
                seg_features_proj,
                key_padding_mask=~seg_mask,
                need_weights=False,
            )  # (B, 1, event_dim)
            
            events.append(event_k)
        
        # Stack events: (B, K, event_dim)
        events = torch.cat(events, dim=1)  # (B, K, event_dim)
        
        # Step 3: Temporal event modeling
        # Model event evolution across segments
        events = self.event_temporal(events)  # (B, K, event_dim)
        
        # Step 4: Broadcast events back to persons and fuse
        # Each person in segment k gets event_k
        enhanced_features = []
        for k in range(K):
            seg_features = person_seg[:, k]  # (B, T_seg, N, D)
            event_k = events[:, k:k+1, :].unsqueeze(2)  # (B, 1, 1, event_dim)
            event_k = event_k.expand(B, T_seg, N, self.event_dim)  # (B, T_seg, N, event_dim)
            
            # Concatenate person features with event context
            combined = torch.cat([seg_features, event_k], dim=-1)  # (B, T_seg, N, D+event_dim)
            
            # Adaptive fusion with gating
            gate = torch.sigmoid(self.fusion_gate(combined))  # (B, T_seg, N, 1)
            fused = self.fusion(combined)  # (B, T_seg, N, D)
            
            # Residual connection with learned gate
            # gate ≈ 0.5 initially, model learns optimal blending
            enhanced = seg_features + gate * fused
            
            enhanced_features.append(enhanced)
        
        # Concatenate back to (B, T, N, D)
        enhanced_features = torch.cat(enhanced_features, dim=1)  # (B, K*T_seg, N, D)
        
        # Remove padding if added
        if T > orig_T:  # ← 修复: 使用原始T
            enhanced_features = enhanced_features[:, :orig_T]
        
        # Apply person mask
        enhanced_features = enhanced_features.masked_fill(~person_mask[:, :orig_T].unsqueeze(-1), 0.0)
        
        return enhanced_features
