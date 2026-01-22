"""Spatiotemporal Graph Transformer for video importance ranking.

Key idea: Model person-person interactions across both space (within frame) and time (across frames).
Instead of separate spatial GATv2 + temporal Transformer, we use a unified spatiotemporal graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpatiotemporalGraphTransformer(nn.Module):
    """Unified spatiotemporal graph modeling with efficient vectorized implementation.
    
    Key improvements:
    1. Vectorized attention (no for loops)
    2. Standard Transformer layer with FFN
    3. Configurable temporal window
    
    Architecture:
        nodes = (person, time), total = T × N nodes
        edges = spatial (same t) + temporal (same person) connections
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.15,
        temporal_window: int = 10,
        use_chunked_attention: bool = True,
        chunk_size: int = 30,
        chunk_threshold: int = 1000,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.temporal_window = temporal_window
        self.use_chunked_attention = use_chunked_attention
        self.chunk_size = chunk_size
        self.chunk_threshold = chunk_threshold
        
        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Spatiotemporal attention layers (with FFN)
        self.st_layers = nn.ModuleList([
            SpatiotemporalAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                temporal_window=temporal_window,
                use_chunked_attention=use_chunked_attention,
                chunk_size=chunk_size,
                chunk_threshold=chunk_threshold,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms (pre-norm style for stability)
        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Feed-forward networks (standard Transformer component)
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        
        # Initialize all weights (critical for gradient stability!)
        self._init_weights()
        
    def forward(
        self,
        features: torch.Tensor,  # (B, T, N, D)
        person_mask: torch.Tensor,  # (B, T, N)
    ) -> torch.Tensor:
        """
        Args:
            features: (B, T, N, D) - fused features
            person_mask: (B, T, N) - valid person mask
            
        Returns:
            (B, T, N, hidden_dim) - spatiotemporal enhanced features
        """
        B, T, N, D = features.shape
        
        # Project to hidden dim
        x = self.input_proj(features)  # (B, T, N, hidden_dim)
        
        # Apply spatiotemporal attention layers with FFN (pre-norm style)
        for layer_idx in range(self.num_layers):
            st_layer = self.st_layers[layer_idx]
            norm1 = self.norms1[layer_idx]
            norm2 = self.norms2[layer_idx]
            ffn = self.ffns[layer_idx]
            
            # Pre-norm + Attention + Residual
            attn_input = norm1(x)
            attn_output = st_layer(attn_input, person_mask)
            x = x + attn_output
            x = x.masked_fill(~person_mask.unsqueeze(-1), 0.0)
            
            # Pre-norm + FFN + Residual
            ffn_input = norm2(x)
            ffn_output = ffn(ffn_input)
            x = x + ffn_output
            x = x.masked_fill(~person_mask.unsqueeze(-1), 0.0)
        
        return x
    
    def _init_weights(self):
        """Initialize weights for gradient stability.
        
        Critical: Without proper initialization, gradients explode (>20,000).
        
        Strategy:
        1. Xavier uniform for most linear layers
        2. Smaller init for FFN output layers (scale down residual contribution)
        3. Zero init for LayerNorm biases
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this is an FFN output layer (maps 4*hidden -> hidden)
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    if module.in_features == self.hidden_dim * 4 and module.out_features == self.hidden_dim:
                        # FFN output layer: use smaller scale
                        # Scale down by 1/sqrt(num_layers) to prevent gradient explosion in deep networks
                        nn.init.xavier_uniform_(module.weight, gain=1.0 / (self.num_layers ** 0.5))
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                    else:
                        # Regular Xavier for other layers
                        nn.init.xavier_uniform_(module.weight, gain=1.0)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    

class SpatiotemporalAttentionLayer(nn.Module):
    """Vectorized spatiotemporal attention layer.
    
    Efficiently computes attention over both spatial and temporal dimensions
    using einsum operations without explicit for loops.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        temporal_window: int,
        use_chunked_attention: bool,
        chunk_size: int,
        chunk_threshold: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temporal_window = temporal_window
        self.use_chunked_attention = use_chunked_attention
        self.chunk_size = chunk_size
        self.chunk_threshold = chunk_threshold
        
        assert self.head_dim * num_heads == hidden_dim, f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,  # (B, T, N, D)
        person_mask: torch.Tensor,  # (B, T, N)
    ) -> torch.Tensor:
        """Vectorized spatiotemporal attention with memory-efficient chunked computation.
        
        Strategy:
        1. Process attention in temporal chunks to reduce memory
        2. Build attention mask efficiently
        3. Single multi-head attention pass per chunk
        """
        B, T, N, D = x.shape
        device = x.device
        
        # QKV projection
        qkv = self.qkv(x)  # (B, T, N, 3*D)
        qkv = qkv.reshape(B, T, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(3)  # Each: (B, T, N, num_heads, head_dim)
        
        # For small T*N, use direct attention; for large, use chunked (based on config)
        TN = T * N
        use_chunked = self.use_chunked_attention and (TN > self.chunk_threshold)
        
        if not use_chunked:
            # Direct attention for small sequences
            return self._direct_attention(q, k, v, person_mask, T, N, device)
        else:
            # Chunked attention for large sequences (memory efficient)
            return self._chunked_attention(q, k, v, person_mask, T, N, device)
    
    def _direct_attention(self, q, k, v, person_mask, T, N, device):
        """Direct full attention (for small sequences)."""
        B = q.shape[0]
        TN = T * N
        
        # Reshape for batch attention: (B, num_heads, T*N, head_dim)
        q = q.permute(0, 3, 1, 2, 4).reshape(B, self.num_heads, TN, self.head_dim)
        k = k.permute(0, 3, 1, 2, 4).reshape(B, self.num_heads, TN, self.head_dim)
        v = v.permute(0, 3, 1, 2, 4).reshape(B, self.num_heads, TN, self.head_dim)
        
        # Build spatiotemporal attention mask
        attn_mask = self._build_spatiotemporal_mask(person_mask, T, N, device)
        attn_mask = attn_mask.unsqueeze(1)  # (B, 1, TN, TN)
        
        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Critical fix: When a node has no valid attention targets, softmax(-inf, -inf, ...) = NaN
        # Replace NaN with 0.0 to effectively ignore these invalid nodes
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention
        out = torch.matmul(attn_weights, v)
        
        # Reshape back
        out = out.reshape(B, self.num_heads, T, N, self.head_dim)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, T, N, self.hidden_dim)
        
        # Output projection
        out = self.out_proj(out)
        out = out.masked_fill(~person_mask.unsqueeze(-1), 0.0)
        
        return out
    
    def _chunked_attention(self, q, k, v, person_mask, T, N, device):
        """Chunked attention to reduce memory (for large sequences).
        
        Process attention in temporal chunks of size chunk_size.
        Each chunk attends to its temporal window.
        """
        B = q.shape[0]
        chunk_size = self.chunk_size  # From config
        
        out_list = []
        
        for t_start in range(0, T, chunk_size):
            t_end = min(t_start + chunk_size, T)
            
            # Query chunk
            q_chunk = q[:, t_start:t_end]  # (B, chunk_T, N, H, D)
            
            # Key/Value window (include temporal context)
            kv_start = max(0, t_start - self.temporal_window)
            kv_end = min(T, t_end + self.temporal_window)
            k_window = k[:, kv_start:kv_end]  # (B, window_T, N, H, D)
            v_window = v[:, kv_start:kv_end]  # (B, window_T, N, H, D)
            
            chunk_T = t_end - t_start
            window_T = kv_end - kv_start
            
            # Reshape
            q_flat = q_chunk.permute(0, 3, 1, 2, 4).reshape(B, self.num_heads, chunk_T * N, self.head_dim)
            k_flat = k_window.permute(0, 3, 1, 2, 4).reshape(B, self.num_heads, window_T * N, self.head_dim)
            v_flat = v_window.permute(0, 3, 1, 2, 4).reshape(B, self.num_heads, window_T * N, self.head_dim)
            
            # Build local mask
            mask_chunk = person_mask[:, t_start:t_end]  # (B, chunk_T, N)
            mask_window = person_mask[:, kv_start:kv_end]  # (B, window_T, N)
            
            # Simplified mask: only spatial connections within chunk (ignore long-range temporal)
            attn_mask = self._build_local_mask(mask_chunk, mask_window, chunk_T, window_T, N, t_start, kv_start, device)
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, chunk_T*N, window_T*N)
            
            # Attention
            attn_scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # Fix NaN from all-masked rows
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
            
            attn_weights = self.attn_dropout(attn_weights)
            
            # Apply
            out_chunk = torch.matmul(attn_weights, v_flat)
            out_chunk = out_chunk.reshape(B, self.num_heads, chunk_T, N, self.head_dim)
            out_chunk = out_chunk.permute(0, 2, 3, 1, 4).reshape(B, chunk_T, N, self.hidden_dim)
            
            out_list.append(out_chunk)
        
        # Concatenate chunks
        out = torch.cat(out_list, dim=1)  # (B, T, N, hidden_dim)
        
        # Output projection
        out = self.out_proj(out)
        out = out.masked_fill(~person_mask.unsqueeze(-1), 0.0)
        
        return out
    
    def _build_local_mask(self, mask_chunk, mask_window, chunk_T, window_T, N, t_start, kv_start, device):
        """Build local attention mask for chunked computation."""
        B = mask_chunk.shape[0]
        chunk_TN = chunk_T * N
        window_TN = window_T * N
        
        # Flatten masks
        mask_chunk_flat = mask_chunk.reshape(B, chunk_TN)
        mask_window_flat = mask_window.reshape(B, window_TN)
        
        # Create indices
        t_chunk = torch.arange(chunk_T, device=device).repeat_interleave(N) + t_start
        n_chunk = torch.arange(N, device=device).repeat(chunk_T)
        
        t_window = torch.arange(window_T, device=device).repeat_interleave(N) + kv_start
        n_window = torch.arange(N, device=device).repeat(window_T)
        
        # Compute connections
        t_diff = (t_chunk.unsqueeze(1) - t_window.unsqueeze(0)).abs()  # (chunk_TN, window_TN)
        n_same = (n_chunk.unsqueeze(1) == n_window.unsqueeze(0))  # (chunk_TN, window_TN)
        t_same = (t_chunk.unsqueeze(1) == t_window.unsqueeze(0))  # (chunk_TN, window_TN)
        
        # Spatial + temporal connections
        spatial_mask = t_same & (~n_same)
        temporal_mask = n_same & (t_diff <= self.temporal_window) & (t_diff > 0)
        self_mask = (t_chunk.unsqueeze(1) == t_window.unsqueeze(0)) & (n_chunk.unsqueeze(1) == n_window.unsqueeze(0))
        
        connection_mask = spatial_mask | temporal_mask | self_mask
        
        # Apply validity
        valid_src = mask_chunk_flat.unsqueeze(2)  # (B, chunk_TN, 1)
        valid_tgt = mask_window_flat.unsqueeze(1)  # (B, 1, window_TN)
        valid_pair = valid_src & valid_tgt  # (B, chunk_TN, window_TN)
        
        mask = connection_mask.unsqueeze(0) & valid_pair
        
        return mask
    
    def _build_spatiotemporal_mask(
        self,
        person_mask: torch.Tensor,  # (B, T, N)
        T: int,
        N: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build attention mask for spatiotemporal connections.
        
        Mask[b, i, j] = True if node i can attend to node j
        where i, j are flattened indices in [0, T*N)
        
        Node index: idx = t * N + n
        
        Connections:
        - Spatial: same t, different n (if both valid)
        - Temporal: same n, |t1 - t2| <= temporal_window (if both valid)
        """
        B = person_mask.shape[0]
        TN = T * N
        
        # Initialize mask as False (no connections)
        mask = torch.zeros(B, TN, TN, dtype=torch.bool, device=device)
        
        # Flatten person_mask: (B, T*N)
        person_mask_flat = person_mask.reshape(B, TN)
        
        # Create t and n indices for all T*N nodes
        t_indices = torch.arange(T, device=device).repeat_interleave(N)  # [0,0,...,0, 1,1,...,1, ..., T-1,...,T-1]
        n_indices = torch.arange(N, device=device).repeat(T)  # [0,1,...,N-1, 0,1,...,N-1, ...]
        
        # Compute pairwise t and n differences (broadcast)
        t_diff = (t_indices.unsqueeze(0) - t_indices.unsqueeze(1)).abs()  # (TN, TN)
        n_same = (n_indices.unsqueeze(0) == n_indices.unsqueeze(1))  # (TN, TN)
        t_same = (t_indices.unsqueeze(0) == t_indices.unsqueeze(1))  # (TN, TN)
        
        # Spatial connections: same t, different n
        spatial_mask = t_same & (~n_same)  # (TN, TN)
        
        # Temporal connections: same n, within window
        temporal_mask = n_same & (t_diff <= self.temporal_window) & (t_diff > 0)  # (TN, TN)
        
        # Self-connection
        self_mask = torch.eye(TN, dtype=torch.bool, device=device)  # (TN, TN)
        
        # Combine: (TN, TN)
        connection_mask = spatial_mask | temporal_mask | self_mask
        
        # Apply person validity: both nodes must be valid (vectorized for all batches)
        valid_src = person_mask_flat.unsqueeze(2)  # (B, TN, 1)
        valid_tgt = person_mask_flat.unsqueeze(1)  # (B, 1, TN)
        valid_pair = valid_src & valid_tgt  # (B, TN, TN)
        
        # Broadcast connection_mask to all batches and apply validity
        mask = connection_mask.unsqueeze(0) & valid_pair  # (B, TN, TN)
        
        return mask
