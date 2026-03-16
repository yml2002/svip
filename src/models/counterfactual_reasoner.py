"""Relation / counterfactual branches with sparse MoE routing."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    masked = logits.masked_fill(~mask, -1e4)
    weights = torch.softmax(masked, dim=dim)
    weights = weights * mask.to(dtype=weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp(min=1e-6)


class SparseMoEAdapter(nn.Module):
    """Token-level sparse MoE adapter with top-k expert routing."""

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_experts: int,
        topk_experts: int,
        dispatch_mode: str,
        router_temperature: float,
        router_noise_std: float,
        capacity_factor: float,
        drop_tokens: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.topk_experts = max(1, min(int(topk_experts), self.num_experts))
        self.dispatch_mode = str(dispatch_mode).strip().lower()
        self.router_temperature = float(router_temperature)
        self.router_noise_std = float(router_noise_std)
        self.capacity_factor = float(max(capacity_factor, 0.1))
        self.drop_tokens = bool(drop_tokens)
        self.out_dim = int(out_dim)

        self.router = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), self.num_experts),
        )

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(in_dim), int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_dim), int(out_dim)),
                    nn.LayerNorm(int(out_dim)),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B,N,C), valid_mask: (B,N)
        if self.dispatch_mode != "sparse":
            return self._forward_dense(x, valid_mask)
        return self._forward_sparse(x, valid_mask)

    def _forward_dense(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape

        router_logits = self.router(x) / max(self.router_temperature, 1e-6)
        if self.training and self.router_noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise_std

        router_probs = torch.softmax(router_logits, dim=-1)
        if self.topk_experts < self.num_experts:
            topk_val, topk_idx = router_probs.topk(k=self.topk_experts, dim=-1)
            sparse = torch.zeros_like(router_probs)
            sparse.scatter_(-1, topk_idx, topk_val)
            router_probs = sparse / sparse.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)  # (B,N,E,D)
        mixed = (expert_outputs * router_probs.unsqueeze(-1)).sum(dim=-2)

        mask_f = valid_mask.to(dtype=mixed.dtype)
        mixed = mixed * mask_f.unsqueeze(-1)
        router_probs = router_probs * mask_f.unsqueeze(-1)
        router_probs = router_probs / router_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        valid_probs = router_probs[valid_mask]
        valid_logits = router_logits[valid_mask]

        if int(valid_probs.numel()) > 0:
            entropy = -(valid_probs.clamp(min=1e-8) * valid_probs.clamp(min=1e-8).log()).sum(dim=-1).mean()
            importance = valid_probs.mean(dim=0)
            load = (valid_probs > 0).to(dtype=valid_probs.dtype).mean(dim=0)
            load = load / load.sum().clamp(min=1e-6)
            router_z = (torch.logsumexp(valid_logits, dim=-1) ** 2).mean()
        else:
            entropy = x.new_tensor(0.0)
            importance = x.new_zeros((self.num_experts,))
            load = x.new_zeros((self.num_experts,))
            router_z = x.new_tensor(0.0)

        uniform = float(1.0 / self.num_experts)
        load_balance = (
            ((importance - uniform) ** 2).mean() + ((load - uniform) ** 2).mean()
        ) * float(self.num_experts)

        dropped_ratio = x.new_tensor(0.0)
        return mixed.view(B, N, self.out_dim), router_probs.view(B, N, self.num_experts), entropy, load_balance, router_z, dropped_ratio

    def _forward_sparse(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        flat_x = x.reshape(B * N, -1)
        flat_valid = valid_mask.reshape(B * N)

        mixed_flat = flat_x.new_zeros((B * N, self.out_dim))
        router_probs_flat = flat_x.new_zeros((B * N, self.num_experts))

        if not bool(flat_valid.any()):
            zero = x.new_tensor(0.0)
            return (
                mixed_flat.view(B, N, self.out_dim),
                router_probs_flat.view(B, N, self.num_experts),
                zero,
                zero,
                zero,
                zero,
            )

        valid_idx = flat_valid.nonzero(as_tuple=False).squeeze(1)
        x_valid = flat_x[valid_idx]
        token_count = int(x_valid.shape[0])

        router_logits = self.router(x_valid) / max(self.router_temperature, 1e-6)
        if self.training and self.router_noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise_std

        router_probs_dense = torch.softmax(router_logits, dim=-1)
        topk_val, topk_idx = router_probs_dense.topk(k=self.topk_experts, dim=-1)
        topk_val = topk_val / topk_val.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        capacity = max(
            1,
            int(math.ceil(self.capacity_factor * token_count * self.topk_experts / max(1, self.num_experts))),
        )

        mixed_valid = x_valid.new_zeros((token_count, self.out_dim))
        gate_sum = x_valid.new_zeros((token_count,))
        router_probs_valid = x_valid.new_zeros((token_count, self.num_experts))
        assigned_count = x_valid.new_zeros((self.num_experts,))

        dropped = 0
        total_assignments = token_count * self.topk_experts

        for expert_idx, expert in enumerate(self.experts):
            routed = (topk_idx == expert_idx).nonzero(as_tuple=False)
            if int(routed.numel()) == 0:
                continue

            token_ids = routed[:, 0]
            slot_ids = routed[:, 1]
            token_gates = topk_val[token_ids, slot_ids]

            if self.drop_tokens and int(token_ids.numel()) > capacity:
                keep_order = token_gates.topk(k=capacity, largest=True).indices
                dropped += int(token_ids.numel()) - capacity
                token_ids = token_ids[keep_order]
                token_gates = token_gates[keep_order]

            assigned_count[expert_idx] = float(token_ids.numel())
            if int(token_ids.numel()) == 0:
                continue

            expert_out = expert(x_valid[token_ids])
            mixed_valid.index_add_(0, token_ids, expert_out * token_gates.unsqueeze(1))
            gate_sum.index_add_(0, token_ids, token_gates)

            onehot = x_valid.new_zeros((int(token_ids.numel()), self.num_experts))
            onehot[:, expert_idx] = token_gates
            router_probs_valid.index_add_(0, token_ids, onehot)

        gate_norm = gate_sum.clamp(min=1e-6)
        mixed_valid = mixed_valid / gate_norm.unsqueeze(1)
        router_probs_valid = router_probs_valid / gate_norm.unsqueeze(1)

        mixed_flat[valid_idx] = mixed_valid
        router_probs_flat[valid_idx] = router_probs_valid

        entropy = -(router_probs_valid.clamp(min=1e-8) * router_probs_valid.clamp(min=1e-8).log()).sum(dim=-1)
        router_entropy = entropy.mean() if int(entropy.numel()) > 0 else x.new_tensor(0.0)

        importance = router_probs_dense.mean(dim=0)
        load = assigned_count / assigned_count.sum().clamp(min=1e-6)
        uniform = float(1.0 / self.num_experts)
        load_balance = (
            ((importance - uniform) ** 2).mean() + ((load - uniform) ** 2).mean()
        ) * float(self.num_experts)

        router_z = (torch.logsumexp(router_logits, dim=-1) ** 2).mean()
        dropped_ratio = x.new_tensor(float(dropped) / float(max(1, total_assignments)))

        return (
            mixed_flat.view(B, N, self.out_dim),
            router_probs_flat.view(B, N, self.num_experts),
            router_entropy,
            load_balance,
            router_z,
            dropped_ratio,
        )


class RelationMoEReasoner(nn.Module):
    """Build person-discriminative relation logits via sparse MoE experts."""

    def __init__(
        self,
        *,
        token_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_experts: int,
        topk_experts: int,
        dispatch_mode: str,
        router_temperature: float,
        router_noise_std: float,
        capacity_factor: float,
        drop_tokens: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        in_dim = int(token_dim) * 4
        self.temporal_attn = nn.Sequential(
            nn.LayerNorm(int(token_dim) * 3),
            nn.Linear(int(token_dim) * 3, 1),
        )
        self.moe = SparseMoEAdapter(
            in_dim=in_dim,
            out_dim=int(out_dim),
            hidden_dim=int(hidden_dim),
            num_experts=int(num_experts),
            topk_experts=int(topk_experts),
            dispatch_mode=str(dispatch_mode),
            router_temperature=float(router_temperature),
            router_noise_std=float(router_noise_std),
            capacity_factor=float(capacity_factor),
            drop_tokens=bool(drop_tokens),
            dropout=float(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(int(out_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, tokens: torch.Tensor, person_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # tokens: (B,T,N,D), person_mask: (B,T,N)
        valid_mask = person_mask.any(dim=1)
        mask_f = person_mask.to(dtype=tokens.dtype)

        frame_sum = (tokens * mask_f.unsqueeze(-1)).sum(dim=2)  # (B,T,D)
        frame_count = mask_f.sum(dim=2, keepdim=True).clamp(min=1.0)
        frame_mean = frame_sum / frame_count

        other_sum = frame_sum.unsqueeze(2) - tokens * mask_f.unsqueeze(-1)
        other_count = (frame_count.unsqueeze(2) - mask_f.unsqueeze(-1)).clamp(min=1.0)
        other_mean = other_sum / other_count

        token_triplet = torch.cat([tokens, other_mean, tokens - other_mean], dim=-1)
        attn_logits = self.temporal_attn(token_triplet).squeeze(-1)
        attn = _masked_softmax(attn_logits, person_mask, dim=1)

        self_summary = (tokens * attn.unsqueeze(-1)).sum(dim=1)
        other_summary = (other_mean * attn.unsqueeze(-1)).sum(dim=1)
        event_summary = (frame_mean.unsqueeze(2) * attn.unsqueeze(-1)).sum(dim=1)

        relation_input = torch.cat(
            [self_summary, other_summary, event_summary, self_summary - other_summary],
            dim=-1,
        )
        (
            relation_feat,
            router_probs,
            router_entropy,
            router_load_balance,
            router_z,
            dropped_ratio,
        ) = self.moe(relation_input, valid_mask)
        relation_logits = self.head(relation_feat).squeeze(-1).masked_fill(~valid_mask, -1e4)

        return {
            "relation_features": relation_feat,
            "relation_logits": relation_logits,
            "relation_router_probs": router_probs,
            "relation_router_entropy": router_entropy,
            "relation_load_balance_loss": router_load_balance,
            "relation_router_z_loss": router_z,
            "relation_dropped_ratio": dropped_ratio,
            "relation_attention": attn,
        }


class CounterfactualMoEReasoner(nn.Module):
    """Estimate person-level counterfactual impact with sparse MoE experts."""

    def __init__(
        self,
        *,
        token_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_experts: int,
        topk_experts: int,
        dispatch_mode: str,
        router_temperature: float,
        router_noise_std: float,
        capacity_factor: float,
        drop_tokens: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        in_dim = int(token_dim) * 4
        self.temporal_attn = nn.Sequential(
            nn.LayerNorm(int(token_dim) * 2),
            nn.Linear(int(token_dim) * 2, 1),
        )
        self.moe = SparseMoEAdapter(
            in_dim=in_dim,
            out_dim=int(out_dim),
            hidden_dim=int(hidden_dim),
            num_experts=int(num_experts),
            topk_experts=int(topk_experts),
            dispatch_mode=str(dispatch_mode),
            router_temperature=float(router_temperature),
            router_noise_std=float(router_noise_std),
            capacity_factor=float(capacity_factor),
            drop_tokens=bool(drop_tokens),
            dropout=float(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(int(out_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, tokens: torch.Tensor, person_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # tokens: (B,T,N,D), person_mask: (B,T,N)
        valid_mask = person_mask.any(dim=1)
        mask_f = person_mask.to(dtype=tokens.dtype)

        frame_sum = (tokens * mask_f.unsqueeze(-1)).sum(dim=2)  # (B,T,D)
        frame_count = mask_f.sum(dim=2, keepdim=True).clamp(min=1.0)
        frame_mean = frame_sum / frame_count

        other_sum = frame_sum.unsqueeze(2) - tokens * mask_f.unsqueeze(-1)
        other_count = (frame_count.unsqueeze(2) - mask_f.unsqueeze(-1)).clamp(min=1.0)
        counterfactual_frame = other_sum / other_count

        attn_in = torch.cat([tokens, frame_mean.unsqueeze(2).expand_as(tokens)], dim=-1)
        attn_logits = self.temporal_attn(attn_in).squeeze(-1)
        attn = _masked_softmax(attn_logits, person_mask, dim=1)

        self_summary = (tokens * attn.unsqueeze(-1)).sum(dim=1)
        factual_event = (frame_mean.unsqueeze(2) * attn.unsqueeze(-1)).sum(dim=1)
        counterfactual_event = (counterfactual_frame * attn.unsqueeze(-1)).sum(dim=1)
        delta = factual_event - counterfactual_event

        cf_input = torch.cat([self_summary, factual_event, counterfactual_event, delta], dim=-1)
        (
            cf_feat,
            router_probs,
            router_entropy,
            router_load_balance,
            router_z,
            dropped_ratio,
        ) = self.moe(cf_input, valid_mask)
        cf_logits = self.head(cf_feat).squeeze(-1).masked_fill(~valid_mask, -1e4)

        return {
            "counterfactual_features": cf_feat,
            "counterfactual_delta": delta,
            "counterfactual_logits": cf_logits,
            "counterfactual_router_probs": router_probs,
            "counterfactual_router_entropy": router_entropy,
            "counterfactual_load_balance_loss": router_load_balance,
            "counterfactual_router_z_loss": router_z,
            "counterfactual_dropped_ratio": dropped_ratio,
            "counterfactual_attention": attn,
            "event_state": factual_event,
        }

