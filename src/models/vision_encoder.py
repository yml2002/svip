"""Local (offline) vision encoder.

Input: crops (B,3,H,W) float in [0,1]
Output: (B,D) L2-normalized
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    def __init__(
        self,
        model_dir: str,
        out_dim: Optional[int] = None,
        image_size: int = 224,
        freeze: bool = True,
        unfreeze_layers: int = 0,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.model_dir = str(model_dir)
        self.initial_freeze = bool(freeze)
        self.unfreeze_layers = int(max(0, unfreeze_layers))
        self._stage_signature: Optional[tuple[int, int, str]] = None

        if not self.model_dir:
            raise ValueError(
                "VisionEncoder model_dir must be provided (offline-only). "
                "Example: data/models/dinov2-base"
            )

        from transformers import AutoModel  # type: ignore

        self.backbone = AutoModel.from_pretrained(self.model_dir, local_files_only=True)

        mean = getattr(self.backbone.config, "image_mean", None)
        std = getattr(self.backbone.config, "image_std", None)
        if mean is None or std is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_image_mean", mean_t, persistent=False)
        self.register_buffer("_image_std", std_t, persistent=False)

        hidden = getattr(self.backbone.config, "hidden_size", None)
        if hidden is None:
            raise RuntimeError("Cannot infer hidden_size from vision backbone config.")
        self.backbone_dim = int(hidden)

        self._proj: Optional[nn.Linear] = None
        if out_dim is not None and int(out_dim) != self.backbone_dim:
            self._proj = nn.Linear(self.backbone_dim, int(out_dim), bias=False)
            self.out_dim = int(out_dim)
        else:
            self.out_dim = self.backbone_dim

        self.freeze_backbone()
        if (not self.initial_freeze) and self.unfreeze_layers > 0:
            self.apply_finetune_mode(self.unfreeze_layers, mode="full_block")

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _encoder_layers(self):
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return list(self.backbone.encoder.layer)
        return []

    def apply_finetune_mode(self, unfreeze_layers: int, mode: str = "attn_ln") -> None:
        self.freeze_backbone()
        if unfreeze_layers <= 0 or str(mode) == "frozen":
            return

        layers = self._encoder_layers()
        if not layers:
            return
        selected = layers[max(0, len(layers) - int(unfreeze_layers)):]

        for layer in selected:
            if str(mode) == "full_block":
                for p in layer.parameters():
                    p.requires_grad = True
                continue

            for name, p in layer.named_parameters():
                key = str(name).lower()
                if any(tok in key for tok in ("attention", "attn", "query", "key", "value", "qkv", "norm", "layernorm")):
                    p.requires_grad = True

        if hasattr(self.backbone, "layernorm"):
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True

    def configure_train_stage(self, epoch: int, *, warmup_epochs: int, train_mode: str) -> None:
        signature = (int(epoch), int(warmup_epochs), str(train_mode))
        if signature == self._stage_signature:
            return
        if int(epoch) <= int(warmup_epochs):
            self.freeze_backbone()
        else:
            self.apply_finetune_mode(self.unfreeze_layers, mode=str(train_mode))
        self._stage_signature = signature

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        x = crops
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        mean = self._image_mean.to(device=x.device, dtype=x.dtype)
        std = self._image_std.to(device=x.device, dtype=x.dtype)
        pixel_values = (x - mean) / std

        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            raise RuntimeError("Vision backbone did not return last_hidden_state.")

        feats = last_hidden[:, 0, :]  # CLS
        if self._proj is not None:
            feats = self._proj(feats)

        feats = feats.to(dtype=crops.dtype)
        feats = F.normalize(feats, dim=-1)
        return feats
