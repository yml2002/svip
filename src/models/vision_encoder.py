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
        finetune_layers: int = 0,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.model_dir = str(model_dir)

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

        # 微调策略: freeze=False + finetune_layers > 0 表示部分层微调
        if freeze:
            # 完全冻结
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif finetune_layers > 0:
            # 部分微调: 冻结前面层,仅微调最后 finetune_layers 层
            # DINOv2 结构: encoder.layer[0..11] (12层)
            for p in self.backbone.parameters():
                p.requires_grad = False  # 先全冻结
            
            # 解冻最后几层
            if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
                total_layers = len(self.backbone.encoder.layer)
                unfreeze_from = max(0, total_layers - finetune_layers)
                for i in range(unfreeze_from, total_layers):
                    for p in self.backbone.encoder.layer[i].parameters():
                        p.requires_grad = True
                print(f"[VisionEncoder] 微调最后 {finetune_layers}/{total_layers} 层 (layer {unfreeze_from}..{total_layers-1})")
            else:
                print("[VisionEncoder] 警告: 无法识别 encoder.layer 结构,启用全模型微调")
                for p in self.backbone.parameters():
                    p.requires_grad = True
        # else: freeze=False and finetune_layers=0 → 全模型微调

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
