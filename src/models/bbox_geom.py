"""BBox geometric feature encoder + edge feature computation."""

from __future__ import annotations

import torch
import torch.nn as nn


class BBoxGeomEncoder(nn.Module):
    """Encode static spatial features from (x1,y1,x2,y2).

    Input:
        bboxes: (B,T,N,4) normalized
        person_mask: (B,T,N)
    Output:
        geom: (B,T,N,D)

    Note: temporal motion features (dcx/dcy/disp/speed/accel) are intentionally
    excluded — motion information is exclusively carried by GAT temporal edge features,
    achieving clean separation of responsibilities.
    """

    def __init__(self, out_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        in_dim = 9
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, bboxes: torch.Tensor, person_mask: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = bboxes.unbind(dim=-1)
        w = (x2 - x1).clamp(min=0.0)
        h = (y2 - y1).clamp(min=0.0)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        area = w * h

        feats = torch.stack([x1, y1, x2, y2, cx, cy, w, h, area], dim=-1)
        out = self.mlp(feats)
        out = out.masked_fill(~person_mask.unsqueeze(-1), 0.0)
        return out
