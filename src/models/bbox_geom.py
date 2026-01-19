"""BBox geometric feature encoder.

Moved from `src/features/bbox_geom.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BBoxGeomEncoder(nn.Module):
    """Encode (x1,y1,x2,y2) + area + center + frame-to-frame displacement.

    Input:
        bboxes: (B,T,N,4) normalized
        person_mask: (B,T,N)
    Output:
        geom: (B,T,N,D)
    """

    def __init__(self, out_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        # Features are normalized to [0,1] space; deltas are in the same normalized space.
        # We explicitly include (dcx, dcy) so docs can truthfully claim Δx/Δy.
        #
        #   base: x1,y1,x2,y2,cx,cy,w,h,area
        #   motion: dcx,dcy,disp, speed, accel
        in_dim = 14
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, bboxes: torch.Tensor, person_mask: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = bboxes.unbind(dim=-1)
        w = (x2 - x1).clamp(min=0.0)
        h = (y2 - y1).clamp(min=0.0)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        area = w * h

        # motion of center across frames
        dcx = torch.zeros_like(cx)
        dcy = torch.zeros_like(cy)
        dcx[:, 1:] = cx[:, 1:] - cx[:, :-1]
        dcy[:, 1:] = cy[:, 1:] - cy[:, :-1]
        disp = (dcx.square() + dcy.square()).sqrt()  # ||Δp||

        # speed/acceleration magnitudes (1st/2nd order temporal dynamics)
        speed = disp
        accel = torch.zeros_like(speed)
        accel[:, 1:] = speed[:, 1:] - speed[:, :-1]
        accel = accel.abs()

        feats = torch.stack([x1, y1, x2, y2, cx, cy, w, h, area, dcx, dcy, disp, speed, accel], dim=-1)
        out = self.mlp(feats)
        out = out.masked_fill(~person_mask.unsqueeze(-1), 0.0)
        return out
