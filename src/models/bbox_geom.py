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


def compute_spatial_edge_features(
    bboxes: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute edge features for spatial (intra-frame) edges.

    Args:
        bboxes: (E, 4) normalized bboxes for source and destination nodes.
                Actually we pass src_bboxes and dst_bboxes separately.
        src_idx, dst_idx: not used directly here; we take pre-gathered bboxes.

    Returns:
        edge_feat: (E, 6) [delta_cx, delta_cy, delta_w, delta_h, center_dist, iou]
    """
    # This function takes pre-gathered bboxes
    raise NotImplementedError("Use compute_spatial_edge_features_from_bboxes instead")


def compute_spatial_edge_features_from_bboxes(
    src_bboxes: torch.Tensor,
    dst_bboxes: torch.Tensor,
) -> torch.Tensor:
    """Compute spatial edge features between pairs of persons.

    Args:
        src_bboxes: (E, 4) [x1, y1, x2, y2] normalized
        dst_bboxes: (E, 4) [x1, y1, x2, y2] normalized

    Returns:
        edge_feat: (E, 6)
    """
    sx1, sy1, sx2, sy2 = src_bboxes.unbind(-1)
    dx1, dy1, dx2, dy2 = dst_bboxes.unbind(-1)

    scx = (sx1 + sx2) * 0.5
    scy = (sy1 + sy2) * 0.5
    dcx = (dx1 + dx2) * 0.5
    dcy = (dy1 + dy2) * 0.5

    delta_cx = dcx - scx
    delta_cy = dcy - scy
    center_dist = (delta_cx.square() + delta_cy.square()).sqrt()

    sw = (sx2 - sx1).clamp(min=1e-6)
    sh = (sy2 - sy1).clamp(min=1e-6)
    dw = (dx2 - dx1).clamp(min=1e-6)
    dh = (dy2 - dy1).clamp(min=1e-6)
    delta_w = dw - sw
    delta_h = dh - sh

    # IoU
    inter_x1 = torch.max(sx1, dx1)
    inter_y1 = torch.max(sy1, dy1)
    inter_x2 = torch.min(sx2, dx2)
    inter_y2 = torch.min(sy2, dy2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h
    s_area = sw * sh
    d_area = dw * dh
    union = s_area + d_area - inter_area
    iou = inter_area / union.clamp(min=1e-6)

    return torch.stack([delta_cx, delta_cy, delta_w, delta_h, center_dist, iou], dim=-1)


def compute_temporal_edge_features_from_bboxes(
    curr_bboxes: torch.Tensor,
    next_bboxes: torch.Tensor,
) -> torch.Tensor:
    """Compute temporal edge features for same person across frames.

    Args:
        curr_bboxes: (E, 4) [x1, y1, x2, y2] normalized, current frame
        next_bboxes: (E, 4) [x1, y1, x2, y2] normalized, next frame

    Returns:
        edge_feat: (E, 4) [delta_cx, delta_cy, delta_area, speed]
    """
    cx1 = (curr_bboxes[:, 0] + curr_bboxes[:, 2]) * 0.5
    cy1 = (curr_bboxes[:, 1] + curr_bboxes[:, 3]) * 0.5
    cx2 = (next_bboxes[:, 0] + next_bboxes[:, 2]) * 0.5
    cy2 = (next_bboxes[:, 1] + next_bboxes[:, 3]) * 0.5

    delta_cx = cx2 - cx1
    delta_cy = cy2 - cy1
    speed = (delta_cx.square() + delta_cy.square()).sqrt()

    area1 = ((curr_bboxes[:, 2] - curr_bboxes[:, 0]) * (curr_bboxes[:, 3] - curr_bboxes[:, 1])).clamp(min=1e-6)
    area2 = ((next_bboxes[:, 2] - next_bboxes[:, 0]) * (next_bboxes[:, 3] - next_bboxes[:, 1])).clamp(min=1e-6)
    delta_area = area2 - area1

    return torch.stack([delta_cx, delta_cy, delta_area, speed], dim=-1)
