"""ROI (Region of Interest) crop utilities.

Extracts per-person image patches from video frames using bbox coordinates,
via grid_sample for differentiable, GPU-native cropping.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def roi_crop_valid_batch(
    frames: torch.Tensor,
    bboxes: torch.Tensor,
    person_mask: torch.Tensor,
    frame_mask: torch.Tensor,
    out_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop person ROIs from frames for all valid (b, t, n) slots.

    Args:
        frames: (B, T, 3, H, W) float
        bboxes: (B, T, N, 4) normalized [x1,y1,x2,y2]
        person_mask: (B, T, N) bool
        frame_mask: (B, T) bool
        out_size: output crop size (square)

    Returns:
        valid_idx: (M, 3) long — indices [b, t, n] for each valid crop
        crops: (M, 3, out_size, out_size) — cropped image patches
    """
    device = frames.device
    valid = person_mask & frame_mask.unsqueeze(-1)
    valid_idx = valid.nonzero(as_tuple=False)
    if valid_idx.numel() == 0:
        return valid_idx, frames.new_zeros((0, 3, out_size, out_size))

    u = torch.linspace(0, 1, out_size, device=device, dtype=frames.dtype)
    v = torch.linspace(0, 1, out_size, device=device, dtype=frames.dtype)
    grid_y, grid_x = torch.meshgrid(v, u, indexing="ij")
    base = torch.stack([grid_x, grid_y], dim=-1)

    b = valid_idx[:, 0]
    t = valid_idx[:, 1]
    n = valid_idx[:, 2]
    boxes = bboxes[b, t, n].to(dtype=frames.dtype)
    frames_sel = frames[b, t]

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    gx = x1[:, None, None] + base[None, :, :, 0] * w[:, None, None]
    gy = y1[:, None, None] + base[None, :, :, 1] * h[:, None, None]
    grid = torch.stack([gx * 2 - 1, gy * 2 - 1], dim=-1)
    crops = F.grid_sample(frames_sel, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return valid_idx, crops
