"""ROI crop utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


_BASE_GRID_CACHE: dict[tuple[str, int, int, str], torch.Tensor] = {}


def _get_base_grid(out_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device.type, int(device.index or -1), int(out_size), str(dtype))
    base = _BASE_GRID_CACHE.get(key)
    if base is not None:
        return base
    u = torch.linspace(0, 1, out_size, device=device, dtype=dtype)
    v = torch.linspace(0, 1, out_size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(v, u, indexing="ij")
    base = torch.stack([grid_x, grid_y], dim=-1)
    _BASE_GRID_CACHE[key] = base
    return base


def roi_valid_indices(person_mask: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    valid = person_mask & frame_mask.unsqueeze(-1)
    return valid.nonzero(as_tuple=False)


def roi_crop_from_indices(
    frames: torch.Tensor,
    bboxes: torch.Tensor,
    valid_idx: torch.Tensor,
    out_size: int,
) -> torch.Tensor:
    """Crop ROIs only for the given valid indices."""
    if valid_idx.numel() == 0:
        return frames.new_zeros((0, 3, out_size, out_size))

    base = _get_base_grid(out_size, frames.device, frames.dtype)
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
    return F.grid_sample(frames_sel, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
