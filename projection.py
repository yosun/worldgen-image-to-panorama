"""Construct camera rays for an ordinary perspective input photograph."""

import numpy as np
import torch
import torch.nn.functional as F


def perspective_predictions(_depth_model, image, *, device="cuda"):
    """Return the RGB pixels and rays for a 90-degree horizontal field of view."""
    width, height = image.size
    focal_length = width / 2.0
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5 - width / 2.0) / focal_length
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5 - height / 2.0) / focal_length
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    rays = torch.stack((x_grid, y_grid, torch.ones_like(x_grid)), dim=-1)
    rgb = torch.from_numpy(np.array(image, copy=True)).to(device)
    return {"rgb": rgb, "rays": F.normalize(rays, dim=-1)}
