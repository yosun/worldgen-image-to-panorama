"""Project a perspective photo while avoiding work outside its output mask.

This follows WorldGen's pinned ``map_image_to_pano`` algorithm, but finds the
preserved region before nearest-neighbor hole filling. Pixels outside that
region are zeroed by the upstream algorithm, so they do not need filling.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from worldgen.utils.general_utils import (
    batch_nearest_dot,
    fill_mask_from_contour,
    pano_unit_rays,
    resize_img_and_rays,
)


def map_perspective_image(predictions, *, device, map_h=1024, map_w=2048, nn_batch=256):
    """Return WorldGen's conditioning panorama and mask for perspective rays."""
    rays = predictions["rays"]
    rgb = predictions["rgb"].float()
    if rays.ndim != 3 or rays.shape[-1] != 3 or rgb.shape != rays.shape:
        raise ValueError("Input image and rays must have matching H x W x 3 shapes.")
    height, width = rays.shape[:2]
    if height < 2 or width < 2:
        raise ValueError("Input image must be at least 2 pixels wide and high.")

    # Upstream rounds angular coverage down to pixels. For very extreme aspect
    # ratios, that can produce a zero-sized interpolation and an opaque error.
    center_row = rays[height // 2]
    center_column = rays[:, width // 2]
    horizontal_fov = (torch.atan2(center_row[-1, 0], center_row[-1, 2]) -
                      torch.atan2(center_row[0, 0], center_row[0, 2])).abs()
    vertical_fov = (torch.asin(center_column[-1, 1].clamp(-1, 1)) -
                    torch.asin(center_column[0, 1].clamp(-1, 1))).abs()
    if int(horizontal_fov * (map_w / (2 * torch.pi))) < 1 or int(vertical_fov * (map_h / torch.pi)) < 1:
        raise ValueError("Input aspect ratio is too extreme to project at the panorama resolution.")

    rgb, rays = resize_img_and_rays(rgb, rays, map_h, map_w)
    colors = rgb.reshape(-1, 3)
    directions = rays.reshape(-1, 3)
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    u = (torch.atan2(x, z) / torch.pi + 1) / 2
    v = torch.asin(y.clamp(-1, 1)) / torch.pi + 0.5
    u_pix = (u * (map_w - 1)).long()
    v_pix = (v * (map_h - 1)).long()

    panorama = torch.zeros((map_h, map_w, 3), dtype=rgb.dtype, device=rgb.device)
    panorama[v_pix, u_pix] = colors
    hits = torch.zeros((map_h, map_w), dtype=torch.bool, device=rgb.device)
    hits[v_pix, u_pix] = True
    expanded = F.max_pool2d(hits.unsqueeze(0).float(), kernel_size=3, stride=1, padding=1).squeeze(0)
    valid = fill_mask_from_contour(expanded).to(device).bool()
    if not valid.any():
        raise ValueError("Could not project the input photo into the panorama.")

    # Upstream fills every non-hit pixel and then zeros all pixels outside
    # ``valid``. Restricting queries to the intersection gives the same result.
    holes = valid & ~hits
    if holes.any():
        queries = pano_unit_rays(map_h, map_w, device)[holes]
        indices = batch_nearest_dot(directions, queries, nn_batch)
        panorama[holes] = colors[indices]
    panorama *= valid[..., None]
    image = Image.fromarray(panorama.clamp(0, 255).cpu().numpy().astype(np.uint8))
    mask = Image.fromarray(((~valid).cpu().numpy() * 255).astype(np.uint8))
    return image, mask
