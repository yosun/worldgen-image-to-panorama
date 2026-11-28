"""Overlap and repair joins when expanding a photo into a panorama."""

import cv2
import numpy as np
from PIL import Image

from projection import perspective_predictions


def _smoothstep(values):
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def make_overlap_mask(mask, overlap=24, feather=24):
    """Erode the conditioned photo; fade the paste inside the kept region.

    Distances are measured in output-panorama pixels. The central source
    pixels remain untouched, while the model redraws the outer source edge.
    """
    keep = np.asarray(mask.convert("L")) < 128
    if not keep.any() or keep.all():
        raise ValueError("Expected a photo region surrounded by a generation mask.")
    distance = cv2.distanceTransform(keep.astype(np.uint8), cv2.DIST_L2, 5)
    # Keep a core even when an extreme aspect ratio makes the source very thin.
    peak = float(distance.max())
    overlap = min(float(overlap), peak * 0.2)
    feather = max(min(float(feather), peak * 0.2), 1.0)
    keep_for_fill = distance > overlap
    alpha = _smoothstep((distance - overlap) / feather)
    return Image.fromarray((~keep_for_fill).astype(np.uint8) * 255), alpha


def composite(original, generated, alpha):
    original = np.asarray(original.convert("RGB"), dtype=np.float32)
    generated = np.asarray(generated.convert("RGB").resize((original.shape[1], original.shape[0]), Image.Resampling.LANCZOS), dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)[..., None]
    return Image.fromarray(np.clip(np.rint(original * alpha + generated * (1.0 - alpha)), 0, 255).astype(np.uint8))


def make_boundary_repair_mask(mask, inner=32, outer=40, feather=12):
    """Mask a band straddling the projected photo boundary for inpainting."""
    keep = np.asarray(mask.convert("L")) < 128
    if not keep.any() or keep.all():
        raise ValueError("Expected a photo region surrounded by a generation mask.")
    inside = cv2.distanceTransform(keep.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((~keep).astype(np.uint8), cv2.DIST_L2, 5)
    # Leave a protected source-photo core even for narrow input shapes.
    inner = min(float(inner), float(inside.max()) * 0.2)
    outer = float(outer)
    signed = inside - outside
    band = (signed <= inner) & (signed >= -outer)
    feather = max(min(float(feather), inner, outer), 0.5)
    alpha = _smoothstep(np.minimum((inner - signed) / feather, (signed + outer) / feather))
    return Image.fromarray(band.astype(np.uint8) * 255), alpha


def soften_wrap(image, band=12):
    """Remove a one-pixel wrap discontinuity with a narrow periodic blend."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    band = min(int(band), pixels.shape[1] // 2)
    if band < 1:
        return image
    left = pixels[:, :band].copy()
    right = pixels[:, -band:][:, ::-1].copy()
    fade = np.linspace(0, 1, band, dtype=np.float32) if band > 1 else np.array([0], dtype=np.float32)
    alpha = (0.5 * (1.0 - _smoothstep(fade)))[None, :, None]
    pixels[:, :band] = left * (1 - alpha) + right * alpha
    pixels[:, -band:] = (right * (1 - alpha) + left * alpha)[:, ::-1]
    return Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8))


def generate_seamless_pano(worldgen, prompt, image, *, repair_boundary=True, repair_wrap=False, fill_fn=None, resize_fn=None, mapper=None):
    """Generate a panorama with a photo-edge overlap and optional wrap repair.

    The optional boundary and wrap passes only replace narrow bands; each
    additional pass increases inference time and may change details there.
    """
    if fill_fn is None:
        from worldgen.pano_gen import gen_pano_fill_image
        fill_fn = gen_pano_fill_image
    if resize_fn is None:
        from worldgen.utils.general_utils import resize_img
        resize_fn = resize_img
    if mapper is None:
        from projection_mapper import map_perspective_image
        mapper = map_perspective_image

    source = resize_fn(image)
    rays = perspective_predictions(None, source, device=worldgen.device)
    condition, mask = mapper(rays, device=worldgen.device)
    fill_mask, source_alpha = make_overlap_mask(mask)
    kwargs = {"prompt": prompt, "height": worldgen.resolution // 2, "width": worldgen.resolution}
    generated = fill_fn(worldgen.pano_gen_model, image=condition, mask=fill_mask, **kwargs)
    panorama = composite(condition, generated, source_alpha)

    if repair_boundary:
        boundary_mask, boundary_alpha = make_boundary_repair_mask(mask)
        replacement = fill_fn(worldgen.pano_gen_model, image=panorama, mask=boundary_mask, **kwargs)
        panorama = composite(panorama, replacement, 1.0 - boundary_alpha)

    if repair_wrap:
        # Move the 360-degree join to the middle so the model has context from
        # both sides, then inpaint only a narrow band and rotate it back.
        data = np.asarray(panorama)
        height, width = data.shape[:2]
        shifted = np.roll(data, width // 2, axis=1)
        center = width // 2
        half_band = min(96, width // 8)
        if half_band >= 2:
            x = np.arange(width, dtype=np.float32)
            distance_to_edge = half_band - np.abs(x - center)
            inpaint = distance_to_edge >= 0
            mask_array = np.broadcast_to(inpaint, (height, width)).astype(np.uint8) * 255
            shifted_image = Image.fromarray(shifted)
            replacement = fill_fn(worldgen.pano_gen_model, image=shifted_image, mask=Image.fromarray(mask_array), **kwargs)
            # Composite only the inpainted band; keep the surrounding pixels.
            feather = max(min(24, half_band // 2), 1)
            generated_alpha = _smoothstep(distance_to_edge / feather)
            repaired = composite(shifted_image, replacement, 1.0 - np.broadcast_to(generated_alpha, (height, width)))
            panorama = Image.fromarray(np.roll(np.asarray(repaired), -(width // 2), axis=1))
    return soften_wrap(panorama)
