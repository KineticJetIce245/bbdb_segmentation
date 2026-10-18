"""Pure view-transform calculations for the segmentation canvas."""
from __future__ import annotations

import math

from .settings import MAX_VIEW_PIXELS, MAX_ZOOM, MIN_ZOOM


def fit_scale(canvas_w: int, canvas_h: int, disp_w: int, disp_h: int) -> float:
    return max(min(canvas_w / disp_w, canvas_h / disp_h), 1e-3)


def clamp_pan(pan: float, canvas: int, image: int) -> float:
    """Keep the image on screen; centre it when it is smaller than the canvas."""
    if image <= canvas:
        return 0.0
    limit = (image - canvas) / 2.0
    return max(-limit, min(limit, pan))


def max_zoom(canvas_w: int, canvas_h: int, disp_w: int, disp_h: int) -> float:
    """Largest zoom relative to fit that respects MAX_VIEW_PIXELS."""
    fit = fit_scale(canvas_w, canvas_h, disp_w, disp_h)
    pixel_limit = math.sqrt(MAX_VIEW_PIXELS / max(disp_w * disp_h, 1)) / fit
    return max(MIN_ZOOM, min(MAX_ZOOM, max(1.0, pixel_limit)))

def visible_bounds(
    canvas_w: int,
    canvas_h: int,
    out_w: int,
    out_h: int,
    offset: tuple[float, float],
) -> tuple[int, int, int, int]:
    """Intersection of the displayed image and the canvas, in output pixels."""
    offset_x, offset_y = offset
    return (
        max(0, int(math.floor(-offset_x))),
        max(0, int(math.floor(-offset_y))),
        min(out_w, int(math.ceil(canvas_w - offset_x))),
        min(out_h, int(math.ceil(canvas_h - offset_y))),
    )
