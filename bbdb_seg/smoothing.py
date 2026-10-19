"""Responsive stroke smoothing, live stroke preview and lasso smoothing.

* Brush strokes are smoothed with the 1-euro filter (Casiez et al., CHI 2012):
  an adaptive low-pass filter whose cutoff rises with pointer speed. Slow,
  careful movements are smoothed strongly (that is where hand jitter shows);
  fast movements are barely filtered, so the line does not trail the pointer.
  When the stroke ends, the line is completed to where the pointer stopped.
* While dragging, the mask's real effect is shown at once: only the small
  canvas rectangle around each new stroke segment is redrawn (image + red
  overlay) instead of the whole overlay, which is rendered once on release.
  This keeps painting responsive however large the image or zoom level.
* Lasso outlines are smoothed with Chaikin corner cutting.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageTk

from .image_ops import BILINEAR, NEAREST
from .transforms import apply_transform
from .view_math import visible_bounds
from .settings import (
    BLOCK_ALPHA_GAIN,
    BLOCK_COLOUR,
    LASSO_SMOOTH_ITERATIONS,
    ONE_EURO_BETA,
    ONE_EURO_D_CUTOFF,
    ONE_EURO_MIN_CUTOFF,
)

PREVIEW_MAX_PATCHES = 400   # long strokes are folded into one overlay render


def chaikin(points: list[tuple[int, int]], iterations: int = LASSO_SMOOTH_ITERATIONS,
            closed: bool = True) -> list[tuple[int, int]]:
    """Chaikin corner cutting; returns integer points, duplicates removed."""
    pts = [(float(r), float(c)) for r, c in points]
    if len(pts) < 3 or iterations <= 0:
        return list(points)
    for _ in range(iterations):
        out: list[tuple[float, float]] = []
        n = len(pts)
        last = n if closed else n - 1
        if not closed:
            out.append(pts[0])
        for i in range(last):
            (r0, c0), (r1, c1) = pts[i], pts[(i + 1) % n]
            out.append((0.75 * r0 + 0.25 * r1, 0.75 * c0 + 0.25 * c1))
            out.append((0.25 * r0 + 0.75 * r1, 0.25 * c0 + 0.75 * c1))
        if not closed:
            out.append(pts[-1])
        pts = out
    result: list[tuple[int, int]] = []
    for r, c in pts:
        point = (int(round(r)), int(round(c)))
        if not result or point != result[-1]:
            result.append(point)
    return result


class OneEuroFilter:
    """1-euro filter for one coordinate (units: px, time in seconds)."""

    def __init__(self, min_cutoff: float = ONE_EURO_MIN_CUTOFF, beta: float = ONE_EURO_BETA,
                 d_cutoff: float = ONE_EURO_D_CUTOFF) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: float | None = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, dt: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x
        dt = max(dt, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


class StrokeSmoothMixin:
    """Needs ``self.smooth_var`` (BooleanVar-like) and, for previews, ``self.canvas``."""

    def reset_smoothing(self) -> None:
        self.smooth_fx = OneEuroFilter()
        self.smooth_fy = OneEuroFilter()
        self.smooth_t_prev: int | None = None
        self.smooth_raw_screen: tuple[int, int] | None = None
        self.preview_last: tuple[float, float] | None = None

    def smoothing_on(self) -> bool:
        var = getattr(self, "smooth_var", None)
        return bool(var.get()) if var is not None else False

    def smooth_screen(self, x: int, y: int, t_ms: int | None = None) -> tuple[float, float]:
        """Filtered screen position for a raw pointer event (Tk event.time in ms)."""
        self.smooth_raw_screen = (x, y)
        if not self.smoothing_on():
            return float(x), float(y)
        if t_ms is None or self.smooth_t_prev is None or t_ms <= self.smooth_t_prev:
            dt = 1.0 / 120.0
        else:
            dt = (t_ms - self.smooth_t_prev) / 1000.0
        if t_ms is not None:
            self.smooth_t_prev = t_ms
        return self.smooth_fx(float(x), dt), self.smooth_fy(float(y), dt)

    def smooth_outline(self, points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return chaikin(points) if self.smoothing_on() else points

    # --- live preview --------------------------------------------------------- #

    def draw_stroke_preview(self, x: float, y: float, erase: bool) -> None:
        """Show the stroke's real effect on the mask immediately.

        Instead of re-rendering the whole overlay on every motion event, only the
        small canvas rectangle around the newest stroke segment is redrawn (image
        + red mask overlay, exactly as the full render would show it) and placed
        on top as a patch. Patches are dropped when the full overlay is rendered
        on release.
        """
        if not hasattr(self, "stroke_preview_items"):
            self.stroke_preview_items = []
        x0, y0 = self.preview_last if self.preview_last is not None else (x, y)
        pad = int(self.brush) + 3
        box = (int(min(x0, x)) - pad, int(min(y0, y)) - pad,
               int(max(x0, x)) + pad + 1, int(max(y0, y)) + pad + 1)
        self.preview_last = (x, y)
        patch = self.composite_region(*box)
        if patch is None:
            return
        image, place_x, place_y = patch
        photo = ImageTk.PhotoImage(image)
        item = self.canvas.create_image(place_x, place_y, anchor="nw", image=photo)
        self.stroke_preview_items.append(item)
        if not hasattr(self, "stroke_preview_photos"):
            self.stroke_preview_photos = []
        self.stroke_preview_photos.append(photo)   # keep a reference (Tk would drop it)
        for roi_item in getattr(self, "roi_items", []):
            self.canvas.tag_raise(roi_item)
        if getattr(self, "cursor_id", None) is not None:
            self.canvas.tag_raise(self.cursor_id)
        if len(self.stroke_preview_items) > PREVIEW_MAX_PATCHES:
            # very long stroke: fold the patches into one overlay render
            self.render_overlay()
            self._drop_preview_patches()

    def _drop_preview_patches(self) -> None:
        for item in getattr(self, "stroke_preview_items", []):
            self.canvas.delete(item)
        self.stroke_preview_items = []
        self.stroke_preview_photos = []

    def clear_stroke_preview(self) -> None:
        self._drop_preview_patches()
        self.preview_last = None

    def _display_gray(self) -> np.ndarray:
        """Contrast-adjusted, flipped/rotated gray image (cached)."""
        key = (id(self.base_img), self.flip, self.rotation, self.contrast_min, self.contrast_max)
        if getattr(self, "_display_gray_key", None) != key:
            self._display_gray_cache = self.apply_contrast(
                apply_transform(self.base_img, self.flip, self.rotation)
            )
            self._display_gray_key = key
        return self._display_gray_cache

    def composite_region(self, cx0: int, cy0: int, cx1: int, cy1: int):
        """Image + mask overlay for canvas rectangle (cx0, cy0)-(cx1, cy1).

        Samples exactly like render_background/render_overlay, so the patch
        matches the full render pixel for pixel. Returns (RGB image, x, y) or None.
        """
        if self.base_img is None or self.base_mask_img is None or self.display_scale <= 0:
            return None
        scale = self.display_scale
        disp_h, disp_w = self.disp_shape
        out_w, out_h = max(1, int(disp_w * scale)), max(1, int(disp_h * scale))
        offset_x, offset_y = self.display_offset
        canvas_w, canvas_h = self.canvas_size()
        left, top, right, bottom = visible_bounds(canvas_w, canvas_h, out_w, out_h, (offset_x, offset_y))
        # canvas -> output-pixel coordinates, clipped to what is visible
        ox0 = max(left, int(math.floor(cx0 - offset_x)))
        oy0 = max(top, int(math.floor(cy0 - offset_y)))
        ox1 = min(right, int(math.ceil(cx1 - offset_x)))
        oy1 = min(bottom, int(math.ceil(cy1 - offset_y)))
        if ox1 <= ox0 or oy1 <= oy0:
            return None
        # source crop (with a margin for bilinear sampling); sampling positions
        # are unchanged because the affine offset is shifted by the crop origin
        sx0 = max(0, int(math.floor(ox0 / scale)) - 2)
        sy0 = max(0, int(math.floor(oy0 / scale)) - 2)
        sx1 = min(disp_w, int(math.ceil(ox1 / scale)) + 3)
        sy1 = min(disp_h, int(math.ceil(oy1 / scale)) + 3)
        size = (ox1 - ox0, oy1 - oy0)
        matrix = (1.0 / scale, 0.0, ox0 / scale - sx0, 0.0, 1.0 / scale, oy0 / scale - sy0)
        gray = Image.fromarray(np.ascontiguousarray(self._display_gray()[sy0:sy1, sx0:sx1]), "L")
        rgb = gray.transform(size, Image.Transform.AFFINE, matrix, resample=BILINEAR).convert("RGB")
        mask = apply_transform(np.asarray(self.base_mask_img), self.flip, self.rotation)
        stencil = Image.fromarray(np.ascontiguousarray(mask[sy0:sy1, sx0:sx1]), "L").transform(
            size, Image.Transform.AFFINE, matrix, resample=NEAREST)
        disp_block = self._display_block()
        if disp_block is not None and disp_block[sy0:sy1, sx0:sx1].any():
            block_stencil = Image.fromarray(
                np.ascontiguousarray(disp_block[sy0:sy1, sx0:sx1]), "L").transform(
                size, Image.Transform.AFFINE, matrix, resample=NEAREST)
            alpha = min(255, int(self.overlay_alpha * BLOCK_ALPHA_GAIN))
            rgb.paste(BLOCK_COLOUR, (0, 0), block_stencil.point(lambda v, a=alpha: a if v else 0))
        red_alpha = int(self.overlay_alpha)
        rgb.paste((255, 40, 40), (0, 0), stencil.point(lambda v, a=red_alpha: a if v else 0))
        return rgb, offset_x + ox0, offset_y + oy0

    def _display_block(self) -> np.ndarray | None:
        """Blue (rejected) layer as a flipped/rotated 0/255 array (cached)."""
        block = getattr(self, "block_mask", None)
        if block is None or self.base_mask_img is None or \
                block.shape != (self.base_mask_img.height, self.base_mask_img.width):
            return None
        key = (id(block), self.flip, self.rotation)
        if getattr(self, "_display_block_key", None) != key:
            self._display_block_cache = apply_transform(
                block.astype(np.uint8) * 255, self.flip, self.rotation)
            self._display_block_key = key
        return self._display_block_cache
