"""Responsive stroke smoothing, live stroke preview and lasso smoothing.

* Brush strokes are smoothed with the 1-euro filter (Casiez et al., CHI 2012):
  an adaptive low-pass filter whose cutoff rises with pointer speed. Slow,
  careful movements are smoothed strongly (that is where hand jitter shows);
  fast movements are barely filtered, so the line does not trail the pointer.
  When the stroke ends, the line is completed to where the pointer stopped.
* While dragging, the stroke is previewed as plain canvas line segments, which
  Tk draws immediately; the raster mask overlay is re-rendered once on release.
  This keeps painting responsive however large the image or zoom level.
* Lasso outlines are smoothed with Chaikin corner cutting.
"""
from __future__ import annotations

import math

from .settings import (
    LASSO_SMOOTH_ITERATIONS,
    ONE_EURO_BETA,
    ONE_EURO_D_CUTOFF,
    ONE_EURO_MIN_CUTOFF,
)

PREVIEW_DRAW = "#ff4d4d"
PREVIEW_ERASE = "#36c5ff"


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
        """Draw the stroke segment on the canvas immediately (no raster render)."""
        if not hasattr(self, "stroke_preview_items"):
            self.stroke_preview_items = []
        colour = PREVIEW_ERASE if erase else PREVIEW_DRAW
        width = max(1, 2 * int(self.brush))
        x0, y0 = self.preview_last if self.preview_last is not None else (x, y)
        if (x0, y0) == (x, y):
            r = width / 2.0
            item = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=colour, outline="")
        else:
            item = self.canvas.create_line(x0, y0, x, y, fill=colour, width=width,
                                           capstyle="round", joinstyle="round")
        self.stroke_preview_items.append(item)
        self.preview_last = (x, y)
        if getattr(self, "cursor_id", None) is not None:
            self.canvas.tag_raise(self.cursor_id)

    def clear_stroke_preview(self) -> None:
        for item in getattr(self, "stroke_preview_items", []):
            self.canvas.delete(item)
        self.stroke_preview_items = []
        self.preview_last = None
