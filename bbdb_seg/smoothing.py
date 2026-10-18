"""Stroke and lasso smoothing.

* Brush strokes use a stabiliser: the drawn point follows the pointer with an
  exponential moving average, which irons out hand jitter. When the stroke ends
  the line is completed to where the pointer actually stopped.
* Lasso outlines are smoothed with Chaikin corner cutting before they become a
  polygon ROI.
"""
from __future__ import annotations

from .settings import LASSO_SMOOTH_ITERATIONS, SMOOTH_ALPHA


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


class StrokeSmoothMixin:
    """Needs ``self.smooth_var`` (BooleanVar-like with .get())."""

    def reset_smoothing(self) -> None:
        self.smooth_pos: tuple[float, float] | None = None
        self.smooth_raw_last: tuple[int, int] | None = None

    def smoothing_on(self) -> bool:
        var = getattr(self, "smooth_var", None)
        return bool(var.get()) if var is not None else False

    def smooth_point(self, point: tuple[int, int], alpha: float = SMOOTH_ALPHA) -> tuple[int, int]:
        """Return the stabilised position for a new raw pointer position."""
        self.smooth_raw_last = point
        if not self.smoothing_on():
            self.smooth_pos = (float(point[0]), float(point[1]))
            return point
        if getattr(self, "smooth_pos", None) is None:
            self.smooth_pos = (float(point[0]), float(point[1]))
        else:
            r, c = self.smooth_pos
            self.smooth_pos = (r + alpha * (point[0] - r), c + alpha * (point[1] - c))
        return (int(round(self.smooth_pos[0])), int(round(self.smooth_pos[1])))

    def smooth_outline(self, points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return chaikin(points) if self.smoothing_on() else points
