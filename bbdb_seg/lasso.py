"""Freehand lasso ROI gesture methods."""
from __future__ import annotations

import tkinter as tk

from .roi import Roi


class LassoMixin:

    def start_lasso_select(self, event: tk.Event) -> None:
        self.flush_slider_updates()
        self.canvas.focus_set()
        self.lasso_dragging = True
        self.lasso_points = [(event.x, event.y)]
        self.lasso_moved = False
        self.on_leave(None)
        self.canvas.configure(cursor="tcross")
        if self.lasso_item is not None:
            self.canvas.delete(self.lasso_item)
        self.lasso_item = self.canvas.create_line(
            event.x, event.y, event.x, event.y,
            fill="#00e5ff", width=2, dash=(5, 4),
        )

    def drag_lasso_select(self, event: tk.Event) -> None:
        if not self.lasso_dragging or not self.lasso_points:
            return
        last_x, last_y = self.lasso_points[-1]
        if abs(event.x - last_x) < 3 and abs(event.y - last_y) < 3:
            return
        self.lasso_points.append((event.x, event.y))
        self.lasso_moved = True
        if self.lasso_item is not None:
            coords = [coord for point in self.lasso_points for coord in point]
            coords.extend(self.lasso_points[0])
            self.canvas.coords(self.lasso_item, *coords)

    def end_lasso_select(self, event: tk.Event) -> None:
        if not self.lasso_dragging:
            return
        self.lasso_dragging = False
        points = list(self.lasso_points)
        moved = self.lasso_moved
        self.lasso_points = []
        self.lasso_moved = False
        if self.lasso_item is not None:
            self.canvas.delete(self.lasso_item)
            self.lasso_item = None
        self.canvas.configure(cursor="crosshair" if self.tool == "fill" else "")

        if not moved or len(points) < 3:
            self._handle_roi_click(event.x, event.y)
            return

        base_points: list[tuple[int, int]] = []
        for x, y in points:
            point = self._canvas_to_base_clamped(x, y)
            if point is None:
                continue
            if not base_points or point != base_points[-1]:
                base_points.append(point)
        if len(base_points) < 3:
            self._handle_roi_click(event.x, event.y)
            return
        self.create_polygon_roi(self.smooth_outline(base_points))

    def create_polygon_roi(self, points: list[tuple[int, int]]) -> None:
        self.flush_slider_updates()
        if self.base_img is None or len(points) < 3:
            return
        height, width = self.base_img.shape
        clean: list[tuple[int, int]] = []
        for row, col in points:
            point = (
                max(0, min(height - 1, int(row))),
                max(0, min(width - 1, int(col))),
            )
            if not clean or point != clean[-1]:
                clean.append(point)
        if len(clean) > 1 and clean[0] == clean[-1]:
            clean.pop()
        if len(clean) < 3:
            return
        rows = [point[0] for point in clean]
        cols = [point[1] for point in clean]
        roi = Roi(
            self.next_roi_id,
            min(rows), min(cols), max(rows) + 1, max(cols) + 1,
            self.threshold_min,
            self.threshold_max,
            bool(self.invert_var.get()),
            shape="polygon",
            points=clean,
        )
        self._register_roi(roi)
