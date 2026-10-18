"""Threshold brush: apply the current threshold only inside the brush disk."""
from __future__ import annotations

import numpy as np
from PIL import Image


class ThresholdBrushMixin:
    def start_threshold_stroke(self, event) -> None:
        if self.base_img is None or self.base_mask_img is None:
            return
        self.flush_slider_updates()
        self.canvas.focus_set()
        self.last_pt = None
        self.stroke_pushed = False
        self.stroke_before = np.asarray(self.base_mask_img).copy()
        self.stroke_op = "threshold_brush"
        self.stroke_roi_id = self.active_roi_id
        self.threshold_stroke_values = np.asarray(self.threshold_mask())
        roi = self.active_roi()
        self.threshold_stroke_roi_mask = (
            roi.mask_array(*self.base_img.shape) if roi is not None else None
        )
        self.threshold_stroke_active = True
        self.continue_threshold_stroke(event)

    def continue_threshold_stroke(self, event) -> None:
        point = self.canvas_to_base(event.x, event.y)
        self.update_cursor(event.x, event.y, False)
        if point is None:
            self.last_pt = None
            return
        if not self.stroke_pushed:
            self.push_undo("threshold_brush")
            self.stroke_pushed = True
        first_point = self.last_pt is None
        if first_point:
            samples = [point]
        else:
            samples = self._threshold_line(self.last_pt, point)
        for row, col in samples:
            self._threshold_stamp(row, col)
        self.last_pt = point
        self.mask_dirty = True
        self.show_brush_overlay(event.x, event.y)
        if first_point:
            self.render_overlay()
        else:
            self.schedule_slider_update("overlay")

    def _threshold_line(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> list[tuple[int, int]]:
        radius = max(1, int(round(self.brush / max(self.display_scale, 1e-6))))
        step = max(1, radius // 2)
        delta_row = end[0] - start[0]
        delta_col = end[1] - start[1]
        count = max(abs(delta_row), abs(delta_col)) // step
        if count <= 0:
            return [end]
        return [
            (
                int(round(start[0] + delta_row * index / count)),
                int(round(start[1] + delta_col * index / count)),
            )
            for index in range(1, count + 1)
        ]

    def _threshold_stamp(self, row: int, col: int) -> bool:
        if (
            self.base_mask_img is None
            or self.threshold_stroke_values is None
        ):
            return False
        mask = np.asarray(self.base_mask_img)
        height, width = mask.shape
        radius = max(1, int(round(self.brush / max(self.display_scale, 1e-6))))
        r0 = max(0, row - radius)
        r1 = min(height, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(width, col + radius + 1)
        if r0 >= r1 or c0 >= c1:
            return False
        rows = np.arange(r0, r1)[:, None]
        cols = np.arange(c0, c1)[None, :]
        disk = (rows - row) ** 2 + (cols - col) ** 2 <= radius * radius
        if self.threshold_stroke_roi_mask is not None:
            disk &= self.threshold_stroke_roi_mask[r0:r1, c0:c1]
        if not disk.any():
            return False
        after = mask.copy()
        values = self.threshold_stroke_values[r0:r1, c0:c1]
        after[r0:r1, c0:c1][disk] = values[disk]
        if np.array_equal(after, mask):
            return False
        self.base_mask_img = Image.fromarray(after)
        return True
