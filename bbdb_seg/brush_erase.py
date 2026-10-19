"""Right-drag erasing for brush-shaped selections."""
from __future__ import annotations

import tkinter as tk

import numpy as np


class BrushEraseMixin:
    def start_brush_erase(self, event: tk.Event) -> None:
        roi = self.active_roi()
        if roi is None or roi.shape != "brush" or roi.mask is None:
            return
        self.push_roi_undo()
        self.brush_select_dragging = False
        self._clear_brush_fill()
        self.cancel_brush_preview()
        self.canvas.focus_set()
        self.brush_erase_dragging = True
        self.brush_erase_roi_id = roi.roi_id
        self.brush_erase_before = roi.mask.copy()
        self.brush_erase_last = None
        self.brush_erase_changed = False
        self._erase_brush_at(event.x, event.y)

    def drag_brush_erase(self, event: tk.Event) -> None:
        if not self.brush_erase_dragging:
            return
        point = self.canvas_to_base(event.x, event.y)
        if point is None:
            return
        if self.brush_erase_last is None:
            self.brush_erase_last = point
        for row, col in self._brush_erase_line(self.brush_erase_last, point):
            self._erase_brush_dab(row, col)
        self.brush_erase_last = point

    def end_brush_erase(self, _event: tk.Event | None = None) -> None:
        if not self.brush_erase_dragging:
            return
        self.brush_erase_dragging = False
        target_id = self.brush_erase_roi_id
        before = self.brush_erase_before
        changed = self.brush_erase_changed
        self.brush_erase_roi_id = None
        self.brush_erase_before = None
        self.brush_erase_last = None
        self.brush_erase_changed = False
        target = next(
            (roi for roi in self.rois if roi.roi_id == target_id),
            None,
        )
        if changed and target is not None and before is not None and target.mask is not None:
            removed = int(np.count_nonzero(before & ~target.mask))
            if removed:
                self.log_op(
                    "roi_erase",
                    roi=target,
                    extra={"removed_selection": removed},
                )
        self.render_rois()
        self.update_status()

    def reset_brush_erase(self) -> None:
        self.brush_erase_dragging = False
        self.brush_erase_roi_id = None
        self.brush_erase_before = None
        self.brush_erase_last = None
        self.brush_erase_changed = False

    def _erase_brush_at(self, x: int, y: int) -> None:
        point = self.canvas_to_base(x, y)
        if point is None:
            return
        if self.brush_erase_last is None:
            self.brush_erase_last = point
        for row, col in self._brush_erase_line(self.brush_erase_last, point):
            self._erase_brush_dab(row, col)
        self.brush_erase_last = point

    def _erase_brush_dab(self, row: int, col: int) -> None:
        target = next(
            (roi for roi in self.rois if roi.roi_id == self.brush_erase_roi_id),
            None,
        )
        if target is None or target.mask is None or self.base_img is None:
            return
        height, width = self.base_img.shape
        radius = max(1, int(round(self.brush / max(self.display_scale, 1e-6))))
        r0 = max(0, row - radius)
        r1 = min(height, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(width, col + radius + 1)
        rows = np.arange(r0, r1)[:, None]
        cols = np.arange(c0, c1)[None, :]
        disk = (rows - row) ** 2 + (cols - col) ** 2 <= radius * radius
        if not target.mask[r0:r1, c0:c1].any():
            return
        target.mask[r0:r1, c0:c1] &= ~disk
        self.brush_erase_changed = True
        if not target.mask.any():
            before = self.brush_erase_before
            removed = int(np.count_nonzero(before)) if before is not None else 0
            self.log_op(
                "roi_delete",
                roi=target,
                extra={"removed_selection": removed},
            )
            self.rois.remove(target)
            self.active_roi_id = None
            self._load_threshold_scope()
            self.render_rois()
            self.update_status()
            return
        rows, cols = np.where(target.mask)
        target.r0, target.c0 = int(rows.min()), int(cols.min())
        target.r1, target.c1 = int(rows.max()) + 1, int(cols.max()) + 1
        self.render_rois()
        self.update_status()

    def _brush_erase_line(
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
