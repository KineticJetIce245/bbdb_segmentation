"""Brush-shaped selection gesture and blue-to-ROI transition."""
from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image, ImageFilter, ImageTk

from .image_ops import NEAREST
from .roi import Roi
from .transforms import apply_transform
from .view_math import visible_bounds


class BrushSelectMixin:
    def start_brush_select(self, event: tk.Event) -> None:
        if self.base_img is None or self.base_mask_img is None:
            return
        self.flush_slider_updates()
        self.canvas.focus_set()
        self.cancel_brush_preview()
        self._clear_brush_fill()
        self.brush_select_dragging = True
        self.brush_select_last = None
        self.brush_select_items = []

        active = self.active_roi()
        if active is not None and active.shape == "brush" and active.mask is not None:
            self.brush_select_target_id = active.roi_id
            self.brush_select_mask = active.mask.copy()
            self.brush_select_points = list(active.points or [])
        else:
            self.brush_select_target_id = None
            self.brush_select_mask = np.zeros(self.base_img.shape, dtype=bool)
            self.brush_select_points = []
        self._add_brush_dab(event.x, event.y)

    def drag_brush_select(self, event: tk.Event) -> None:
        if not self.brush_select_dragging:
            return
        point = self.canvas_to_base(event.x, event.y)
        if point is None:
            return
        if self.brush_select_last is None:
            self.brush_select_last = point
        for row, col in self._brush_line(self.brush_select_last, point):
            self._add_brush_dab_at(row, col)
        self.brush_select_last = point

    def end_brush_select(self, _event: tk.Event | None = None) -> None:
        if not self.brush_select_dragging:
            return
        self.brush_select_dragging = False
        self._clear_brush_fill()
        mask = self.brush_select_mask
        points = self.brush_select_points
        target_id = self.brush_select_target_id
        self.brush_select_mask = None
        self.brush_select_points = []
        self.brush_select_last = None
        self.brush_select_target_id = None
        if mask is None or not mask.any():
            return
        target = self._commit_brush_selection(mask, points, target_id)
        self.show_brush_preview(target)
        self.update_status()
        self.brush_select_after_id = self.after(200, self._finish_brush_preview)

    def _commit_brush_selection(
        self,
        mask: np.ndarray,
        points: list[tuple[int, int]],
        target_id: int | None,
    ) -> Roi:
        rows, cols = np.where(mask)
        target = next(
            (roi for roi in self.rois if roi.roi_id == target_id),
            None,
        )
        if target is not None and target.shape == "brush":
            target.mask = mask
            target.points = points
            target.r0, target.c0 = int(rows.min()), int(cols.min())
            target.r1, target.c1 = int(rows.max()) + 1, int(cols.max()) + 1
            self.active_roi_id = target.roi_id
            self.log_op(
                "roi_extend",
                roi=target,
                threshold=self._threshold_payload(
                    target,
                    (self.threshold_min, self.threshold_max, bool(self.invert_var.get())),
                ),
            )
            return target
        target = Roi(
            self.next_roi_id,
            int(rows.min()),
            int(cols.min()),
            int(rows.max()) + 1,
            int(cols.max()) + 1,
            self.threshold_min,
            self.threshold_max,
            bool(self.invert_var.get()),
            shape="brush",
            points=points,
            mask=mask,
        )
        self._register_roi(target, render=False)
        return target

    def reset_brush_select(self, render: bool = True) -> None:
        self.brush_select_dragging = False
        self._clear_brush_fill()
        self.brush_select_mask = None
        self.brush_select_points = []
        self.brush_select_last = None
        self.brush_select_target_id = None
        self.cancel_brush_preview()
        if render and self.base_img is not None:
            self.render_rois()

    def _clear_brush_fill(self) -> None:
        for item in self.brush_select_items:
            try:
                self.canvas.delete(item)
            except tk.TclError:
                pass
        self.brush_select_items = []

    def _add_brush_dab(self, x: int, y: int) -> None:
        point = self.canvas_to_base(x, y)
        if point is None:
            return
        if self.brush_select_last is None:
            self.brush_select_last = point
        for row, col in self._brush_line(self.brush_select_last, point):
            self._add_brush_dab_at(row, col)
        self.brush_select_last = point

    def _add_brush_dab_at(self, row: int, col: int) -> None:
        if self.brush_select_mask is None or self.base_img is None:
            return
        height, width = self.base_img.shape
        radius = max(1, int(round(self.brush / max(self.display_scale, 1e-6))))
        r0 = max(0, row - radius)
        r1 = min(height, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(width, col + radius + 1)
        rows = np.arange(r0, r1)[:, None]
        cols = np.arange(c0, c1)[None, :]
        self.brush_select_mask[r0:r1, c0:c1] |= (
            (rows - row) ** 2 + (cols - col) ** 2 <= radius * radius
        )
        if not self.brush_select_points or self.brush_select_points[-1] != (row, col):
            self.brush_select_points.append((row, col))

        disp_row, disp_col = self._base_to_display(row, col)
        offset_x, offset_y = self.display_offset
        x = offset_x + disp_col * self.display_scale
        y = offset_y + disp_row * self.display_scale
        radius_screen = max(1, int(self.brush))
        self.brush_select_items.append(
            self.canvas.create_oval(
                x - radius_screen, y - radius_screen,
                x + radius_screen, y + radius_screen,
                fill="#009dff", outline="", stipple="gray25",
            )
        )

    def _brush_line(
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

    def brush_outline_item(
        self, roi: Roi, colour: str
    ) -> tuple[int, ImageTk.PhotoImage] | None:
        """Rasterize a brush ROI boundary in the current viewport."""
        if (
            self.base_img is None
            or roi.mask is None
            or self.display_scale <= 0
        ):
            return None
        disp_mask = apply_transform(
            roi.mask.astype(np.uint8), self.flip, self.rotation
        )
        out_w = max(1, int(self.disp_shape[1] * self.display_scale))
        out_h = max(1, int(self.disp_shape[0] * self.display_scale))
        offset_x, offset_y = self.display_offset
        canvas_w, canvas_h = self.canvas_size()
        left, top, right, bottom = visible_bounds(
            canvas_w, canvas_h, out_w, out_h, (offset_x, offset_y)
        )
        if right <= left or bottom <= top:
            return None
        source = Image.fromarray(disp_mask * 255, "L")
        stencil = source.transform(
            (right - left, bottom - top),
            Image.Transform.AFFINE,
            (1.0 / self.display_scale, 0.0, left / self.display_scale,
             0.0, 1.0 / self.display_scale, top / self.display_scale),
            resample=NEAREST,
        )
        edge = stencil.filter(ImageFilter.FIND_EDGES).point(
            lambda value: 255 if value else 0
        )
        rgba = Image.new("RGBA", edge.size, (0, 0, 0, 0))
        rgba.paste(Image.new("RGBA", edge.size, colour), (0, 0), edge)
        photo = ImageTk.PhotoImage(rgba)
        item = self.canvas.create_image(
            offset_x + left, offset_y + top, anchor="nw", image=photo
        )
        self.canvas.tag_raise(item)
        return item, photo

    def show_brush_preview(self, roi: Roi) -> None:
        self.cancel_brush_preview()
        outline = self.brush_outline_item(roi, "#00b7ff")
        if outline is None:
            return
        self.brush_preview_item, self.brush_preview_photo = outline
        self.canvas.tag_raise(self.brush_preview_item)

    def cancel_brush_preview(self) -> None:
        if self.brush_select_after_id is not None:
            try:
                self.after_cancel(self.brush_select_after_id)
            except tk.TclError:
                pass
            self.brush_select_after_id = None
        if self.brush_preview_item is not None:
            try:
                self.canvas.delete(self.brush_preview_item)
            except tk.TclError:
                pass
            self.brush_preview_item = None
        self.brush_preview_photo = None

    def _finish_brush_preview(self) -> None:
        self.brush_select_after_id = None
        self.cancel_brush_preview()
        self.render_rois()
