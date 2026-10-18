"""Rectangular and freehand lasso ROI selection methods."""

from __future__ import annotations

import tkinter as tk

from .roi import Roi, base_to_display, roi_at_point, roi_screen_points, roi_screen_rect

class RoiMixin:

    def active_roi(self) -> Roi | None:
        for roi in self.rois:
            if roi.roi_id == self.active_roi_id:
                return roi
        return None

    def clear_roi_items(self) -> None:
        if self.lasso_item is not None:
            try:
                self.canvas.delete(self.lasso_item)
            except tk.TclError:
                pass
            self.lasso_item = None
        if self.roi_drag_item is not None:
            try:
                self.canvas.delete(self.roi_drag_item)
            except tk.TclError:
                pass
            self.roi_drag_item = None
        for item in self.roi_items:
            try:
                self.canvas.delete(item)
            except tk.TclError:
                pass
        self.roi_items.clear()
        self.roi_photos.clear()

    def _base_to_display(self, row: int, col: int) -> tuple[int, int]:
        assert self.base_img is not None
        height, width = self.base_img.shape
        return base_to_display(row, col, height, width, self.flip, self.rotation)

    def _roi_screen_rect(self, roi: Roi) -> tuple[float, float, float, float] | None:
        if self.base_img is None or self.display_scale <= 0:
            return None
        height, width = self.base_img.shape
        return roi_screen_rect(
            roi,
            height,
            width,
            self.flip,
            self.rotation,
            self.display_scale,
            self.display_offset,
        )

    def render_rois(self) -> None:
        if self.base_img is None:
            return
        self.clear_roi_items()
        for roi in self.rois:
            rect = self._roi_screen_rect(roi)
            if rect is None:
                continue
            x0, y0, x1, y1 = rect
            active = roi.roi_id == self.active_roi_id
            colour = "#00e5ff" if active else "#ffd400"
            width = 3 if active else 2
            self.roi_items.extend(self._roi_outline_items(roi, colour, width))
            self.roi_items.append(
                self.canvas.create_text(
                    x0 + 4, y0 + 2, text=f"R{roi.roi_id}", anchor="nw",
                    fill=colour, font=("TkDefaultFont", 9, "bold"),
                )
            )
        for item in self.roi_items:
            self.canvas.tag_raise(item)
        if self.cursor_id is not None:
            self.canvas.tag_raise(self.cursor_id)

    def _roi_outline_items(self, roi: Roi, colour: str, width: int) -> list[int]:
        assert self.base_img is not None
        if roi.shape == "brush":
            outline = self.brush_outline_item(roi, colour)
            if outline is None:
                return []
            item, photo = outline
            self.roi_photos.append(photo)
            return [item]
        if roi.shape == "polygon":
            points = roi_screen_points(
                roi,
                *self.base_img.shape,
                self.flip,
                self.rotation,
                self.display_scale,
                self.display_offset,
            )
            if len(points) < 3:
                return []
            coords = [coord for point in points for coord in point]
            coords.extend(points[0])
            return [
                self.canvas.create_line(
                    *coords, fill=colour, width=width, dash=(6, 4),
                    joinstyle="round",
                )
            ]
        rect = self._roi_screen_rect(roi)
        if rect is None:
            return []
        return [
            self.canvas.create_rectangle(
                *rect, outline=colour, width=width, dash=(6, 4),
            )
        ]

    def _canvas_to_base_clamped(self, x: float, y: float) -> tuple[int, int] | None:
        if self.base_img is None or self.display_scale <= 0:
            return None
        if self.disp_shape[0] <= 0 or self.disp_shape[1] <= 0:
            return None
        offset_x, offset_y = self.display_offset
        right = offset_x + self.disp_shape[1] * self.display_scale
        bottom = offset_y + self.disp_shape[0] * self.display_scale
        x = min(max(x, offset_x), right - 1)
        y = min(max(y, offset_y), bottom - 1)
        return self.canvas_to_base(int(x), int(y))

    def start_roi_select(self, event: tk.Event) -> None:
        self.flush_slider_updates()
        self.canvas.focus_set()
        self.roi_dragging = True
        self.roi_drag_start = (event.x, event.y)
        self.roi_drag_moved = False
        self.on_leave(None)
        self.canvas.configure(cursor="tcross")
        if self.roi_drag_item is not None:
            self.canvas.delete(self.roi_drag_item)
        self.roi_drag_item = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#00e5ff", width=2, dash=(5, 4),
        )

    def drag_roi_select(self, event: tk.Event) -> None:
        if not self.roi_dragging or self.roi_drag_start is None:
            return
        start_x, start_y = self.roi_drag_start
        if abs(event.x - start_x) > 3 or abs(event.y - start_y) > 3:
            self.roi_drag_moved = True
        if self.roi_drag_item is not None:
            self.canvas.coords(self.roi_drag_item, start_x, start_y, event.x, event.y)

    def _handle_roi_click(self, x: float, y: float) -> None:
        point = self._canvas_to_base_clamped(x, y)
        if point is None:
            return
        row, col = point
        hit = roi_at_point(self.rois, row, col)
        if hit is not None:
            if hit.roi_id != self.active_roi_id:
                self.activate_roi(hit.roi_id)
            return
        if self.active_roi_id is not None:
            self.deactivate_roi()

    def end_roi_select(self, event: tk.Event) -> None:
        if not self.roi_dragging:
            return
        start = self.roi_drag_start or (event.x, event.y)
        moved = self.roi_drag_moved
        self.roi_dragging = False
        self.roi_drag_start = None
        self.roi_drag_moved = False
        if self.roi_drag_item is not None:
            self.canvas.delete(self.roi_drag_item)
            self.roi_drag_item = None
        self.canvas.configure(cursor="crosshair" if self.tool == "fill" else "")

        if not moved:
            self._handle_roi_click(event.x, event.y)
            return

        start_point = self._canvas_to_base_clamped(*start)
        end_point = self._canvas_to_base_clamped(event.x, event.y)
        if start_point is None or end_point is None:
            self.render_rois()
            return
        row0, row1 = sorted((start_point[0], end_point[0]))
        col0, col1 = sorted((start_point[1], end_point[1]))
        if row1 - row0 < 2 or col1 - col0 < 2:
            self._handle_roi_click(event.x, event.y)
            return
        self.create_roi(row0, col0, row1 + 1, col1 + 1)

    def _register_roi(self, roi: Roi, render: bool = True) -> None:
        self.next_roi_id += 1
        self.rois.append(roi)
        self.active_roi_id = roi.roi_id
        self.log_op(
            "roi_create",
            roi=roi,
            threshold=self._threshold_payload(
                roi, (self.threshold_min, self.threshold_max, bool(self.invert_var.get()))
            ),
        )
        if render:
            self.render_rois()
        self.update_status()

    def create_roi(self, r0: int, c0: int, r1: int, c1: int) -> None:
        self.flush_slider_updates()
        roi = Roi(
            self.next_roi_id,
            int(r0), int(c0), int(r1), int(c1),
            self.threshold_min,
            self.threshold_max,
            bool(self.invert_var.get()),
        )
        self._register_roi(roi)

    def activate_roi(self, roi_id: int) -> None:
        self.flush_slider_updates()
        roi = next((item for item in self.rois if item.roi_id == roi_id), None)
        if roi is None or self.active_roi_id == roi.roi_id:
            return
        self.active_roi_id = roi.roi_id
        self._load_threshold_scope()
        self.log_op(
            "roi_activate",
            roi=roi,
            threshold=self._threshold_payload(roi, (self.threshold_min, self.threshold_max, bool(self.invert_var.get()))),
        )
        self.render_rois()
        self.update_status()

    def deactivate_roi(self) -> None:
        self.flush_slider_updates()
        roi = self.active_roi()
        if roi is None:
            return
        self.active_roi_id = None
        self._load_threshold_scope()
        self.log_op(
            "roi_deactivate",
            roi=roi,
            threshold=self._threshold_payload(roi, (self.threshold_min, self.threshold_max, bool(self.invert_var.get()))),
        )
        self.render_rois()
        self.update_status()

    def delete_active_roi(self) -> None:
        self.flush_slider_updates()
        roi = self.active_roi()
        if roi is None:
            return
        self.rois.remove(roi)
        self.active_roi_id = None
        self._load_threshold_scope()
        self.log_op(
            "roi_delete",
            roi=roi,
            threshold=self._threshold_payload(roi, (self.threshold_min, self.threshold_max, bool(self.invert_var.get()))),
        )
        self.render_rois()
        self.update_status()

    # --- rendering ---------------------------------------------------------- #

