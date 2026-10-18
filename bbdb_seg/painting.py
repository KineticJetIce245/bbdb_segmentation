"""Mask painting, tools, undo and view transforms."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from .transforms import decode_flip, encode_flip, normalize_rotation
from .image_ops import mask_diff as diff_masks
from .settings import CONTROL_MASK, SHIFT_MASK, UNDO_LIMIT

class PaintMixin:

    def stamp(self, r0: int, c0: int, r1: int, c1: int, value: int) -> None:
        if self.base_mask_img is None:
            return
        radius = max(1, int(round(self.brush / max(self.display_scale, 1e-6))))
        targets = [(self.base_mask_img, value)]
        footprint = getattr(self, "stroke_footprint", None)
        if footprint is not None:
            targets.append((footprint, 255))   # everything the brush touched
        for image, fill in targets:
            draw = ImageDraw.Draw(image)
            draw.line([(c0, r0), (c1, r1)], fill=fill, width=2 * radius)
            for row, col in ((r0, c0), (r1, c1)):
                draw.ellipse([col - radius, row - radius, col + radius, row + radius], fill=fill)

    def on_press(self, event: tk.Event, right: bool) -> None:
        self.flush_slider_updates()
        self.canvas.focus_set()
        if event.state & SHIFT_MASK:
            if right:
                self.start_lasso_select(event)
            else:
                self.start_roi_select(event)
            return
        if event.state & CONTROL_MASK:
            self.start_pan(event)
            return
        if self.tool == "fill":
            self.bucket_fill(event, 0 if right else 255)
            return
        if self.tool == "select":
            if right:
                self.start_brush_erase(event)
            else:
                self.start_brush_select(event)
            return
        self.last_pt = None
        self.reset_smoothing()
        self.stroke_pushed = False
        self.stroke_before = np.asarray(self.base_mask_img).copy()
        self.stroke_footprint = Image.new("L", self.base_mask_img.size, 0)
        self.stroke_op = "erase" if (right or self.tool == "erase") else "draw"
        self.stroke_roi_id = self.active_roi_id
        self.continue_stroke(event, right or self.tool == "erase")

    def on_drag(self, event: tk.Event, right: bool) -> None:
        if self.brush_erase_dragging:
            self.drag_brush_erase(event)
            return
        if self.brush_select_dragging:
            self.drag_brush_select(event)
            return
        if self.lasso_dragging:
            self.drag_lasso_select(event)
            return
        if self.roi_dragging:
            self.drag_roi_select(event)
            return
        if event.state & SHIFT_MASK:
            if right:
                self.drag_lasso_select(event)
            else:
                self.drag_roi_select(event)
            return
        if self.panning:
            self.pan_drag(event)
            return
        if event.state & CONTROL_MASK:
            self.start_pan(event)
            return
        if self.tool != "fill":
            self.continue_stroke(event, right or self.tool == "erase")

    def continue_stroke(self, event: tk.Event, erase: bool) -> None:
        # The cursor ring follows the raw pointer; the stroke follows the
        # (lag-free, speed-adaptive) smoothed position.
        self.update_cursor(event.x, event.y, erase)
        sx, sy = self.smooth_screen(event.x, event.y, getattr(event, "time", None))
        point = self.canvas_to_base(int(round(sx)), int(round(sy)))
        if point is None:
            self.last_pt = None
            self.preview_last = None
            return
        if not self.stroke_pushed:
            self.push_undo("stroke")
            self.stroke_pushed = True
        value = 0 if erase else 255
        if self.last_pt is None:
            self.stamp(point[0], point[1], point[0], point[1], value)
        else:
            self.stamp(self.last_pt[0], self.last_pt[1], point[0], point[1], value)
        self.last_pt = point
        self.mask_dirty = True
        # Instant feedback: vector preview on the canvas; the raster overlay is
        # rendered once when the stroke ends.
        self.draw_stroke_preview(sx, sy, erase)

    def end_stroke(self, _event: tk.Event | None = None) -> None:
        if self.brush_erase_dragging:
            if _event is not None:
                self.end_brush_erase(_event)
            return
        if self.brush_select_dragging:
            if _event is not None:
                self.end_brush_select(_event)
            return
        if self.lasso_dragging:
            if _event is not None:
                self.end_lasso_select(_event)
            return
        if self.roi_dragging:
            if _event is not None:
                self.end_roi_select(_event)
            return
        if self.panning:
            self.end_pan(_event)
            return
        self.flush_slider_updates()
        raw_screen = getattr(self, "smooth_raw_screen", None)
        raw_end = self.canvas_to_base(*raw_screen) if raw_screen is not None else None
        if self.last_pt is not None and raw_end is not None and raw_end != self.last_pt:
            # finish the smoothed line where the pointer actually stopped
            value = 0 if self.stroke_op == "erase" else 255
            self.stamp(self.last_pt[0], self.last_pt[1], raw_end[0], raw_end[1], value)
        self.last_pt = None
        self.clear_stroke_preview()
        self.reset_smoothing()
        before = self.stroke_before
        self.stroke_before = None
        if before is not None:
            if self.stroke_op == "draw":
                painted = np.asarray(self.base_mask_img)
                kept = self.exclude_background(painted, strict=True)
                if not np.array_equal(kept, painted):
                    self.base_mask_img = Image.fromarray(kept)
            self.render_overlay()   # one raster render per stroke
            after = np.asarray(self.base_mask_img)
            footprint = getattr(self, "stroke_footprint", None)
            self.record_hand_edit(
                before, after, self.stroke_op,
                footprint=np.asarray(footprint) > 0 if footprint is not None else None,
            )
            added, removed, bbox = diff_masks(before, after)
            if bbox is not None:
                roi = next(
                    (item for item in self.rois if item.roi_id == self.stroke_roi_id),
                    None,
                )
                self.log_op(
                    self.stroke_op,
                    roi=roi,
                    added=added,
                    removed=removed,
                    bbox=bbox,
                )
        self.stroke_footprint = None
        self.stroke_roi_id = None
        if self.mask_dirty:
            self.mask_source = "edited"
            self.update_status()

    def bucket_fill(self, event: tk.Event, value: int) -> None:
        """Set the 4-connected mask region under the cursor to ``value``."""
        point = self.canvas_to_base(event.x, event.y)
        if point is None or self.base_mask_img is None:
            return
        mask = np.array(self.base_mask_img)
        row, col = point
        if mask[row, col] == value:
            return
        labels, _ = ndimage.label(mask == mask[row, col])
        self.push_undo("fill")
        before = mask.copy()
        mask[labels == labels[row, col]] = value
        if value:
            mask = self.exclude_background(mask, strict=True)
        self.record_hand_edit(before, mask, "draw" if value else "erase")
        self.base_mask_img = Image.fromarray(mask)
        added, removed, bbox = diff_masks(before, mask)
        if bbox is not None:
            self.log_op(
                "fill",
                roi=self.active_roi(),
                added=added,
                removed=removed,
                bbox=bbox,
                extra={"value": int(value)},
            )
        self.mask_source = "edited"
        self.mark_edited()

    def push_undo(self, action: str) -> None:
        if self.base_mask_img is None:
            return
        self.undo_stack.append(self.base_mask_img.copy())
        del self.undo_stack[:-UNDO_LIMIT]
        self.last_action = action

    def undo(self) -> None:
        self.flush_slider_updates()
        if not self.undo_stack:
            self.status_var.set("Nothing to undo")
            return
        before = np.asarray(self.base_mask_img).copy()
        self.base_mask_img = self.undo_stack.pop()
        self.on_mask_restored()
        added, removed, bbox = diff_masks(before, np.asarray(self.base_mask_img))
        if bbox is not None:
            self.log_op(
                "undo",
                roi=self.active_roi(),
                added=added,
                removed=removed,
                bbox=bbox,
            )
        self.last_action = None
        self.mask_source = "edited"
        self.mark_edited()

    def mark_edited(self) -> None:
        self.mask_dirty = True
        self.render_overlay()
        self.update_status()

    def clear_mask(self) -> None:
        self.flush_slider_updates()
        if self.base_mask_img is None:
            return
        before = np.asarray(self.base_mask_img).copy()
        self.push_undo("clear")
        self.base_mask_img.paste(0, (0, 0, self.base_mask_img.width, self.base_mask_img.height))
        self.reset_hand_edits()
        added, removed, bbox = diff_masks(before, np.asarray(self.base_mask_img))
        if bbox is not None:
            self.log_op(
                "clear",
                roi=self.active_roi(),
                added=added,
                removed=removed,
                bbox=bbox,
            )
        self.mask_source = "edited"
        self.mark_edited()

    def apply_morphology(self, grow: bool) -> None:
        """Dilate (grow) or erode (shrink) the mask by one pixel step."""
        self.flush_slider_updates()
        if self.base_mask_img is None:
            return
        kernel = ImageFilter.MaxFilter(3) if grow else ImageFilter.MinFilter(3)
        before = np.asarray(self.base_mask_img).copy()
        self.push_undo("morphology")
        self.base_mask_img = self.base_mask_img.filter(kernel)
        added, removed, bbox = diff_masks(before, np.asarray(self.base_mask_img))
        if bbox is not None:
            self.log_op(
                "dilate" if grow else "erode",
                roi=self.active_roi(),
                added=added,
                removed=removed,
                bbox=bbox,
            )
        self.mask_source = "edited"
        self.mark_edited()

    def erode_mask(self) -> None:
        self.apply_morphology(grow=False)

    def dilate_mask(self) -> None:
        self.apply_morphology(grow=True)

    # --- transforms --------------------------------------------------------- #

    def set_transform(self, flip: str, rotation: int) -> None:
        self.flush_slider_updates()
        self.flip = flip
        self.rotation = normalize_rotation(rotation)
        self.records.at[self.index, "flip"] = self.flip
        self.records.at[self.index, "rotation"] = self.rotation
        self.log_op("transform", extra={"flip": self.flip, "rotation": self.rotation})
        self.render_all()
        self.schedule_csv_save()

    def rotate(self, delta: int) -> None:
        self.set_transform(self.flip, self.rotation + delta)

    def flip_h(self) -> None:
        lr_flip, up_flip = decode_flip(self.flip)
        self.set_transform(encode_flip(not lr_flip, up_flip), self.rotation)

    def flip_v(self) -> None:
        lr_flip, up_flip = decode_flip(self.flip)
        self.set_transform(encode_flip(lr_flip, not up_flip), self.rotation)

    # --- brush / mode ------------------------------------------------------- #


    # --- navigation / keys -------------------------------------------------- #

