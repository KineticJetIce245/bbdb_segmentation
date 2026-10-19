"""Canvas rendering, view transforms, zoom and pan methods."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

from .transforms import apply_transform
from .image_ops import BILINEAR, NEAREST, apply_contrast as contrast_array
from .settings import (
    BLOCK_ALPHA_GAIN,
    BLOCK_COLOUR,
    MIN_ZOOM,
    UNET_PREDICTION_COLOUR,
)
from .view_math import (
    clamp_pan as clamp_pan_value,
    fit_scale,
    max_zoom as max_zoom_value,
    visible_bounds,
)

class RenderMixin:

    def render_all(self) -> None:
        if self.base_img is None:
            return
        self.render_background()
        self.render_overlay()
        self.render_rois()
        self.update_status()

    def canvas_size(self) -> tuple[int, int]:
        return max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100)

    @staticmethod
    def clamp_pan(pan: float, canvas: int, image: int) -> float:
        return clamp_pan_value(pan, canvas, image)

    def max_zoom(self, disp_w: int, disp_h: int) -> float:
        canvas_w, canvas_h = self.canvas_size()
        return max_zoom_value(canvas_w, canvas_h, disp_w, disp_h)

    def render_background(self) -> None:
        if self.base_img is None or self.base_index is None:
            return
        disp_img = apply_transform(self.base_img, self.flip, self.rotation)
        self.disp_index = apply_transform(self.base_index, self.flip, self.rotation)
        disp_h, disp_w = disp_img.shape
        self.disp_shape = (disp_h, disp_w)

        canvas_w, canvas_h = self.canvas_size()
        fit = fit_scale(canvas_w, canvas_h, disp_w, disp_h)
        self.zoom = max(MIN_ZOOM, min(self.zoom, self.max_zoom(disp_w, disp_h)))
        scale = fit * self.zoom
        out_w = max(1, int(disp_w * scale))
        out_h = max(1, int(disp_h * scale))
        self.display_scale = scale

        self.pan_x = self.clamp_pan(self.pan_x, canvas_w, out_w)
        self.pan_y = self.clamp_pan(self.pan_y, canvas_h, out_h)
        offset_x = (canvas_w - out_w) / 2.0 + self.pan_x
        offset_y = (canvas_h - out_h) / 2.0 + self.pan_y
        self.display_offset = (offset_x, offset_y)

        disp_img = self.apply_contrast(disp_img)
        left, top, right, bottom = visible_bounds(
            canvas_w, canvas_h, out_w, out_h, (offset_x, offset_y)
        )
        if right <= left or bottom <= top:
            if self.bg_id is not None:
                self.canvas.itemconfigure(self.bg_id, state="hidden")
            return
        source = Image.fromarray(disp_img, "L")
        visible = source.transform(
            (right - left, bottom - top),
            Image.Transform.AFFINE,
            (1.0 / scale, 0.0, left / scale, 0.0, 1.0 / scale, top / scale),
            resample=BILINEAR,
        ).convert("RGB")
        self.bg_photo = ImageTk.PhotoImage(visible)
        place_x = offset_x + left
        place_y = offset_y + top
        # kept so the overlay can be blended onto it and shown as one opaque
        # image (Tk is slow at drawing semi-transparent images over others)
        self.bg_visible = visible
        self.bg_place = (place_x, place_y)
        if self.bg_id is None:
            self.bg_id = self.canvas.create_image(
                place_x, place_y, anchor="nw", image=self.bg_photo
            )
        else:
            self.canvas.itemconfigure(self.bg_id, image=self.bg_photo, state="normal")
            self.canvas.coords(self.bg_id, place_x, place_y)

    def render_overlay(self) -> None:
        if self.base_mask_img is None:
            return
        mask_np = np.asarray(self.base_mask_img)
        disp_mask = apply_transform(mask_np, self.flip, self.rotation)
        out_w = max(1, int(self.disp_shape[1] * self.display_scale))
        out_h = max(1, int(self.disp_shape[0] * self.display_scale))
        offset_x, offset_y = self.display_offset
        canvas_w, canvas_h = self.canvas_size()

        # Only build the part of the overlay that can be seen. At high zoom the
        # full RGBA image can be tens of times larger than the canvas.
        left, top, right, bottom = visible_bounds(
            canvas_w, canvas_h, out_w, out_h, (offset_x, offset_y)
        )
        if right <= left or bottom <= top:
            if self.ov_id is not None:
                self.canvas.itemconfigure(self.ov_id, state="hidden")
            return

        matrix = (1.0 / self.display_scale, 0.0, left / self.display_scale,
                  0.0, 1.0 / self.display_scale, top / self.display_scale)
        source = Image.fromarray(disp_mask, "L")
        stencil = source.transform(
            (right - left, bottom - top), Image.Transform.AFFINE, matrix, resample=NEAREST,
        )
        block_stencil = None
        disp_block = self._display_block() if hasattr(self, "_display_block") else None
        if disp_block is not None:
            block_stencil = Image.fromarray(disp_block, "L").transform(
                (right - left, bottom - top), Image.Transform.AFFINE, matrix, resample=NEAREST,
            )
        pred_stencil = None
        separate = getattr(self, "prediction_separate", None)
        if separate is not None and separate():
            disp_pred = apply_transform(np.asarray(self.prediction_img), self.flip, self.rotation)
            pred_stencil = Image.fromarray(disp_pred, "L").transform(
                (right - left, bottom - top), Image.Transform.AFFINE, matrix, resample=NEAREST,
            )
        self._set_overlay_photo(stencil, offset_x + left, offset_y + top,
                                block_stencil, pred_stencil)

    def _set_overlay_photo(self, stencil: Image.Image, x: float, y: float,
                           block_stencil: Image.Image | None = None,
                           pred_stencil: Image.Image | None = None) -> None:
        # A prediction used as the working mask is yellow until the first edit;
        # a separate prediction (Run UNet on a saved section) is yellow on top
        # of the red manual mask.
        base_colour = (
            UNET_PREDICTION_COLOUR
            if (getattr(self, "mask_source", "") == "unet"
                and getattr(self, "prediction_yellow", False)
                and pred_stencil is None)
            else (255, 40, 40)
        )
        alpha = int(self.overlay_alpha)
        background = getattr(self, "bg_visible", None)
        if (background is not None and background.size == stencil.size
                and getattr(self, "bg_place", None) == (x, y)):
            # Fast path: blend colours onto the cached background and show one
            # opaque RGB image on top of it.
            out = background.copy()
            if block_stencil is not None:
                block_alpha = min(255, int(self.overlay_alpha * BLOCK_ALPHA_GAIN))
                out.paste(BLOCK_COLOUR, (0, 0),
                          block_stencil.point(lambda v, a=block_alpha: a if v else 0))
            out.paste(base_colour, (0, 0), stencil.point(lambda v, a=alpha: a if v else 0))
            if pred_stencil is not None:
                out.paste(UNET_PREDICTION_COLOUR, (0, 0),
                          pred_stencil.point(lambda v, a=alpha: a if v else 0))
            self._place_overlay_image(out, x, y)
            return
        rgba = Image.new("RGBA", stencil.size, (0, 0, 0, 0))
        if block_stencil is not None:   # rejected regions (colour picker), under the mask
            block_alpha = min(255, int(self.overlay_alpha * BLOCK_ALPHA_GAIN))
            rgba.paste(Image.new("RGBA", stencil.size, (*BLOCK_COLOUR, block_alpha)),
                       (0, 0), block_stencil)
        rgba.paste(
            Image.new("RGBA", stencil.size, (*base_colour, self.overlay_alpha)),
            (0, 0),
            stencil,
        )
        if pred_stencil is not None:
            rgba.paste(
                Image.new("RGBA", stencil.size, (*UNET_PREDICTION_COLOUR, self.overlay_alpha)),
                (0, 0),
                pred_stencil,
            )
        self._place_overlay_image(rgba, x, y)

    def _place_overlay_image(self, image: Image.Image, x: float, y: float) -> None:
        self.ov_image = image
        self.ov_photo = ImageTk.PhotoImage(image)
        if self.ov_id is None:
            self.ov_id = self.canvas.create_image(
                x, y, anchor="nw", image=self.ov_photo
            )
        else:
            self.canvas.itemconfigure(
                self.ov_id, image=self.ov_photo, state="normal"
            )
            self.canvas.coords(self.ov_id, x, y)
        self.canvas.tag_raise(self.ov_id)
        for item in self.roi_items:
            self.canvas.tag_raise(item)
        if self.cursor_id is not None:
            self.canvas.tag_raise(self.cursor_id)

    # --- display controls --------------------------------------------------- #

    def apply_contrast(self, arr: np.ndarray) -> np.ndarray:
        return contrast_array(arr, self.contrast_min, self.contrast_max)

    def on_alpha_change(self, value: str) -> None:
        self.overlay_alpha = int(float(value))
        self.schedule_slider_update("alpha")

    def on_contrast_change(self, _value: str | None = None) -> None:
        self.contrast_min = int(self.min_scale.get())
        self.contrast_max = int(self.max_scale.get())
        self.schedule_slider_update("contrast")

    def auto_contrast(self) -> None:
        """Stretch the window to the current image's actual value range."""
        if self.base_img is None:
            return
        lo = int(self.base_img.min())
        hi = int(self.base_img.max())
        if hi <= lo:
            hi = min(255, lo + 1)
        self.contrast_min, self.contrast_max = lo, hi
        self.min_scale.set(lo)
        self.max_scale.set(hi)
        self.schedule_slider_update("contrast")
        self.flush_slider_updates()

    def reset_contrast(self) -> None:
        self.contrast_min, self.contrast_max = 0, 255
        self.min_scale.set(0)
        self.max_scale.set(255)
        self.schedule_slider_update("contrast")
        self.flush_slider_updates()

    # --- zoom / pan --------------------------------------------------------- #

    def zoom_at(self, x: float, y: float, factor: float) -> None:
        """Zoom by ``factor`` while keeping the image point under (x, y) fixed."""
        if self.base_img is None or self.display_scale <= 0:
            return
        disp_h, disp_w = self.disp_shape
        if disp_w <= 0 or disp_h <= 0:
            return
        new_zoom = max(MIN_ZOOM, min(self.max_zoom(disp_w, disp_h), self.zoom * factor))
        if abs(new_zoom - self.zoom) < 1e-9:
            return

        old_scale = self.display_scale
        old_x, old_y = self.display_offset
        image_x = (x - old_x) / old_scale
        image_y = (y - old_y) / old_scale

        self.zoom = new_zoom
        canvas_w, canvas_h = self.canvas_size()
        fit = fit_scale(canvas_w, canvas_h, disp_w, disp_h)
        new_scale = fit * new_zoom
        out_w = max(1, int(disp_w * new_scale))
        out_h = max(1, int(disp_h * new_scale))
        self.pan_x = x - image_x * new_scale - (canvas_w - out_w) / 2.0
        self.pan_y = y - image_y * new_scale - (canvas_h - out_h) / 2.0
        self.pan_x = self.clamp_pan(self.pan_x, canvas_w, out_w)
        self.pan_y = self.clamp_pan(self.pan_y, canvas_h, out_h)
        self.render_all()

    def zoom_center(self, factor: float) -> None:
        if self.last_mouse is not None:
            x, y = self.last_mouse
        else:
            canvas_w, canvas_h = self.canvas_size()
            x, y = canvas_w / 2.0, canvas_h / 2.0
        self.zoom_at(x, y, factor)

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.render_all()

    # --- painting ----------------------------------------------------------- #

    def canvas_to_base(self, event_x: int, event_y: int) -> tuple[int, int] | None:
        if self.disp_index is None or self.base_img is None:
            return None
        offset_x, offset_y = self.display_offset
        scale = self.display_scale
        disp_x = (event_x - offset_x) / scale
        disp_y = (event_y - offset_y) / scale
        disp_h, disp_w = self.disp_shape
        if disp_x < 0 or disp_y < 0 or disp_x >= disp_w or disp_y >= disp_h:
            return None
        flat = int(self.disp_index[int(disp_y), int(disp_x)])
        base_w = self.base_img.shape[1]
        return flat // base_w, flat % base_w

    def start_pan(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        self.panning = True
        self.pan_last = (event.x, event.y)
        self.last_pt = None
        self.stroke_pushed = False
        self.on_leave(None)
        self.canvas.configure(cursor="fleur")

    def pan_drag(self, event: tk.Event) -> None:
        if not self.panning or self.pan_last is None:
            return
        dx = event.x - self.pan_last[0]
        dy = event.y - self.pan_last[1]
        self.pan_last = (event.x, event.y)
        self.last_mouse = (event.x, event.y)
        if not dx and not dy:
            return

        canvas_w, canvas_h = self.canvas_size()
        disp_h, disp_w = self.disp_shape
        out_w = max(1, int(disp_w * self.display_scale))
        out_h = max(1, int(disp_h * self.display_scale))
        new_x = self.clamp_pan(self.pan_x + dx, canvas_w, out_w)
        new_y = self.clamp_pan(self.pan_y + dy, canvas_h, out_h)
        move_x = new_x - self.pan_x
        move_y = new_y - self.pan_y
        if not move_x and not move_y:
            return

        # Moving the existing canvas items is much cheaper than re-rendering
        # the full background and overlay on every drag event.
        self.pan_x, self.pan_y = new_x, new_y
        self.display_offset = (
            self.display_offset[0] + move_x,
            self.display_offset[1] + move_y,
        )
        self.render_background()
        for item in self.roi_items:
            self.canvas.move(item, move_x, move_y)
        if self.roi_drag_item is not None:
            self.canvas.move(self.roi_drag_item, move_x, move_y)
        # The overlay is viewport-cropped, so it has to be rebuilt at the new
        # offset instead of moving the old cropped image.
        self.render_overlay()

    def end_pan(self, _event: tk.Event | None = None) -> None:
        if not self.panning:
            return
        self.panning = False
        self.pan_last = None
        self.canvas.configure(cursor="crosshair" if self.tool == "fill" else "")
        if self.last_mouse is not None and self.tool != "fill":
            self.update_cursor(self.last_mouse[0], self.last_mouse[1], self.tool == "erase")

    def on_canvas_configure(self, _event: tk.Event) -> None:
        if self.base_img is None:
            return
        if self.resize_after_id is not None:
            self.after_cancel(self.resize_after_id)
        self.resize_after_id = self.after(120, self.render_all)

    # --- saving ------------------------------------------------------------- #

