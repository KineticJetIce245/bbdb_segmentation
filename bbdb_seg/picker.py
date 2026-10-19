"""Colour picker tool (p): scribble over an area to include or exclude every
region of that gray level.

Left-drag includes, right-drag excludes: every continuous block of that gray
level touched by the scribble changes as one solid piece (small holes inside a
block are filled). An active selection limits the change to the selection.

Excluded blocks become a "rejected" layer shown in transparent dark blue. Blue
pixels are never cortex: the threshold sliders, brush, fill, Clean and include
picks all leave them out (like background). Ctrl+Z undoes a pick (blue
included); "Clear blue" removes the whole layer. The layer is kept per
section for the session, not saved to disk.
"""
from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .clean import colour_region
from .image_ops import mask_diff as diff_masks
from .settings import PICK_SIGMA

PICK_INCLUDE_COLOUR = "#ffff40"
PICK_EXCLUDE_COLOUR = "#40c4ff"


class PickMixin:

    def _init_pick_state(self) -> None:
        self.pick_dragging = False
        self.pick_points: list[tuple[int, int]] = []
        self.pick_include = False
        self.pick_line_id: int | None = None
        self.pick_screen: list[float] = []

    # --- rejected (blue) layer ------------------------------------------------ #

    def _section_key(self) -> str | None:
        try:
            return str(self.records.at[self.index, "base"])
        except Exception:
            return None

    def load_block_layer(self) -> None:
        """On section load: restore this section's blue layer (if any)."""
        if not hasattr(self, "block_layers"):
            self.block_layers = {}
        self.block_mask = self.block_layers.get(self._section_key())

    def _store_block_layer(self) -> None:
        if not hasattr(self, "block_layers"):
            self.block_layers = {}
        key = self._section_key()
        if key is not None:
            self.block_layers[key] = self.block_mask

    def set_block_layer(self, block: np.ndarray | None) -> None:
        self.block_mask = block if block is not None and block.any() else None
        self._store_block_layer()

    def clear_block_layer(self) -> None:
        self.flush_slider_updates()
        if getattr(self, "block_mask", None) is None:
            self.status_var.set("No blue (rejected) regions to clear")
            return
        self.push_undo("clear_blue")
        removed = int(self.block_mask.sum())
        self.set_block_layer(None)
        self.log_op("clear_blue", extra={"cleared_px": removed})
        self.render_overlay()
        self.status_var.set(f"Cleared {removed} blue (rejected) px (Ctrl+Z to undo)")

    def _pick_smoothed(self) -> np.ndarray:
        """Smoothed copy of the section image, computed once per section."""
        key = id(self.base_img)
        if getattr(self, "_pick_smooth_key", None) != key:
            self._pick_smooth_cache = ndimage.gaussian_filter(
                np.asarray(self.base_img, dtype=np.float32), PICK_SIGMA)
            self._pick_smooth_key = key
        return self._pick_smooth_cache

    def toggle_pick(self) -> None:
        self.set_tool("draw" if self.tool == "pick" else "pick")

    def start_pick(self, event: tk.Event, right: bool) -> None:
        self.canvas.focus_set()
        self.pick_dragging = True
        self.pick_include = not right
        self.pick_points = []
        self.pick_screen = []
        self._cancel_pick_line()
        self.drag_pick(event)

    def drag_pick(self, event: tk.Event) -> None:
        if not self.pick_dragging:
            return
        point = self.canvas_to_base(event.x, event.y)
        if point is not None:
            self.pick_points.append(point)
        self.pick_screen += [event.x, event.y]
        colour = PICK_INCLUDE_COLOUR if self.pick_include else PICK_EXCLUDE_COLOUR
        coords = self.pick_screen if len(self.pick_screen) >= 4 else self.pick_screen * 2
        if self.pick_line_id is None:
            self.pick_line_id = self.canvas.create_line(*coords, fill=colour, width=2,
                                                        capstyle="round", joinstyle="round")
        else:
            self.canvas.coords(self.pick_line_id, *coords)
        if getattr(self, "cursor_id", None) is not None:
            self.canvas.tag_raise(self.cursor_id)

    def _cancel_pick_line(self) -> None:
        if self.pick_line_id is not None:
            self.canvas.delete(self.pick_line_id)
            self.pick_line_id = None

    def end_pick(self, _event: tk.Event | None = None) -> None:
        if not self.pick_dragging:
            return
        self.pick_dragging = False
        self._cancel_pick_line()
        points, self.pick_points = self.pick_points, []
        self.pick_screen = []
        if not points or self.base_img is None or self.base_mask_img is None:
            return
        self.flush_slider_updates()
        height, width = self.base_img.shape
        scribble_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(scribble_img)
        if len(points) == 1:
            draw.point([(points[0][1], points[0][0])], fill=255)
        else:
            draw.line([(c, r) for r, c in points], fill=255, width=1)
        scribble = np.asarray(scribble_img) > 0
        region, low, high = colour_region(
            self.base_img, scribble,
            background=self.current_background(strict=True),
            smoothed=self._pick_smoothed(),
        )
        roi = self.active_roi()
        if roi is not None:
            region &= roi.mask_array(height, width)
        before = np.asarray(self.base_mask_img).copy()
        after = before.copy()
        old_block = getattr(self, "block_mask", None)
        block = None if old_block is None else old_block.copy()
        if self.pick_include:
            after[region] = 255
            after = self.exclude_background(after, strict=True)   # never into blue/background
        else:
            after[region] = 0
            block = region.copy() if block is None else (block | region)
        added, removed, bbox = diff_masks(before, after)
        verb = "include" if self.pick_include else "exclude"
        block_grew = block is not None and (old_block is None or (block & ~old_block).any())
        if bbox is None and not block_grew:
            why = "already blue" if not self.pick_include and region.any() else "nothing to change"
            self.status_var.set(f"Colour pick ({verb}, gray {low:.0f}-{high:.0f}): {why}")
            return
        self.push_undo("pick")
        self.set_block_layer(block)
        self.base_mask_img = Image.fromarray(after)
        self.record_hand_edit(before, after, "draw" if self.pick_include else "erase")
        self.log_op("pick", roi=roi, added=added, removed=removed, bbox=bbox,
                    extra={"mode": verb, "gray_range": [round(low, 1), round(high, 1)],
                           "region_px": int(region.sum()),
                           "blue_px": int(block.sum()) if block is not None else 0})
        self.mask_source = "edited"
        self.mark_edited()
        self.status_var.set(
            f"Colour pick: {verb} gray {low:.0f}-{high:.0f}: {int(region.sum())} px block"
            f"{' marked blue' if not self.pick_include else ''}, mask -{removed} / +{added} px"
            " (Ctrl+Z to undo)"
        )
