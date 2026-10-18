"""GUI glue for background exclusion and the Clean (k) button."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .clean import background_mask, clean_mask
from .image_ops import mask_diff as diff_masks


class CleanMixin:

    def _init_clean_state(self) -> None:
        self.bg_cache_key: int | None = None
        self.bg_cache: np.ndarray | None = None

    def current_background(self) -> np.ndarray | None:
        """Background mask of the current working image (cached per section)."""
        if self.base_img is None:
            return None
        key = id(self.base_img)
        if self.bg_cache_key != key or self.bg_cache is None:
            self.bg_cache = background_mask(self.base_img)
            self.bg_cache_key = key
        return self.bg_cache

    def exclude_background(self, mask_arr: np.ndarray) -> np.ndarray:
        """Zero background pixels in a 0/255 mask when "Exclude background" is on."""
        if not bool(self.exclude_bg_var.get()):
            return mask_arr
        bg = self.current_background()
        if bg is None or bg.shape != mask_arr.shape:
            return mask_arr
        out = np.array(mask_arr, copy=True)
        out[bg] = 0
        return out

    def on_exclude_bg_toggle(self) -> None:
        self.flush_slider_updates()
        if bool(self.exclude_bg_var.get()) and self.base_mask_img is not None:
            before = np.asarray(self.base_mask_img).copy()
            after = self.exclude_background(before)
            added, removed, bbox = diff_masks(before, after)
            if bbox is not None:
                self.push_undo("exclude_background")
                self.base_mask_img = Image.fromarray(after)
                self.log_op("exclude_background", removed=removed, bbox=bbox)
                self.mask_source = "edited"
                self.mark_edited()
        self.update_status()

    def clean_current(self) -> None:
        """Background exclusion + edge snapping + small-island removal (undoable).

        With an active ROI, only pixels inside the ROI change.
        """
        self.flush_slider_updates()
        if self.base_img is None or self.base_mask_img is None:
            return
        before = np.asarray(self.base_mask_img).copy()
        cleaned = clean_mask(self.base_img, before > 0, background=self.current_background())
        after = (cleaned.astype(np.uint8) * 255)
        roi = self.active_roi()
        if roi is not None:
            inside = roi.mask_array(*before.shape)
            after = np.where(inside, after, before).astype(np.uint8)
        added, removed, bbox = diff_masks(before, after)
        if bbox is None:
            self.status_var.set("Clean: nothing to change")
            return
        self.push_undo("clean")
        self.base_mask_img = Image.fromarray(after)
        self.log_op("clean", roi=roi, added=added, removed=removed, bbox=bbox)
        self.status_var.set(f"Clean: -{removed} / +{added} px (Ctrl+Z to undo)")
        self.mask_source = "edited"
        self.mark_edited()
